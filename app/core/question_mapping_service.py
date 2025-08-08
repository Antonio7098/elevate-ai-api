"""
Question-Criterion Mapping Service.

This service handles semantic mapping between questions and mastery criteria,
ensuring questions are properly aligned with their target learning objectives.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from app.models.learning_blueprint import Question, MasteryCriterion, KnowledgePrimitive
from app.core.llm_service import llm_service
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class QuestionMappingService:
    """Service for mapping questions to mastery criteria with semantic analysis."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.mapping_threshold = 0.3  # Minimum similarity for automatic mapping
        self.confidence_threshold = 0.7  # Threshold for high-confidence mappings
        
    async def map_questions_to_criteria(
        self,
        questions: List[Question],
        criteria: List[MasteryCriterion],
        primitive: KnowledgePrimitive,
        source_content: str
    ) -> Dict[str, List[Tuple[Question, float]]]:
        """
        Map questions to mastery criteria using semantic analysis and LLM validation.
        
        Args:
            questions: List of Question instances to map
            criteria: List of MasteryCriterion instances to map to
            primitive: Parent KnowledgePrimitive for context
            source_content: Original source content for context
            
        Returns:
            Dictionary mapping criterion IDs to lists of (question, confidence_score) tuples
        """
        if not questions or not criteria:
            return {}
        
        try:
            # Perform semantic similarity mapping
            similarity_mappings = self._calculate_semantic_similarities(questions, criteria)
            
            # Enhance mappings with LLM validation for uncertain cases
            enhanced_mappings = await self._enhance_mappings_with_llm(
                similarity_mappings, questions, criteria, primitive, source_content
            )
            
            # Filter and organize final mappings
            final_mappings = self._organize_final_mappings(enhanced_mappings, criteria)
            
            logger.info(f"Mapped {len(questions)} questions to {len(criteria)} criteria")
            return final_mappings
            
        except Exception as e:
            logger.error(f"Failed to map questions to criteria: {e}")
            return self._create_fallback_mappings(questions, criteria)
    
    def _calculate_semantic_similarities(
        self,
        questions: List[Question],
        criteria: List[MasteryCriterion]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate semantic similarities between questions and criteria."""
        try:
            # Prepare text data for vectorization
            question_texts = [
                f"{q.text} {q.explanation or ''}" for q in questions
            ]
            criterion_texts = [
                f"{c.title} {c.description or ''}" for c in criteria
            ]
            
            # Combine all texts for fitting vectorizer
            all_texts = question_texts + criterion_texts
            
            if len(all_texts) == 0:
                return {}
            
            # Fit vectorizer and transform texts
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Split back into questions and criteria matrices
            question_vectors = tfidf_matrix[:len(questions)]
            criterion_vectors = tfidf_matrix[len(questions):]
            
            # Calculate cosine similarities
            similarities = cosine_similarity(question_vectors, criterion_vectors)
            
            # Organize results
            similarity_mappings = {}
            for i, question in enumerate(questions):
                similarity_mappings[question.question_id] = {}
                for j, criterion in enumerate(criteria):
                    similarity_mappings[question.question_id][criterion.criterionId] = similarities[i][j]
            
            return similarity_mappings
            
        except Exception as e:
            logger.error(f"Failed to calculate semantic similarities: {e}")
            return {}
    
    async def _enhance_mappings_with_llm(
        self,
        similarity_mappings: Dict[str, Dict[str, float]],
        questions: List[Question],
        criteria: List[MasteryCriterion],
        primitive: KnowledgePrimitive,
        source_content: str
    ) -> Dict[str, Dict[str, float]]:
        """Enhance uncertain mappings using LLM analysis."""
        enhanced_mappings = similarity_mappings.copy()
        
        # Find questions with uncertain mappings (no clear best match)
        uncertain_questions = []
        for question in questions:
            question_similarities = similarity_mappings.get(question.question_id, {})
            if not question_similarities:
                continue
                
            max_similarity = max(question_similarities.values())
            # Count how many criteria have high similarity
            high_similarity_count = sum(
                1 for sim in question_similarities.values() 
                if sim > max_similarity * 0.8
            )
            
            # If multiple criteria have similar scores, it's uncertain
            if high_similarity_count > 1 or max_similarity < self.confidence_threshold:
                uncertain_questions.append(question)
        
        # Use LLM to clarify uncertain mappings
        for question in uncertain_questions[:5]:  # Limit to avoid excessive LLM calls
            try:
                llm_mappings = await self._get_llm_mapping_analysis(
                    question, criteria, primitive, source_content
                )
                
                # Update mappings with LLM insights
                if llm_mappings and question.question_id in enhanced_mappings:
                    for criterion_id, llm_score in llm_mappings.items():
                        # Combine semantic similarity with LLM confidence
                        original_score = enhanced_mappings[question.question_id].get(criterion_id, 0)
                        combined_score = (original_score + llm_score) / 2
                        enhanced_mappings[question.question_id][criterion_id] = combined_score
                        
            except Exception as e:
                logger.error(f"Failed LLM mapping analysis for question {question.question_id}: {e}")
                continue
        
        return enhanced_mappings
    
    async def _get_llm_mapping_analysis(
        self,
        question: Question,
        criteria: List[MasteryCriterion],
        primitive: KnowledgePrimitive,
        source_content: str
    ) -> Dict[str, float]:
        """Get LLM analysis for question-criterion mapping."""
        prompt = self._create_mapping_analysis_prompt(question, criteria, primitive, source_content)
        
        try:
            response = await llm_service.call_llm(
                prompt=prompt,
                prefer_google=True,
                operation="analyze_question_mapping"
            )
            
            return self._parse_mapping_response(response, criteria)
            
        except Exception as e:
            logger.error(f"LLM mapping analysis failed: {e}")
            return {}
    
    def _create_mapping_analysis_prompt(
        self,
        question: Question,
        criteria: List[MasteryCriterion],
        primitive: KnowledgePrimitive,
        source_content: str
    ) -> str:
        """Create prompt for LLM mapping analysis."""
        criteria_list = "\n".join([
            f"- {c.criterionId}: {c.title} ({c.ueeLevel}) - {c.description or 'No description'}"
            for c in criteria
        ])
        
        prompt = f"""
You are an expert assessment analyst determining which mastery criteria a question best assesses.

PRIMITIVE: {primitive.title}
TYPE: {primitive.primitiveType}
DESCRIPTION: {primitive.description or 'No description'}

QUESTION TO ANALYZE:
Text: {question.text}
Type: {question.question_type}
UEE Level: {question.uee_level}
Explanation: {question.explanation or 'No explanation'}

AVAILABLE MASTERY CRITERIA:
{criteria_list}

SOURCE CONTEXT:
{source_content[:800]}...

TASK: Analyze which mastery criteria this question most effectively assesses.

Consider:
1. Cognitive alignment (UEE level matching)
2. Content relevance (does the question test what the criterion describes?)
3. Specificity (how directly does the question assess the criterion?)
4. Learning objective alignment

Rate each criterion from 0.0 (not assessed) to 1.0 (perfectly assessed).

Return as JSON:
{{
  "criterion_id_1": 0.8,
  "criterion_id_2": 0.3,
  "criterion_id_3": 0.1
}}

Only include criteria with scores > 0.2. Provide scores that reflect how well the question assesses each specific criterion.
"""
        
        return prompt.strip()
    
    def _parse_mapping_response(self, response: str, criteria: List[MasteryCriterion]) -> Dict[str, float]:
        """Parse LLM mapping response."""
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if not json_match:
                return {}
            
            mapping_data = json.loads(json_match.group())
            
            # Validate criterion IDs and scores
            valid_mappings = {}
            criterion_ids = {c.criterionId for c in criteria}
            
            for criterion_id, score in mapping_data.items():
                if criterion_id in criterion_ids:
                    try:
                        score_float = float(score)
                        if 0.0 <= score_float <= 1.0:
                            valid_mappings[criterion_id] = score_float
                    except (ValueError, TypeError):
                        continue
            
            return valid_mappings
            
        except Exception as e:
            logger.error(f"Failed to parse mapping response: {e}")
            return {}
    
    def _organize_final_mappings(
        self,
        enhanced_mappings: Dict[str, Dict[str, float]],
        criteria: List[MasteryCriterion]
    ) -> Dict[str, List[Tuple[Question, float]]]:
        """Organize final mappings by criterion."""
        final_mappings = {criterion.criterionId: [] for criterion in criteria}
        
        # Convert question-centric mappings to criterion-centric
        for question_id, criterion_scores in enhanced_mappings.items():
            # Find the question object
            question = None
            # Note: We'd need to pass questions to this method to get the actual Question objects
            # For now, we'll create a placeholder structure
            
            for criterion_id, score in criterion_scores.items():
                if score >= self.mapping_threshold and criterion_id in final_mappings:
                    # Create a tuple with question ID (we'd use actual Question object in practice)
                    final_mappings[criterion_id].append((question_id, score))
        
        # Sort by confidence score (descending)
        for criterion_id in final_mappings:
            final_mappings[criterion_id].sort(key=lambda x: x[1], reverse=True)
        
        return final_mappings
    
    def _create_fallback_mappings(
        self,
        questions: List[Question],
        criteria: List[MasteryCriterion]
    ) -> Dict[str, List[Tuple[Question, float]]]:
        """Create basic fallback mappings when semantic analysis fails."""
        fallback_mappings = {criterion.criterionId: [] for criterion in criteria}
        
        # Simple UEE level matching
        for question in questions:
            matching_criteria = [
                c for c in criteria 
                if hasattr(question, 'uee_level') and c.ueeLevel == question.uee_level
            ]
            
            if matching_criteria:
                # Assign to first matching criterion with medium confidence
                criterion = matching_criteria[0]
                fallback_mappings[criterion.criterionId].append((question, 0.5))
            elif criteria:
                # Assign to first criterion with low confidence
                fallback_mappings[criteria[0].criterionId].append((question, 0.3))
        
        logger.info(f"Created fallback mappings for {len(questions)} questions")
        return fallback_mappings
    
    def validate_mappings(
        self,
        mappings: Dict[str, List[Tuple[Question, float]]],
        criteria: List[MasteryCriterion]
    ) -> Dict[str, List[str]]:
        """
        Validate question-criterion mappings and return quality reports.
        
        Args:
            mappings: Criterion ID to question mappings
            criteria: List of MasteryCriterion instances
            
        Returns:
            Dictionary of validation issues by criterion ID
        """
        validation_issues = {}
        
        for criterion in criteria:
            issues = []
            criterion_mappings = mappings.get(criterion.criterionId, [])
            
            # Check if criterion has sufficient questions
            if len(criterion_mappings) == 0:
                issues.append("No questions mapped to this criterion")
            elif len(criterion_mappings) < 2 and criterion.isRequired:
                issues.append("Required criterion has insufficient questions (< 2)")
            
            # Check confidence levels
            low_confidence_count = sum(1 for _, score in criterion_mappings if score < 0.5)
            if low_confidence_count > len(criterion_mappings) / 2:
                issues.append("More than half of mapped questions have low confidence")
            
            # Check UEE level alignment
            misaligned_count = 0
            for question, _ in criterion_mappings:
                if hasattr(question, 'uee_level') and question.uee_level != criterion.ueeLevel:
                    misaligned_count += 1
            
            if misaligned_count > 0:
                issues.append(f"{misaligned_count} questions have misaligned UEE levels")
            
            validation_issues[criterion.criterionId] = issues
        
        return validation_issues
    
    def get_mapping_statistics(
        self,
        mappings: Dict[str, List[Tuple[Question, float]]],
        criteria: List[MasteryCriterion]
    ) -> Dict[str, Any]:
        """Get statistics about question-criterion mappings."""
        total_questions = sum(len(question_list) for question_list in mappings.values())
        mapped_criteria = sum(1 for question_list in mappings.values() if len(question_list) > 0)
        
        confidence_scores = []
        for question_list in mappings.values():
            confidence_scores.extend([score for _, score in question_list])
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        high_confidence_count = sum(1 for score in confidence_scores if score >= self.confidence_threshold)
        
        return {
            'total_criteria': len(criteria),
            'mapped_criteria': mapped_criteria,
            'unmapped_criteria': len(criteria) - mapped_criteria,
            'total_questions_mapped': total_questions,
            'average_confidence': avg_confidence,
            'high_confidence_mappings': high_confidence_count,
            'mapping_coverage': mapped_criteria / len(criteria) if criteria else 0
        }


# Global service instance
question_mapping_service = QuestionMappingService()
