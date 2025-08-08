"""
Criterion-Mapped Question Generation Service.

This service generates questions specifically mapped to mastery criteria,
supporting the Core API's primitive-based spaced repetition system.
"""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum

from app.models.learning_blueprint import MasteryCriterion, KnowledgePrimitive
from app.core.llm_service import LLMService

logger = logging.getLogger(__name__)


class QuestionType(str, Enum):
    """Types of questions that can be generated."""
    MULTIPLE_CHOICE = "multiple_choice"
    SHORT_ANSWER = "short_answer"
    TRUE_FALSE = "true_false"
    ESSAY = "essay"
    PRACTICAL = "practical"


class CriterionQuestionGenerator:
    """Generate questions specifically mapped to mastery criteria."""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
    
    def get_uee_question_prompt(self, uee_level: str, criterion: MasteryCriterion, primitive: KnowledgePrimitive) -> str:
        """Generate UEE-level specific prompts for question generation."""
        
        if uee_level == "UNDERSTAND":
            return f"""
Generate a question that tests UNDERSTANDING of the concept "{primitive.title}".

Mastery Criterion: {criterion.title}
Criterion Description: {criterion.description}
Primitive Description: {primitive.description}
Primitive Type: {primitive.primitiveType}

The question should test if the learner can:
- Define, explain, or describe the concept
- Identify key characteristics or properties
- Recognize the concept in different contexts
- Demonstrate comprehension of fundamental principles

Generate a clear, focused question that directly assesses understanding of this specific criterion.
Focus on comprehension, not application or analysis.
"""

        elif uee_level == "USE":
            return f"""
Generate a question that tests USAGE/APPLICATION of the concept "{primitive.title}".

Mastery Criterion: {criterion.title}
Criterion Description: {criterion.description}
Primitive Description: {primitive.description}
Primitive Type: {primitive.primitiveType}

The question should test if the learner can:
- Apply the concept to solve problems
- Use the concept in practical scenarios
- Execute procedures or processes correctly
- Implement the concept in real situations

Generate a practical question that requires active application of this specific criterion.
Focus on doing, applying, or implementing, not just understanding.
"""

        elif uee_level == "EXPLORE":
            return f"""
Generate a question that tests EXPLORATION/ANALYSIS of the concept "{primitive.title}".

Mastery Criterion: {criterion.title}
Criterion Description: {criterion.description}
Primitive Description: {primitive.description}
Primitive Type: {primitive.primitiveType}

The question should test if the learner can:
- Analyze relationships and implications
- Evaluate effectiveness or appropriateness
- Create new applications or variations
- Synthesize with other concepts
- Critically assess limitations or alternatives

Generate a higher-order question that requires analysis, evaluation, or creation.
Focus on critical thinking, not just application or understanding.
"""
        
        return f"Generate a question about {primitive.title} related to {criterion.title}"
    
    async def generate_questions_for_criterion(
        self,
        criterion: MasteryCriterion,
        primitive: KnowledgePrimitive,
        num_questions: int = 3,
        question_types: Optional[List[QuestionType]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate questions specifically for a mastery criterion.
        
        Args:
            criterion: MasteryCriterion to generate questions for
            primitive: Parent KnowledgePrimitive for context
            num_questions: Number of questions to generate
            question_types: Types of questions to generate
            
        Returns:
            List of generated questions with metadata
        """
        if question_types is None:
            question_types = [QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER]
        
        questions = []
        
        # Generate UEE-level appropriate prompt
        prompt = self.get_uee_question_prompt(criterion.ueeLevel, criterion, primitive)
        
        for i in range(num_questions):
            question_type = question_types[i % len(question_types)]
            
            # Add question type specific instructions
            type_prompt = prompt + f"\n\nGenerate a {question_type.value.replace('_', ' ')} question."
            
            try:
                # Generate question using LLM
                response = await self.llm_service.generate_question(
                    prompt=type_prompt,
                    question_type=question_type.value
                )
                
                question_data = {
                    "criterionId": criterion.criterionId,
                    "primitiveId": primitive.primitiveId,
                    "questionType": question_type.value,
                    "ueeLevel": criterion.ueeLevel,
                    "weight": criterion.weight,
                    "questionText": response.get("question", ""),
                    "answerText": response.get("answer", ""),
                    "options": response.get("options", []),
                    "explanation": response.get("explanation", ""),
                    "difficulty": primitive.difficultyLevel,
                    "estimatedTimeMinutes": response.get("estimated_time", 2)
                }
                
                questions.append(question_data)
                logger.debug(f"Generated {question_type.value} question for criterion {criterion.criterionId}")
                
            except Exception as e:
                logger.error(f"Failed to generate question for criterion {criterion.criterionId}: {e}")
                # Continue with other questions
        
        return questions
    
    async def generate_questions_for_primitive(
        self,
        primitive: KnowledgePrimitive,
        questions_per_criterion: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Generate questions for all criteria of a primitive.
        
        Args:
            primitive: KnowledgePrimitive to generate questions for
            questions_per_criterion: Number of questions per criterion
            
        Returns:
            List of all generated questions for the primitive
        """
        all_questions = []
        
        for criterion in primitive.masteryCriteria:
            try:
                questions = await self.generate_questions_for_criterion(
                    criterion=criterion,
                    primitive=primitive,
                    num_questions=questions_per_criterion
                )
                all_questions.extend(questions)
                
            except Exception as e:
                logger.error(f"Failed to generate questions for criterion {criterion.criterionId}: {e}")
        
        logger.info(f"Generated {len(all_questions)} questions for primitive {primitive.primitiveId}")
        return all_questions
    
    async def generate_questions_for_primitives(
        self,
        primitives: List[KnowledgePrimitive],
        questions_per_criterion: int = 2
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate questions for multiple primitives.
        
        Args:
            primitives: List of KnowledgePrimitive instances
            questions_per_criterion: Number of questions per criterion
            
        Returns:
            Dictionary mapping primitiveId to list of questions
        """
        primitive_questions = {}
        
        for primitive in primitives:
            try:
                questions = await self.generate_questions_for_primitive(
                    primitive=primitive,
                    questions_per_criterion=questions_per_criterion
                )
                primitive_questions[primitive.primitiveId] = questions
                
            except Exception as e:
                logger.error(f"Failed to generate questions for primitive {primitive.primitiveId}: {e}")
                primitive_questions[primitive.primitiveId] = []
        
        total_questions = sum(len(questions) for questions in primitive_questions.values())
        logger.info(f"Generated {total_questions} questions across {len(primitives)} primitives")
        
        return primitive_questions
    
    def filter_questions_by_uee_level(
        self,
        questions: List[Dict[str, Any]],
        uee_level: str
    ) -> List[Dict[str, Any]]:
        """Filter questions by UEE level."""
        return [q for q in questions if q.get("ueeLevel") == uee_level]
    
    def filter_questions_by_weight(
        self,
        questions: List[Dict[str, Any]],
        min_weight: float = 0.0,
        max_weight: float = 5.0
    ) -> List[Dict[str, Any]]:
        """Filter questions by criterion weight."""
        return [
            q for q in questions 
            if min_weight <= q.get("weight", 0) <= max_weight
        ]
    
    def sort_questions_by_weight(
        self,
        questions: List[Dict[str, Any]],
        descending: bool = True
    ) -> List[Dict[str, Any]]:
        """Sort questions by criterion weight."""
        return sorted(
            questions,
            key=lambda q: q.get("weight", 0),
            reverse=descending
        )


# Global generator instance
criterion_question_generator = CriterionQuestionGenerator()
