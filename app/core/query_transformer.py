"""
Query Transformation Module for RAG Chat Core.

This module handles intelligent query processing including:
- Query expansion and reformulation
- Intent classification (factual, conceptual, procedural)
- Query optimization for different search types
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from app.core.embeddings import GoogleEmbeddingService
import asyncio


class QueryIntent(Enum):
    """Classification of query intent types."""
    FACTUAL = "factual"           # Asking for specific facts or information
    CONCEPTUAL = "conceptual"     # Asking for explanations or understanding
    PROCEDURAL = "procedural"     # Asking for how-to or step-by-step instructions
    COMPARATIVE = "comparative"   # Asking for comparisons or contrasts
    ANALYTICAL = "analytical"     # Asking for analysis or evaluation
    CREATIVE = "creative"         # Asking for creative or open-ended responses


@dataclass
class QueryTransformation:
    """Result of query transformation process."""
    original_query: str
    expanded_query: str
    reformulated_queries: List[str]
    intent: QueryIntent
    confidence: float
    search_terms: List[str]
    metadata_filters: Dict[str, Any]
    search_strategy: str


class QueryTransformer:
    """
    Transforms user queries for optimal retrieval and response generation.
    
    This class handles:
    1. Query expansion and reformulation
    2. Intent classification
    3. Search optimization
    4. Metadata filter generation
    """
    
    def __init__(self, embedding_service: Optional[GoogleEmbeddingService] = None):
        self.embedding_service = embedding_service
        
        # Intent classification keywords
        self.intent_keywords = {
            QueryIntent.FACTUAL: [
                "what is", "who is", "when did", "where is", "how many", 
                "define", "definition", "fact", "information", "data",
                "tell me about", "explain what", "what are the facts"
            ],
            QueryIntent.CONCEPTUAL: [
                "explain", "why", "how does", "concept", "understand", 
                "meaning", "significance", "importance", "theory",
                "help me understand", "what does this mean", "clarify"
            ],
            QueryIntent.PROCEDURAL: [
                "how to", "steps", "procedure", "process", "method",
                "guide", "tutorial", "instructions", "walk me through",
                "show me how", "what are the steps"
            ],
            QueryIntent.COMPARATIVE: [
                "compare", "contrast", "difference", "similar", "versus",
                "vs", "better", "worse", "advantages", "disadvantages",
                "what's the difference", "how do they compare"
            ],
            QueryIntent.ANALYTICAL: [
                "analyze", "evaluate", "assess", "critique", "review",
                "pros and cons", "strengths", "weaknesses", "implications",
                "what do you think", "your opinion"
            ],
            QueryIntent.CREATIVE: [
                "brainstorm", "ideas", "creative", "innovative", "imagine",
                "what if", "possibilities", "alternatives", "suggestions",
                "come up with", "think outside"
            ]
        }
        
        # Search strategy mapping
        self.search_strategies = {
            QueryIntent.FACTUAL: "exact_match",
            QueryIntent.CONCEPTUAL: "semantic_broad",
            QueryIntent.PROCEDURAL: "sequential_search",
            QueryIntent.COMPARATIVE: "multi_concept_search",
            QueryIntent.ANALYTICAL: "contextual_search",
            QueryIntent.CREATIVE: "associative_search"
        }
    
    async def transform_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> QueryTransformation:
        """
        Transform a user query for optimal retrieval.
        
        Args:
            query: The original user query
            user_context: Optional context about the user and conversation
            
        Returns:
            QueryTransformation with expanded query, intent, and search parameters
        """
        # Normalize query
        normalized_query = self._normalize_query(query)
        
        # Classify intent
        intent, confidence = self._classify_intent(normalized_query)
        
        # Expand query based on intent
        expanded_query = await self._expand_query(normalized_query, intent)
        
        # Generate reformulated queries
        reformulated_queries = self._generate_reformulations(normalized_query, intent)
        
        # Extract search terms
        search_terms = self._extract_search_terms(expanded_query)
        
        # Generate metadata filters
        metadata_filters = self._generate_metadata_filters(query, intent, user_context)
        
        # Determine search strategy
        search_strategy = self.search_strategies.get(intent, "semantic_broad")
        
        return QueryTransformation(
            original_query=query,
            expanded_query=expanded_query,
            reformulated_queries=reformulated_queries,
            intent=intent,
            confidence=confidence,
            search_terms=search_terms,
            metadata_filters=metadata_filters,
            search_strategy=search_strategy
        )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query text for processing."""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Convert to lowercase for intent classification
        return query.lower()
    
    def _classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """
        Classify the intent of the query.
        
        Returns:
            Tuple of (intent, confidence_score)
        """
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query:
                    # Weight longer keywords more heavily
                    score += len(keyword.split()) * 2
                    # Boost score if keyword appears early in query
                    if query.startswith(keyword):
                        score += 3
            
            intent_scores[intent] = score
        
        # Find the intent with highest score
        if not intent_scores or max(intent_scores.values()) == 0:
            return QueryIntent.FACTUAL, 0.3  # Default with low confidence
        
        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]
        
        # Calculate confidence (normalize to 0-1 scale)
        confidence = min(max_score / 10.0, 1.0)
        
        return best_intent, confidence
    
    async def _expand_query(self, query: str, intent: QueryIntent) -> str:
        """
        Expand query based on intent and context.
        
        This method adds relevant terms and context to improve search results.
        """
        expanded_terms = []
        
        if intent == QueryIntent.FACTUAL:
            # Add definitional terms
            expanded_terms.extend(["definition", "explanation", "information"])
        
        elif intent == QueryIntent.CONCEPTUAL:
            # Add conceptual terms
            expanded_terms.extend(["concept", "theory", "principle", "understanding"])
        
        elif intent == QueryIntent.PROCEDURAL:
            # Add procedural terms
            expanded_terms.extend(["steps", "process", "method", "procedure", "guide"])
        
        elif intent == QueryIntent.COMPARATIVE:
            # Add comparative terms
            expanded_terms.extend(["comparison", "difference", "similarity", "contrast"])
        
        elif intent == QueryIntent.ANALYTICAL:
            # Add analytical terms
            expanded_terms.extend(["analysis", "evaluation", "assessment", "critique"])
        
        elif intent == QueryIntent.CREATIVE:
            # Add creative terms
            expanded_terms.extend(["ideas", "possibilities", "alternatives", "innovation"])
        
        # Create expanded query
        expanded_query = query
        if expanded_terms:
            expanded_query += " " + " ".join(expanded_terms)
        
        return expanded_query
    
    def _generate_reformulations(self, query: str, intent: QueryIntent) -> List[str]:
        """
        Generate alternative formulations of the query.
        
        This helps capture different ways the same information might be expressed.
        """
        reformulations = []
        
        if intent == QueryIntent.FACTUAL:
            # Generate fact-seeking reformulations
            reformulations.extend([
                f"What is {query}?",
                f"Tell me about {query}",
                f"Information about {query}",
                f"Facts about {query}"
            ])
        
        elif intent == QueryIntent.CONCEPTUAL:
            # Generate understanding-seeking reformulations
            reformulations.extend([
                f"Explain {query}",
                f"Help me understand {query}",
                f"What does {query} mean?",
                f"Concept of {query}"
            ])
        
        elif intent == QueryIntent.PROCEDURAL:
            # Generate how-to reformulations
            reformulations.extend([
                f"How to {query}",
                f"Steps for {query}",
                f"Process of {query}",
                f"Guide to {query}"
            ])
        
        elif intent == QueryIntent.COMPARATIVE:
            # Generate comparative reformulations
            reformulations.extend([
                f"Compare {query}",
                f"Difference between {query}",
                f"Similarities in {query}",
                f"Contrast {query}"
            ])
        
        elif intent == QueryIntent.ANALYTICAL:
            # Generate analytical reformulations
            reformulations.extend([
                f"Analyze {query}",
                f"Evaluate {query}",
                f"Assessment of {query}",
                f"Critical analysis of {query}"
            ])
        
        elif intent == QueryIntent.CREATIVE:
            # Generate creative reformulations
            reformulations.extend([
                f"Ideas for {query}",
                f"Creative approaches to {query}",
                f"Innovative solutions for {query}",
                f"Brainstorm {query}"
            ])
        
        # Filter out reformulations that are too similar to original
        unique_reformulations = []
        for reformulation in reformulations:
            if reformulation.lower() != query and reformulation not in unique_reformulations:
                unique_reformulations.append(reformulation)
        
        return unique_reformulations[:5]  # Limit to top 5
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """
        Extract key search terms from the expanded query.
        
        This identifies the most important terms for vector search.
        """
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'what', 'how', 'when', 'where', 'why', 'who', 'which'
        }
        
        # Extract words and filter stop words
        words = re.findall(r'\b\w+\b', query.lower())
        search_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Remove duplicates while preserving order
        unique_terms = []
        for term in search_terms:
            if term not in unique_terms:
                unique_terms.append(term)
        
        return unique_terms
    
    def _generate_metadata_filters(self, query: str, intent: QueryIntent, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate metadata filters based on query and intent.
        
        This helps narrow down search results to relevant content types.
        """
        filters = {}
        
        # Intent-based filtering
        if intent == QueryIntent.FACTUAL:
            filters['locus_type'] = ['key_propositions_and_facts', 'key_entities_and_definitions']
        
        elif intent == QueryIntent.CONCEPTUAL:
            filters['locus_type'] = ['key_propositions_and_facts', 'key_entities_and_definitions']
        
        elif intent == QueryIntent.PROCEDURAL:
            filters['locus_type'] = ['described_processes_and_steps']
        
        elif intent == QueryIntent.COMPARATIVE:
            filters['locus_type'] = ['key_propositions_and_facts', 'identified_relationships']
        
        elif intent == QueryIntent.ANALYTICAL:
            filters['locus_type'] = ['key_propositions_and_facts', 'identified_relationships']
        
        elif intent == QueryIntent.CREATIVE:
            filters['locus_type'] = ['implicit_and_open_questions', 'identified_relationships']
        
        # Add user context filters if available
        if user_context:
            if 'preferred_difficulty' in user_context:
                filters['difficulty_level'] = user_context['preferred_difficulty']
            
            if 'learning_stage' in user_context:
                filters['uue_stage'] = user_context['learning_stage']
            
            if 'subject_area' in user_context:
                filters['subject_area'] = user_context['subject_area']
        
        return filters
    
    def get_search_parameters(self, transformation: QueryTransformation) -> Dict[str, Any]:
        """
        Get search parameters optimized for the query transformation.
        
        This provides the search service with optimal parameters for retrieval.
        """
        base_params = {
            'query': transformation.expanded_query,
            'top_k': 10,
            'metadata_filters': transformation.metadata_filters,
            'search_strategy': transformation.search_strategy
        }
        
        # Adjust parameters based on intent
        if transformation.intent == QueryIntent.FACTUAL:
            base_params.update({
                'top_k': 5,
                'similarity_threshold': 0.8,
                'exact_match_boost': 2.0
            })
        
        elif transformation.intent == QueryIntent.CONCEPTUAL:
            base_params.update({
                'top_k': 15,
                'similarity_threshold': 0.7,
                'diversity_factor': 0.3
            })
        
        elif transformation.intent == QueryIntent.PROCEDURAL:
            base_params.update({
                'top_k': 8,
                'similarity_threshold': 0.75,
                'sequential_boost': 1.5
            })
        
        elif transformation.intent == QueryIntent.COMPARATIVE:
            base_params.update({
                'top_k': 20,
                'similarity_threshold': 0.6,
                'diversity_factor': 0.5
            })
        
        elif transformation.intent == QueryIntent.ANALYTICAL:
            base_params.update({
                'top_k': 15,
                'similarity_threshold': 0.65,
                'context_window': 3
            })
        
        elif transformation.intent == QueryIntent.CREATIVE:
            base_params.update({
                'top_k': 25,
                'similarity_threshold': 0.5,
                'diversity_factor': 0.7,
                'exploration_factor': 0.4
            })
        
        return base_params
