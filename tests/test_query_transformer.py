"""
Unit tests for Query Transformer component.

Tests query transformation, intent classification, and query optimization.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import List, Dict, Any

from app.core.query_transformer import (
    QueryTransformer,
    QueryTransformation,
    QueryIntent,
    QueryFilterParams,
    QueryOptimization
)


class TestQueryTransformer:
    """Test suite for QueryTransformer class."""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service for testing."""
        mock_service = AsyncMock()
        mock_service.embed_text.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        return mock_service
    
    @pytest.fixture
    def query_transformer(self, mock_embedding_service):
        """Create QueryTransformer instance with mocked dependencies."""
        return QueryTransformer(mock_embedding_service)
    
    @pytest.mark.asyncio
    async def test_transform_query_basic(self, query_transformer):
        """Test basic query transformation."""
        query = "What is machine learning?"
        user_context = {"learning_stage": "understand"}
        conversation_history = []
        
        result = await query_transformer.transform_query(
            query=query,
            user_context=user_context,
            conversation_history=conversation_history
        )
        
        assert isinstance(result, QueryTransformation)
        assert result.original_query == query
        assert result.normalized_query is not None
        assert result.intent in QueryIntent
        assert result.expanded_query is not None
        assert result.filter_params is not None
        assert result.optimization is not None
    
    @pytest.mark.asyncio
    async def test_intent_classification_factual(self, query_transformer):
        """Test intent classification for factual queries."""
        factual_queries = [
            "What is Python?",
            "Define machine learning",
            "What are the components of a neural network?",
            "Who invented the internet?",
            "When was JavaScript created?"
        ]
        
        for query in factual_queries:
            result = await query_transformer.transform_query(
                query=query,
                user_context={},
                conversation_history=[]
            )
            
            assert result.intent == QueryIntent.FACTUAL, f"Query '{query}' should be classified as FACTUAL"
    
    @pytest.mark.asyncio
    async def test_intent_classification_procedural(self, query_transformer):
        """Test intent classification for procedural queries."""
        procedural_queries = [
            "How do I install Python?",
            "How to create a neural network?",
            "Steps to deploy a web application",
            "How can I implement sorting algorithm?",
            "What are the steps to learn programming?"
        ]
        
        for query in procedural_queries:
            result = await query_transformer.transform_query(
                query=query,
                user_context={},
                conversation_history=[]
            )
            
            assert result.intent == QueryIntent.PROCEDURAL, f"Query '{query}' should be classified as PROCEDURAL"
    
    @pytest.mark.asyncio
    async def test_intent_classification_conceptual(self, query_transformer):
        """Test intent classification for conceptual queries."""
        conceptual_queries = [
            "Why is recursion important?",
            "Explain the concept of inheritance",
            "What is the relationship between AI and ML?",
            "Why do we need databases?",
            "Explain the theory behind neural networks"
        ]
        
        for query in conceptual_queries:
            result = await query_transformer.transform_query(
                query=query,
                user_context={},
                conversation_history=[]
            )
            
            assert result.intent == QueryIntent.CONCEPTUAL, f"Query '{query}' should be classified as CONCEPTUAL"
    
    @pytest.mark.asyncio
    async def test_intent_classification_comparative(self, query_transformer):
        """Test intent classification for comparative queries."""
        comparative_queries = [
            "Python vs JavaScript",
            "Compare SQL and NoSQL databases",
            "Difference between AI and ML",
            "React vs Vue.js performance",
            "What are the pros and cons of microservices?"
        ]
        
        for query in comparative_queries:
            result = await query_transformer.transform_query(
                query=query,
                user_context={},
                conversation_history=[]
            )
            
            assert result.intent == QueryIntent.COMPARATIVE, f"Query '{query}' should be classified as COMPARATIVE"
    
    @pytest.mark.asyncio
    async def test_query_normalization(self, query_transformer):
        """Test query normalization."""
        test_cases = [
            ("WHAT IS MACHINE LEARNING???", "what is machine learning"),
            ("  How   do   I   learn  Python?  ", "how do i learn python"),
            ("Python vs JavaScript!!!", "python vs javascript"),
            ("What's the difference between AI & ML?", "what is the difference between ai and ml")
        ]
        
        for original, expected in test_cases:
            result = await query_transformer.transform_query(
                query=original,
                user_context={},
                conversation_history=[]
            )
            
            assert result.normalized_query == expected, f"Expected '{expected}', got '{result.normalized_query}'"
    
    @pytest.mark.asyncio
    async def test_query_expansion(self, query_transformer):
        """Test query expansion functionality."""
        query = "ML algorithms"
        
        result = await query_transformer.transform_query(
            query=query,
            user_context={},
            conversation_history=[]
        )
        
        expanded = result.expanded_query
        assert "machine learning" in expanded.lower()
        assert "algorithms" in expanded.lower()
        assert len(expanded) > len(query)
    
    @pytest.mark.asyncio
    async def test_filter_params_generation(self, query_transformer):
        """Test filter parameters generation."""
        query = "Python programming basics"
        user_context = {
            "learning_stage": "understand",
            "preferred_difficulty": "beginner"
        }
        
        result = await query_transformer.transform_query(
            query=query,
            user_context=user_context,
            conversation_history=[]
        )
        
        filters = result.filter_params
        assert filters.uue_stage == "understand"
        assert filters.difficulty_level == "beginner"
        assert filters.locus_types is not None
        assert len(filters.locus_types) > 0
    
    @pytest.mark.asyncio
    async def test_conversation_context_integration(self, query_transformer):
        """Test integration of conversation context."""
        query = "Can you explain more about that?"
        conversation_history = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI..."},
            {"role": "user", "content": "Can you explain more about that?"}
        ]
        
        result = await query_transformer.transform_query(
            query=query,
            user_context={},
            conversation_history=conversation_history
        )
        
        # The expanded query should include context from conversation
        assert "machine learning" in result.expanded_query.lower()
        assert len(result.expanded_query) > len(query)
    
    @pytest.mark.asyncio
    async def test_optimization_parameters(self, query_transformer):
        """Test query optimization parameters."""
        query = "Python tutorial"
        
        result = await query_transformer.transform_query(
            query=query,
            user_context={},
            conversation_history=[]
        )
        
        optimization = result.optimization
        assert optimization.search_strategy in ["semantic", "keyword", "hybrid"]
        assert 0.0 <= optimization.similarity_threshold <= 1.0
        assert optimization.max_results > 0
        assert 0.0 <= optimization.diversity_factor <= 1.0
    
    @pytest.mark.asyncio
    async def test_creative_intent_classification(self, query_transformer):
        """Test creative intent classification."""
        creative_queries = [
            "Create a Python script for data analysis",
            "Generate a story about AI",
            "Write a poem about programming",
            "Design a solution for this problem",
            "Brainstorm ideas for machine learning project"
        ]
        
        for query in creative_queries:
            result = await query_transformer.transform_query(
                query=query,
                user_context={},
                conversation_history=[]
            )
            
            assert result.intent == QueryIntent.CREATIVE, f"Query '{query}' should be classified as CREATIVE"
    
    @pytest.mark.asyncio
    async def test_analytical_intent_classification(self, query_transformer):
        """Test analytical intent classification."""
        analytical_queries = [
            "Analyze this code for bugs",
            "Evaluate the performance of this algorithm",
            "Review my solution to this problem",
            "What are the trade-offs of this approach?",
            "Assess the complexity of this system"
        ]
        
        for query in analytical_queries:
            result = await query_transformer.transform_query(
                query=query,
                user_context={},
                conversation_history=[]
            )
            
            assert result.intent == QueryIntent.ANALYTICAL, f"Query '{query}' should be classified as ANALYTICAL"
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self, query_transformer):
        """Test handling of empty or invalid queries."""
        invalid_queries = ["", "   ", "?", "!!!"]
        
        for query in invalid_queries:
            result = await query_transformer.transform_query(
                query=query,
                user_context={},
                conversation_history=[]
            )
            
            # Should handle gracefully
            assert result.original_query == query
            assert result.normalized_query is not None
            assert result.intent is not None
    
    @pytest.mark.asyncio
    async def test_long_query_handling(self, query_transformer):
        """Test handling of very long queries."""
        long_query = "What is machine learning and how does it work and what are its applications " * 50
        
        result = await query_transformer.transform_query(
            query=long_query,
            user_context={},
            conversation_history=[]
        )
        
        # Should handle gracefully without errors
        assert result.original_query == long_query
        assert result.normalized_query is not None
        assert result.intent is not None
    
    @pytest.mark.asyncio
    async def test_special_characters_handling(self, query_transformer):
        """Test handling of special characters in queries."""
        special_queries = [
            "What is C++ programming?",
            "How to use @decorators in Python?",
            "JavaScript's async/await syntax",
            "SQL SELECT * FROM table",
            "Regular expressions: /[a-z]+/g"
        ]
        
        for query in special_queries:
            result = await query_transformer.transform_query(
                query=query,
                user_context={},
                conversation_history=[]
            )
            
            # Should handle special characters gracefully
            assert result.original_query == query
            assert result.normalized_query is not None
            assert result.intent is not None
    
    @pytest.mark.asyncio
    async def test_multilingual_query_handling(self, query_transformer):
        """Test handling of non-English queries."""
        multilingual_queries = [
            "¿Qué es el aprendizaje automático?",  # Spanish
            "Qu'est-ce que l'intelligence artificielle?",  # French
            "Was ist maschinelles Lernen?",  # German
            "什么是机器学习？",  # Chinese
            "मशीन लर्निंग क्या है?"  # Hindi
        ]
        
        for query in multilingual_queries:
            result = await query_transformer.transform_query(
                query=query,
                user_context={},
                conversation_history=[]
            )
            
            # Should handle gracefully
            assert result.original_query == query
            assert result.normalized_query is not None
            assert result.intent is not None
    
    @pytest.mark.asyncio
    async def test_context_aware_filtering(self, query_transformer):
        """Test context-aware filter generation."""
        query = "Python basics"
        user_context = {
            "learning_stage": "use",
            "preferred_difficulty": "advanced",
            "current_topic": "web development",
            "discussed_concepts": ["functions", "classes"]
        }
        
        result = await query_transformer.transform_query(
            query=query,
            user_context=user_context,
            conversation_history=[]
        )
        
        filters = result.filter_params
        assert filters.uue_stage == "use"
        assert filters.difficulty_level == "advanced"
        assert filters.topic_focus == "web development"
        assert "functions" in filters.discussed_concepts
        assert "classes" in filters.discussed_concepts
    
    def test_query_transformation_serialization(self, query_transformer):
        """Test QueryTransformation object serialization."""
        transformation = QueryTransformation(
            original_query="test query",
            normalized_query="test query",
            intent=QueryIntent.FACTUAL,
            expanded_query="expanded test query",
            filter_params=QueryFilterParams(
                uue_stage="understand",
                locus_types=["foundational_concept"],
                difficulty_level="beginner"
            ),
            optimization=QueryOptimization(
                search_strategy="semantic",
                similarity_threshold=0.7,
                max_results=10,
                diversity_factor=0.3
            ),
            processing_time_ms=100.0,
            metadata={"test": "value"}
        )
        
        # Test that the object can be serialized
        dict_repr = transformation.__dict__
        assert "original_query" in dict_repr
        assert "intent" in dict_repr
        assert "filter_params" in dict_repr
