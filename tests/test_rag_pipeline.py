"""
Integration tests for RAG pipeline components.

Tests the full RAG pipeline including query transformation, search, context assembly,
and response generation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import List, Dict, Any

from app.core.query_transformer import QueryTransformer, QueryTransformation, QueryIntent
from app.core.rag_search import RAGSearchService, RAGSearchRequest, RAGSearchResult
from app.core.context_assembly import ContextAssembler, AssembledContext
from app.core.response_generation import ResponseGenerator, ResponseGenerationRequest, GeneratedResponse


class TestRAGPipeline:
    """Integration tests for the complete RAG pipeline."""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service."""
        mock_service = AsyncMock()
        mock_service.embed_text.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        return mock_service
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store."""
        mock_store = AsyncMock()
        mock_store.search.return_value = [
            {
                "id": "test_node_1",
                "content": "Python is a programming language",
                "metadata": {
                    "source_id": "blueprint_1",
                    "locus_type": "foundational_concept",
                    "uue_stage": "understand"
                },
                "distance": 0.2
            },
            {
                "id": "test_node_2", 
                "content": "Machine learning uses algorithms to learn patterns",
                "metadata": {
                    "source_id": "blueprint_2",
                    "locus_type": "use_case",
                    "uue_stage": "use"
                },
                "distance": 0.3
            }
        ]
        return mock_store
    
    @pytest.fixture
    def mock_gemini_service(self):
        """Mock Gemini service."""
        mock_service = AsyncMock()
        mock_service.generate_response.return_value = "This is a test response from the AI assistant."
        return mock_service
    
    @pytest.fixture
    def rag_components(self, mock_embedding_service, mock_vector_store, mock_gemini_service):
        """Create RAG pipeline components."""
        query_transformer = QueryTransformer(mock_embedding_service)
        rag_search_service = RAGSearchService(mock_vector_store, mock_embedding_service)
        context_assembler = ContextAssembler(rag_search_service)
        response_generator = ResponseGenerator(mock_gemini_service)
        
        return {
            "query_transformer": query_transformer,
            "rag_search_service": rag_search_service,
            "context_assembler": context_assembler,
            "response_generator": response_generator
        }
    
    @pytest.mark.asyncio
    async def test_full_rag_pipeline_factual_query(self, rag_components):
        """Test complete RAG pipeline with factual query."""
        query_transformer = rag_components["query_transformer"]
        context_assembler = rag_components["context_assembler"]
        response_generator = rag_components["response_generator"]
        
        # Step 1: Transform query
        user_query = "What is Python programming?"
        query_transformation = await query_transformer.transform_query(
            query=user_query,
            user_context={"learning_stage": "understand"},
            conversation_history=[]
        )
        
        assert query_transformation.intent == QueryIntent.FACTUAL
        assert "python" in query_transformation.normalized_query
        
        # Step 2: Assemble context
        assembled_context = await context_assembler.assemble_context(
            user_id="test_user",
            session_id="test_session",
            current_query=user_query,
            query_transformation=query_transformation
        )
        
        assert isinstance(assembled_context, AssembledContext)
        assert len(assembled_context.retrieved_knowledge) > 0
        assert assembled_context.context_quality_score > 0
        
        # Step 3: Generate response
        response_request = ResponseGenerationRequest(
            user_query=user_query,
            query_transformation=query_transformation,
            assembled_context=assembled_context,
            max_tokens=500,
            temperature=0.7
        )
        
        generated_response = await response_generator.generate_response(response_request)
        
        assert isinstance(generated_response, GeneratedResponse)
        assert len(generated_response.content) > 0
        assert generated_response.confidence_score > 0
        assert generated_response.factual_accuracy_score > 0
    
    @pytest.mark.asyncio
    async def test_rag_pipeline_with_conversation_history(self, rag_components):
        """Test RAG pipeline with conversation history."""
        query_transformer = rag_components["query_transformer"]
        context_assembler = rag_components["context_assembler"]
        response_generator = rag_components["response_generator"]
        
        # Simulate conversation history
        conversation_history = [
            {"role": "user", "content": "Tell me about programming"},
            {"role": "assistant", "content": "Programming is the process of creating software..."},
            {"role": "user", "content": "What about Python specifically?"}
        ]
        
        # Add messages to conversation buffer
        session_id = "test_session_with_history"
        for msg in conversation_history:
            context_assembler.add_message_to_buffer(
                session_id=session_id,
                role=msg["role"],
                content=msg["content"]
            )
        
        # Transform current query
        current_query = "Can you give me examples?"
        query_transformation = await query_transformer.transform_query(
            query=current_query,
            user_context={"learning_stage": "use"},
            conversation_history=conversation_history
        )
        
        # Should expand query based on conversation context
        assert "python" in query_transformation.expanded_query.lower()
        assert "examples" in query_transformation.expanded_query.lower()
        
        # Assemble context
        assembled_context = await context_assembler.assemble_context(
            user_id="test_user",
            session_id=session_id,
            current_query=current_query,
            query_transformation=query_transformation
        )
        
        # Should have conversation context
        assert len(assembled_context.conversational_context) > 0
        assert assembled_context.context_quality_score > 0
    
    @pytest.mark.asyncio
    async def test_rag_pipeline_procedural_query(self, rag_components):
        """Test RAG pipeline with procedural query."""
        query_transformer = rag_components["query_transformer"]
        context_assembler = rag_components["context_assembler"]
        response_generator = rag_components["response_generator"]
        
        # Procedural query
        user_query = "How do I create a Python function?"
        query_transformation = await query_transformer.transform_query(
            query=user_query,
            user_context={"learning_stage": "use", "preferred_difficulty": "beginner"},
            conversation_history=[]
        )
        
        assert query_transformation.intent == QueryIntent.PROCEDURAL
        assert query_transformation.filter_params.uue_stage == "use"
        assert query_transformation.filter_params.difficulty_level == "beginner"
        
        # Assemble context
        assembled_context = await context_assembler.assemble_context(
            user_id="test_user",
            session_id="test_session_procedural",
            current_query=user_query,
            query_transformation=query_transformation
        )
        
        # Generate response
        response_request = ResponseGenerationRequest(
            user_query=user_query,
            query_transformation=query_transformation,
            assembled_context=assembled_context
        )
        
        generated_response = await response_generator.generate_response(response_request)
        
        # Should be appropriate for procedural queries
        assert generated_response.response_type.value in ["explanation", "clarification"]
        assert generated_response.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_rag_pipeline_with_metadata_filtering(self, rag_components):
        """Test RAG pipeline with metadata filtering."""
        query_transformer = rag_components["query_transformer"]
        context_assembler = rag_components["context_assembler"]
        
        # Query with specific filtering needs
        user_query = "Python fundamentals"
        user_context = {
            "learning_stage": "understand",
            "preferred_difficulty": "beginner",
            "current_topic": "programming basics"
        }
        
        query_transformation = await query_transformer.transform_query(
            query=user_query,
            user_context=user_context,
            conversation_history=[]
        )
        
        # Should generate appropriate filters
        filters = query_transformation.filter_params
        assert filters.uue_stage == "understand"
        assert filters.difficulty_level == "beginner"
        assert filters.topic_focus == "programming basics"
        
        # Assemble context with filtering
        assembled_context = await context_assembler.assemble_context(
            user_id="test_user",
            session_id="test_session_filtering",
            current_query=user_query,
            query_transformation=query_transformation
        )
        
        # Should have retrieved relevant knowledge
        assert len(assembled_context.retrieved_knowledge) > 0
        assert assembled_context.context_quality_score > 0
    
    @pytest.mark.asyncio
    async def test_rag_pipeline_error_handling(self, rag_components):
        """Test RAG pipeline error handling."""
        query_transformer = rag_components["query_transformer"]
        context_assembler = rag_components["context_assembler"]
        response_generator = rag_components["response_generator"]
        
        # Mock an error in the embedding service
        query_transformer.embedding_service.embed_text.side_effect = Exception("Embedding service error")
        
        # Should handle error gracefully
        with pytest.raises(Exception):
            await query_transformer.transform_query(
                query="test query",
                user_context={},
                conversation_history=[]
            )
        
        # Reset mock
        query_transformer.embedding_service.embed_text.side_effect = None
        query_transformer.embedding_service.embed_text.return_value = [0.1, 0.2, 0.3]
        
        # Test response generation error handling
        response_generator.gemini_service.generate_response.side_effect = Exception("LLM service error")
        
        # Should handle gracefully
        query_transformation = await query_transformer.transform_query(
            query="test query",
            user_context={},
            conversation_history=[]
        )
        
        assembled_context = await context_assembler.assemble_context(
            user_id="test_user",
            session_id="test_session_error",
            current_query="test query",
            query_transformation=query_transformation
        )
        
        response_request = ResponseGenerationRequest(
            user_query="test query",
            query_transformation=query_transformation,
            assembled_context=assembled_context
        )
        
        generated_response = await response_generator.generate_response(response_request)
        
        # Should contain error fallback message
        assert "trouble generating" in generated_response.content.lower()
    
    @pytest.mark.asyncio
    async def test_rag_pipeline_context_pruning(self, rag_components):
        """Test RAG pipeline context pruning for token limits."""
        context_assembler = rag_components["context_assembler"]
        
        # Set low token limit to trigger pruning
        context_assembler.max_context_tokens = 100
        
        # Create query transformation
        query_transformation = QueryTransformation(
            original_query="test query",
            normalized_query="test query",
            intent=QueryIntent.FACTUAL,
            expanded_query="expanded test query",
            filter_params=None,
            optimization=None,
            processing_time_ms=10.0,
            metadata={}
        )
        
        # Add many messages to conversation buffer
        session_id = "test_session_pruning"
        for i in range(20):
            context_assembler.add_message_to_buffer(
                session_id=session_id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"This is message {i} with some content to fill tokens " * 10
            )
        
        # Assemble context
        assembled_context = await context_assembler.assemble_context(
            user_id="test_user",
            session_id=session_id,
            current_query="test query",
            query_transformation=query_transformation
        )
        
        # Should have pruned context to fit token limit
        assert assembled_context.total_tokens <= context_assembler.max_context_tokens
        assert len(assembled_context.conversational_context) < 20  # Should be pruned
    
    @pytest.mark.asyncio
    async def test_rag_pipeline_session_state_updates(self, rag_components):
        """Test RAG pipeline session state updates."""
        context_assembler = rag_components["context_assembler"]
        
        session_id = "test_session_updates"
        
        # Update session state
        updates = {
            "current_topic": "machine learning",
            "discussed_concepts": ["neural networks", "algorithms"],
            "learning_objectives": ["understand ML basics"]
        }
        
        context_assembler.update_session_state(session_id, updates)
        
        # Retrieve session state
        session_state = context_assembler._get_or_create_session_state("test_user", session_id)
        
        assert session_state.current_topic == "machine learning"
        assert "neural networks" in session_state.discussed_concepts
        assert "algorithms" in session_state.discussed_concepts
        assert "understand ML basics" in session_state.learning_objectives
    
    @pytest.mark.asyncio
    async def test_rag_pipeline_cognitive_profile_integration(self, rag_components):
        """Test RAG pipeline cognitive profile integration."""
        context_assembler = rag_components["context_assembler"]
        response_generator = rag_components["response_generator"]
        
        # Create user with specific cognitive profile
        user_id = "test_user_profile"
        cognitive_profile = context_assembler._get_or_create_cognitive_profile(user_id)
        
        # Update profile
        cognitive_profile.learning_style = "visual"
        cognitive_profile.preferred_difficulty = "advanced"
        cognitive_profile.preferred_explanation_style = "examples"
        
        # Create query transformation
        query_transformation = QueryTransformation(
            original_query="explain neural networks",
            normalized_query="explain neural networks",
            intent=QueryIntent.CONCEPTUAL,
            expanded_query="explain neural networks machine learning",
            filter_params=None,
            optimization=None,
            processing_time_ms=10.0,
            metadata={}
        )
        
        # Assemble context
        assembled_context = await context_assembler.assemble_context(
            user_id=user_id,
            session_id="test_session_profile",
            current_query="explain neural networks",
            query_transformation=query_transformation
        )
        
        # Should include cognitive profile
        assert assembled_context.cognitive_profile.learning_style == "visual"
        assert assembled_context.cognitive_profile.preferred_difficulty == "advanced"
        assert assembled_context.cognitive_profile.preferred_explanation_style == "examples"
        
        # Generate response
        response_request = ResponseGenerationRequest(
            user_query="explain neural networks",
            query_transformation=query_transformation,
            assembled_context=assembled_context
        )
        
        generated_response = await response_generator.generate_response(response_request)
        
        # Should use appropriate tone for profile
        assert generated_response.tone_style.value == "conversational"  # Based on "examples" style
    
    @pytest.mark.asyncio
    async def test_rag_pipeline_performance_metrics(self, rag_components):
        """Test RAG pipeline performance metrics tracking."""
        query_transformer = rag_components["query_transformer"]
        context_assembler = rag_components["context_assembler"]
        response_generator = rag_components["response_generator"]
        
        # Execute full pipeline
        user_query = "What is machine learning?"
        
        query_transformation = await query_transformer.transform_query(
            query=user_query,
            user_context={},
            conversation_history=[]
        )
        
        assembled_context = await context_assembler.assemble_context(
            user_id="test_user_perf",
            session_id="test_session_perf",
            current_query=user_query,
            query_transformation=query_transformation
        )
        
        response_request = ResponseGenerationRequest(
            user_query=user_query,
            query_transformation=query_transformation,
            assembled_context=assembled_context
        )
        
        generated_response = await response_generator.generate_response(response_request)
        
        # Check performance metrics
        assert query_transformation.processing_time_ms > 0
        assert assembled_context.assembly_time_ms > 0
        assert generated_response.generation_time_ms > 0
        assert assembled_context.total_tokens > 0
        assert generated_response.token_count > 0
        assert 0 <= generated_response.confidence_score <= 1
        assert 0 <= generated_response.factual_accuracy_score <= 1
