"""
Unit tests for Response Generation component.

Tests prompt assembly, LLM response generation, factual accuracy checking,
and response formatting.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import List, Dict, Any

from app.core.response_generation import (
    ResponseGenerator,
    ResponseGenerationRequest,
    GeneratedResponse,
    ResponseType,
    ToneStyle,
    PromptTemplate
)
from app.core.query_transformer import QueryTransformation, QueryIntent
from app.core.context_assembly import AssembledContext, ConversationMessage, SessionState, CognitiveProfile
from app.core.rag_search import RAGSearchResult


class TestResponseGenerator:
    """Test suite for ResponseGenerator class."""
    
    @pytest.fixture
    def mock_gemini_service(self):
        """Mock Gemini service."""
        mock_service = AsyncMock()
        mock_service.generate_response.return_value = "This is a generated response about Python programming."
        return mock_service
    
    @pytest.fixture
    def response_generator(self, mock_gemini_service):
        """Create ResponseGenerator instance."""
        return ResponseGenerator(mock_gemini_service)
    
    @pytest.fixture
    def sample_query_transformation(self):
        """Sample query transformation."""
        return QueryTransformation(
            original_query="What is Python?",
            normalized_query="what is python",
            intent=QueryIntent.FACTUAL,
            expanded_query="what is python programming language",
            filter_params=None,
            optimization=None,
            processing_time_ms=10.0,
            metadata={}
        )
    
    @pytest.fixture
    def sample_assembled_context(self):
        """Sample assembled context."""
        return AssembledContext(
            conversational_context=[
                ConversationMessage(
                    role="user",
                    content="I want to learn programming",
                    timestamp=datetime.utcnow(),
                    message_id="msg_1"
                ),
                ConversationMessage(
                    role="assistant",
                    content="Great! Let's start with Python.",
                    timestamp=datetime.utcnow(),
                    message_id="msg_2"
                )
            ],
            session_context=SessionState(
                session_id="test_session",
                user_id="test_user",
                current_topic="Python Programming",
                discussed_concepts=["variables", "functions"],
                learning_objectives=["Learn Python basics"]
            ),
            retrieved_knowledge=[
                RAGSearchResult(
                    content="Python is an interpreted, high-level programming language",
                    source_id="source_1",
                    locus_type="foundational_concept",
                    uue_stage="understand",
                    final_score=0.9,
                    metadata={"difficulty": "beginner"}
                ),
                RAGSearchResult(
                    content="Python has simple, readable syntax",
                    source_id="source_2",
                    locus_type="use_case",
                    uue_stage="use",
                    final_score=0.8,
                    metadata={"difficulty": "beginner"}
                )
            ],
            cognitive_profile=CognitiveProfile(
                user_id="test_user",
                learning_style="visual",
                preferred_difficulty="beginner",
                knowledge_level={"python": "beginner"},
                learning_pace="medium",
                preferred_explanation_style="examples"
            ),
            context_summary="User is learning Python programming basics",
            total_tokens=150,
            assembly_time_ms=25.0,
            context_quality_score=0.85
        )
    
    def test_response_type_determination(self, response_generator):
        """Test response type determination based on query intent."""
        test_cases = [
            (QueryIntent.FACTUAL, ResponseType.EXPLANATION),
            (QueryIntent.CONCEPTUAL, ResponseType.EXPLANATION),
            (QueryIntent.PROCEDURAL, ResponseType.EXPLANATION),
            (QueryIntent.COMPARATIVE, ResponseType.EXPLANATION),
            (QueryIntent.ANALYTICAL, ResponseType.CLARIFICATION),
            (QueryIntent.CREATIVE, ResponseType.ENCOURAGEMENT)
        ]
        
        for intent, expected_type in test_cases:
            response_type = response_generator._determine_response_type(intent)
            assert response_type == expected_type
    
    def test_tone_style_determination(self, response_generator):
        """Test tone style determination from cognitive profile."""
        test_cases = [
            ("examples", ToneStyle.CONVERSATIONAL),
            ("detailed", ToneStyle.FORMAL),
            ("simple", ToneStyle.SIMPLIFIED),
            ("technical", ToneStyle.TECHNICAL),
            ("encouraging", ToneStyle.ENCOURAGING),
            ("questioning", ToneStyle.SOCRATIC)
        ]
        
        for explanation_style, expected_tone in test_cases:
            profile = CognitiveProfile(
                user_id="test_user",
                preferred_explanation_style=explanation_style
            )
            tone = response_generator._determine_tone_style(profile)
            assert tone == expected_tone
    
    def test_prompt_template_selection(self, response_generator):
        """Test prompt template selection based on response type."""
        for response_type in ResponseType:
            template = response_generator._get_prompt_template(response_type)
            assert isinstance(template, PromptTemplate)
            assert template.name == response_type.value
            assert "{context}" in template.template
            assert "{query}" in template.template
    
    def test_prompt_assembly(self, response_generator, sample_query_transformation, sample_assembled_context):
        """Test prompt assembly with context and query."""
        response_type = ResponseType.EXPLANATION
        tone_style = ToneStyle.CONVERSATIONAL
        
        prompt = response_generator._assemble_prompt(
            query_transformation=sample_query_transformation,
            assembled_context=sample_assembled_context,
            response_type=response_type,
            tone_style=tone_style
        )
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "What is Python?" in prompt
        assert "Python Programming" in prompt
        assert "variables, functions" in prompt
        assert "Python is an interpreted" in prompt
        assert "conversational" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_generate_response_basic(self, response_generator, sample_query_transformation, sample_assembled_context):
        """Test basic response generation."""
        request = ResponseGenerationRequest(
            user_query="What is Python?",
            query_transformation=sample_query_transformation,
            assembled_context=sample_assembled_context,
            max_tokens=500,
            temperature=0.7
        )
        
        response = await response_generator.generate_response(request)
        
        assert isinstance(response, GeneratedResponse)
        assert len(response.content) > 0
        assert response.response_type == ResponseType.EXPLANATION
        assert response.tone_style == ToneStyle.CONVERSATIONAL
        assert 0 <= response.confidence_score <= 1
        assert 0 <= response.factual_accuracy_score <= 1
        assert response.token_count > 0
        assert response.generation_time_ms > 0
        assert len(response.sources) > 0
    
    @pytest.mark.asyncio
    async def test_generate_response_procedural(self, response_generator, sample_assembled_context):
        """Test response generation for procedural queries."""
        procedural_query = QueryTransformation(
            original_query="How do I create a Python function?",
            normalized_query="how do i create a python function",
            intent=QueryIntent.PROCEDURAL,
            expanded_query="how do i create a python function definition",
            filter_params=None,
            optimization=None,
            processing_time_ms=10.0,
            metadata={}
        )
        
        request = ResponseGenerationRequest(
            user_query="How do I create a Python function?",
            query_transformation=procedural_query,
            assembled_context=sample_assembled_context
        )
        
        response = await response_generator.generate_response(request)
        
        assert response.response_type == ResponseType.EXPLANATION
        assert response.confidence_score > 0
        assert len(response.content) > 0
    
    @pytest.mark.asyncio
    async def test_generate_response_different_tones(self, response_generator, sample_query_transformation, sample_assembled_context):
        """Test response generation with different tone styles."""
        tone_styles = [
            ("detailed", ToneStyle.FORMAL),
            ("technical", ToneStyle.TECHNICAL),
            ("simple", ToneStyle.SIMPLIFIED),
            ("encouraging", ToneStyle.ENCOURAGING)
        ]
        
        for explanation_style, expected_tone in tone_styles:
            # Update cognitive profile
            sample_assembled_context.cognitive_profile.preferred_explanation_style = explanation_style
            
            request = ResponseGenerationRequest(
                user_query="What is Python?",
                query_transformation=sample_query_transformation,
                assembled_context=sample_assembled_context
            )
            
            response = await response_generator.generate_response(request)
            
            assert response.tone_style == expected_tone
            assert len(response.content) > 0
    
    def test_factual_accuracy_checking(self, response_generator):
        """Test factual accuracy checking against retrieved knowledge."""
        generated_content = "Python is a high-level programming language with simple syntax."
        
        retrieved_knowledge = [
            RAGSearchResult(
                content="Python is an interpreted, high-level programming language",
                source_id="source_1",
                locus_type="foundational_concept",
                uue_stage="understand",
                final_score=0.9,
                metadata={}
            ),
            RAGSearchResult(
                content="Python has simple, readable syntax",
                source_id="source_2",
                locus_type="use_case",
                uue_stage="use",
                final_score=0.8,
                metadata={}
            )
        ]
        
        accuracy_score = response_generator._check_factual_accuracy(
            generated_content, retrieved_knowledge
        )
        
        assert 0 <= accuracy_score <= 1
        assert accuracy_score > 0.5  # Should be reasonably accurate
    
    def test_confidence_calculation(self, response_generator):
        """Test confidence score calculation."""
        context_quality = 0.8
        factual_accuracy = 0.9
        content_length = 100
        
        confidence = response_generator._calculate_confidence_score(
            context_quality, factual_accuracy, content_length
        )
        
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be reasonably confident
    
    def test_response_post_processing(self, response_generator):
        """Test response post-processing and formatting."""
        raw_response = "  Python is a programming language.  \n\n  It has simple syntax.  "
        
        processed = response_generator._post_process_response(raw_response)
        
        assert processed == "Python is a programming language.\n\nIt has simple syntax."
        assert not processed.startswith(" ")
        assert not processed.endswith(" ")
    
    def test_source_extraction(self, response_generator):
        """Test source extraction from retrieved knowledge."""
        retrieved_knowledge = [
            RAGSearchResult(
                content="Python is interpreted",
                source_id="blueprint_1",
                locus_type="foundational_concept",
                uue_stage="understand",
                final_score=0.9,
                metadata={"title": "Python Basics"}
            ),
            RAGSearchResult(
                content="Python has simple syntax",
                source_id="blueprint_2",
                locus_type="use_case",
                uue_stage="use",
                final_score=0.8,
                metadata={"title": "Python Features"}
            )
        ]
        
        sources = response_generator._extract_sources(retrieved_knowledge)
        
        assert len(sources) == 2
        assert sources[0]["source_id"] == "blueprint_1"
        assert sources[0]["title"] == "Python Basics"
        assert sources[1]["source_id"] == "blueprint_2"
        assert sources[1]["title"] == "Python Features"
    
    def test_token_counting(self, response_generator):
        """Test token counting for generated responses."""
        test_texts = [
            "Short response",
            "This is a longer response with multiple sentences and words.",
            "Very long response " * 50
        ]
        
        for text in test_texts:
            token_count = response_generator._count_tokens(text)
            assert token_count > 0
            assert isinstance(token_count, int)
            # Rough approximation: longer texts should have more tokens
            assert token_count >= len(text.split()) * 0.5
    
    @pytest.mark.asyncio
    async def test_error_handling_llm_failure(self, response_generator, sample_query_transformation, sample_assembled_context):
        """Test error handling when LLM fails."""
        # Mock LLM to raise an exception
        response_generator.gemini_service.generate_response.side_effect = Exception("LLM service error")
        
        request = ResponseGenerationRequest(
            user_query="What is Python?",
            query_transformation=sample_query_transformation,
            assembled_context=sample_assembled_context
        )
        
        response = await response_generator.generate_response(request)
        
        # Should return fallback response
        assert "trouble generating" in response.content.lower()
        assert response.confidence_score < 0.5
        assert response.factual_accuracy_score < 0.5
    
    @pytest.mark.asyncio
    async def test_response_with_empty_context(self, response_generator, sample_query_transformation):
        """Test response generation with empty context."""
        empty_context = AssembledContext(
            conversational_context=[],
            session_context=SessionState(
                session_id="empty_session",
                user_id="empty_user"
            ),
            retrieved_knowledge=[],
            cognitive_profile=CognitiveProfile(user_id="empty_user"),
            context_summary="",
            total_tokens=0,
            assembly_time_ms=0.0,
            context_quality_score=0.0
        )
        
        request = ResponseGenerationRequest(
            user_query="What is Python?",
            query_transformation=sample_query_transformation,
            assembled_context=empty_context
        )
        
        response = await response_generator.generate_response(request)
        
        # Should still generate a response
        assert len(response.content) > 0
        assert response.confidence_score >= 0
    
    def test_prompt_template_formatting(self, response_generator):
        """Test prompt template formatting."""
        template = PromptTemplate(
            name="test_template",
            template="Query: {query}\nContext: {context}\nTone: {tone_style}",
            description="Test template"
        )
        
        formatted = response_generator._format_template(
            template=template,
            query="What is Python?",
            context="Python is a programming language",
            tone_style="conversational"
        )
        
        assert "Query: What is Python?" in formatted
        assert "Context: Python is a programming language" in formatted
        assert "Tone: conversational" in formatted
    
    @pytest.mark.asyncio
    async def test_response_with_different_difficulties(self, response_generator, sample_query_transformation, sample_assembled_context):
        """Test response generation with different difficulty levels."""
        difficulty_levels = ["beginner", "intermediate", "advanced"]
        
        for difficulty in difficulty_levels:
            # Update cognitive profile
            sample_assembled_context.cognitive_profile.preferred_difficulty = difficulty
            
            request = ResponseGenerationRequest(
                user_query="What is Python?",
                query_transformation=sample_query_transformation,
                assembled_context=sample_assembled_context
            )
            
            response = await response_generator.generate_response(request)
            
            assert len(response.content) > 0
            assert response.confidence_score > 0
    
    def test_response_metadata_generation(self, response_generator):
        """Test response metadata generation."""
        metadata = response_generator._generate_response_metadata(
            query_intent=QueryIntent.FACTUAL,
            context_quality=0.8,
            processing_time=150.0,
            token_usage={"prompt_tokens": 100, "completion_tokens": 50}
        )
        
        assert metadata["query_intent"] == "factual"
        assert metadata["context_quality"] == 0.8
        assert metadata["processing_time_ms"] == 150.0
        assert metadata["token_usage"]["prompt_tokens"] == 100
        assert metadata["token_usage"]["completion_tokens"] == 50
    
    @pytest.mark.asyncio
    async def test_response_with_conversation_context(self, response_generator, sample_query_transformation, sample_assembled_context):
        """Test response generation with rich conversation context."""
        # Add more conversation history
        sample_assembled_context.conversational_context.extend([
            ConversationMessage(
                role="user",
                content="Tell me about variables in Python",
                timestamp=datetime.utcnow(),
                message_id="msg_3"
            ),
            ConversationMessage(
                role="assistant",
                content="Variables in Python are used to store data values.",
                timestamp=datetime.utcnow(),
                message_id="msg_4"
            )
        ])
        
        request = ResponseGenerationRequest(
            user_query="Can you give me examples?",
            query_transformation=sample_query_transformation,
            assembled_context=sample_assembled_context
        )
        
        response = await response_generator.generate_response(request)
        
        # Should incorporate conversation context
        assert len(response.content) > 0
        assert response.confidence_score > 0
    
    def test_response_serialization(self, response_generator):
        """Test GeneratedResponse serialization."""
        response = GeneratedResponse(
            content="This is a test response",
            response_type=ResponseType.EXPLANATION,
            tone_style=ToneStyle.CONVERSATIONAL,
            confidence_score=0.85,
            factual_accuracy_score=0.90,
            token_count=25,
            generation_time_ms=150.0,
            sources=[{"source_id": "test_source", "title": "Test Source"}],
            metadata={"test": "value"}
        )
        
        # Test that the object can be serialized
        dict_repr = response.__dict__
        assert "content" in dict_repr
        assert "response_type" in dict_repr
        assert "confidence_score" in dict_repr
        assert "sources" in dict_repr
    
    @pytest.mark.asyncio
    async def test_response_with_max_tokens_limit(self, response_generator, sample_query_transformation, sample_assembled_context):
        """Test response generation with token limits."""
        request = ResponseGenerationRequest(
            user_query="What is Python?",
            query_transformation=sample_query_transformation,
            assembled_context=sample_assembled_context,
            max_tokens=50,  # Low limit
            temperature=0.7
        )
        
        response = await response_generator.generate_response(request)
        
        # Should respect token limit
        assert len(response.content) > 0
        assert response.token_count <= 50 or response.token_count <= 60  # Some tolerance
    
    @pytest.mark.asyncio
    async def test_response_with_temperature_variations(self, response_generator, sample_query_transformation, sample_assembled_context):
        """Test response generation with different temperature values."""
        temperatures = [0.1, 0.5, 0.9]
        
        for temp in temperatures:
            request = ResponseGenerationRequest(
                user_query="What is Python?",
                query_transformation=sample_query_transformation,
                assembled_context=sample_assembled_context,
                temperature=temp
            )
            
            response = await response_generator.generate_response(request)
            
            assert len(response.content) > 0
            assert response.confidence_score > 0
