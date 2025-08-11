"""
Unit tests for Context Assembly component.

Tests multi-tier memory system, context assembly, and context quality scoring.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import List, Dict, Any

from app.core.context_assembly import (
    ContextAssembler,
    ConversationMessage,
    SessionState,
    CognitiveProfile,
    AssembledContext,
    ContextTier
)
from app.core.query_transformer import QueryTransformation, QueryIntent
from app.core.rag_search import RAGSearchResult


class TestContextAssembler:
    """Test suite for ContextAssembler class."""
    
    @pytest.fixture
    def mock_search_service(self):
        """Mock RAG search service."""
        mock_service = AsyncMock()
        mock_service.search.return_value = MagicMock(
            results=[
                RAGSearchResult(
                    content="Python is a programming language",
                    source_id="test_source_1",
                    locus_type="foundational_concept",
                    uue_stage="understand",
                    final_score=0.9,
                    metadata={"difficulty": "beginner"}
                ),
                RAGSearchResult(
                    content="Machine learning algorithms learn from data",
                    source_id="test_source_2", 
                    locus_type="use_case",
                    uue_stage="use",
                    final_score=0.8,
                    metadata={"difficulty": "intermediate"}
                )
            ]
        )
        return mock_service
    
    @pytest.fixture
    def context_assembler(self, mock_search_service):
        """Create ContextAssembler instance."""
        return ContextAssembler(mock_search_service)
    
    @pytest.fixture
    def sample_query_transformation(self):
        """Sample query transformation for testing."""
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
    
    def test_conversation_message_creation(self):
        """Test ConversationMessage creation."""
        message = ConversationMessage(
            role="user",
            content="Hello, how are you?",
            timestamp=datetime.utcnow(),
            message_id="test_msg_1",
            metadata={"test": "value"}
        )
        
        assert message.role == "user"
        assert message.content == "Hello, how are you?"
        assert isinstance(message.timestamp, datetime)
        assert message.message_id == "test_msg_1"
        assert message.metadata["test"] == "value"
    
    def test_session_state_creation(self):
        """Test SessionState creation."""
        session_state = SessionState(
            session_id="test_session",
            user_id="test_user",
            current_topic="Python programming",
            learning_objectives=["Learn Python basics"],
            discussed_concepts=["variables", "functions"],
            user_questions=["What is a variable?"],
            clarifications_needed=["Explain functions"],
            progress_indicators={"completion": 0.5},
            context_summary="Learning Python basics",
            last_updated=datetime.utcnow(),
            metadata={"level": "beginner"}
        )
        
        assert session_state.session_id == "test_session"
        assert session_state.user_id == "test_user"
        assert session_state.current_topic == "Python programming"
        assert "Learn Python basics" in session_state.learning_objectives
        assert "variables" in session_state.discussed_concepts
        assert "functions" in session_state.discussed_concepts
    
    def test_cognitive_profile_creation(self):
        """Test CognitiveProfile creation."""
        profile = CognitiveProfile(
            user_id="test_user",
            learning_style="visual",
            preferred_difficulty="intermediate",
            knowledge_level={"python": "beginner", "javascript": "advanced"},
            learning_pace="medium",
            preferred_explanation_style="examples",
            misconceptions=["Python is slow"],
            strengths=["Problem solving"],
            areas_for_improvement=["Code organization"],
            last_updated=datetime.utcnow()
        )
        
        assert profile.user_id == "test_user"
        assert profile.learning_style == "visual"
        assert profile.preferred_difficulty == "intermediate"
        assert profile.knowledge_level["python"] == "beginner"
        assert profile.learning_pace == "medium"
        assert profile.preferred_explanation_style == "examples"
    
    def test_add_message_to_buffer(self, context_assembler):
        """Test adding messages to conversation buffer."""
        session_id = "test_session"
        
        # Add first message
        context_assembler.add_message_to_buffer(
            session_id=session_id,
            role="user",
            content="Hello",
            metadata={"test": "value"}
        )
        
        # Add second message
        context_assembler.add_message_to_buffer(
            session_id=session_id,
            role="assistant",
            content="Hi there!",
            metadata={}
        )
        
        messages = context_assembler._get_conversational_buffer(session_id)
        
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[0].metadata["test"] == "value"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi there!"
    
    def test_buffer_size_limit(self, context_assembler):
        """Test conversation buffer size limit."""
        session_id = "test_session"
        max_messages = context_assembler.tier_config[ContextTier.CONVERSATIONAL_BUFFER]['max_messages']
        
        # Add more messages than the limit
        for i in range(max_messages + 5):
            context_assembler.add_message_to_buffer(
                session_id=session_id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}"
            )
        
        messages = context_assembler._get_conversational_buffer(session_id)
        
        # Should not exceed max_messages
        assert len(messages) == max_messages
        
        # Should keep the most recent messages
        assert messages[-1].content == f"Message {max_messages + 4}"
    
    def test_session_state_updates(self, context_assembler):
        """Test session state updates."""
        session_id = "test_session"
        user_id = "test_user"
        
        # Create session state
        session_state = context_assembler._get_or_create_session_state(user_id, session_id)
        original_update_time = session_state.last_updated
        
        # Update session state
        updates = {
            "current_topic": "Machine Learning",
            "discussed_concepts": ["neural networks", "algorithms"],
            "learning_objectives": ["Understand ML basics"]
        }
        
        context_assembler.update_session_state(session_id, updates)
        
        # Verify updates
        updated_session = context_assembler.session_states[session_id]
        assert updated_session.current_topic == "Machine Learning"
        assert "neural networks" in updated_session.discussed_concepts
        assert "algorithms" in updated_session.discussed_concepts
        assert "Understand ML basics" in updated_session.learning_objectives
        assert updated_session.last_updated > original_update_time
    
    def test_cognitive_profile_default_creation(self, context_assembler):
        """Test default cognitive profile creation."""
        user_id = "new_user"
        profile = context_assembler._get_or_create_cognitive_profile(user_id)
        
        assert profile.user_id == user_id
        assert profile.learning_style == "balanced"
        assert profile.preferred_difficulty == "intermediate"
        assert profile.learning_pace == "medium"
        assert profile.preferred_explanation_style == "detailed"
        assert profile.misconceptions == []
        assert profile.strengths == []
        assert profile.areas_for_improvement == []
    
    @pytest.mark.asyncio
    async def test_assemble_context_basic(self, context_assembler, sample_query_transformation):
        """Test basic context assembly."""
        user_id = "test_user"
        session_id = "test_session"
        current_query = "What is Python?"
        
        # Add some conversation history
        context_assembler.add_message_to_buffer(
            session_id=session_id,
            role="user",
            content="I want to learn programming"
        )
        context_assembler.add_message_to_buffer(
            session_id=session_id,
            role="assistant",
            content="Great! Let's start with Python."
        )
        
        # Assemble context
        assembled_context = await context_assembler.assemble_context(
            user_id=user_id,
            session_id=session_id,
            current_query=current_query,
            query_transformation=sample_query_transformation
        )
        
        assert isinstance(assembled_context, AssembledContext)
        assert len(assembled_context.conversational_context) == 2
        assert assembled_context.session_context.session_id == session_id
        assert assembled_context.session_context.user_id == user_id
        assert assembled_context.cognitive_profile.user_id == user_id
        assert len(assembled_context.retrieved_knowledge) > 0
        assert assembled_context.context_quality_score > 0
        assert assembled_context.total_tokens > 0
        assert assembled_context.assembly_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_context_assembly_with_session_updates(self, context_assembler, sample_query_transformation):
        """Test context assembly with session state updates."""
        user_id = "test_user"
        session_id = "test_session"
        
        # Update session state first
        updates = {
            "current_topic": "Python Programming",
            "discussed_concepts": ["variables", "functions", "loops"],
            "learning_objectives": ["Master Python basics"]
        }
        context_assembler.update_session_state(session_id, updates)
        
        # Assemble context
        assembled_context = await context_assembler.assemble_context(
            user_id=user_id,
            session_id=session_id,
            current_query="What are Python data types?",
            query_transformation=sample_query_transformation
        )
        
        # Check session context
        assert assembled_context.session_context.current_topic == "Python Programming"
        assert "variables" in assembled_context.session_context.discussed_concepts
        assert "functions" in assembled_context.session_context.discussed_concepts
        assert "loops" in assembled_context.session_context.discussed_concepts
        assert "Master Python basics" in assembled_context.session_context.learning_objectives
    
    def test_context_summary_creation(self, context_assembler):
        """Test context summary creation."""
        session_id = "test_session"
        
        # Add conversation messages
        context_assembler.add_message_to_buffer(
            session_id=session_id,
            role="user",
            content="What is machine learning?"
        )
        context_assembler.add_message_to_buffer(
            session_id=session_id,
            role="assistant",
            content="Machine learning is a subset of AI..."
        )
        
        # Update session state
        updates = {
            "current_topic": "Machine Learning",
            "discussed_concepts": ["algorithms", "neural networks"]
        }
        context_assembler.update_session_state(session_id, updates)
        
        # Get components
        conversational_context = context_assembler._get_conversational_buffer(session_id)
        session_state = context_assembler.session_states[session_id]
        retrieved_knowledge = []  # Empty for this test
        
        # Create summary
        summary = context_assembler._create_context_summary(
            conversational_context,
            session_state,
            retrieved_knowledge
        )
        
        assert "What is machine learning" in summary
        assert "Machine Learning" in summary
        assert "algorithms" in summary or "neural networks" in summary
    
    def test_token_calculation(self, context_assembler):
        """Test token calculation for context components."""
        session_id = "test_session"
        user_id = "test_user"
        
        # Add messages
        context_assembler.add_message_to_buffer(
            session_id=session_id,
            role="user",
            content="This is a test message with multiple words"
        )
        
        # Get components
        conversational_context = context_assembler._get_conversational_buffer(session_id)
        session_state = context_assembler._get_or_create_session_state(user_id, session_id)
        cognitive_profile = context_assembler._get_or_create_cognitive_profile(user_id)
        retrieved_knowledge = []
        
        # Calculate tokens
        total_tokens = context_assembler._calculate_total_tokens(
            conversational_context,
            session_state,
            retrieved_knowledge,
            cognitive_profile
        )
        
        assert total_tokens > 0
        assert isinstance(total_tokens, int)
    
    def test_context_pruning(self, context_assembler):
        """Test context pruning for token limits."""
        session_id = "test_session"
        
        # Create long conversation
        for i in range(10):
            context_assembler.add_message_to_buffer(
                session_id=session_id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"This is a very long message number {i} with lots of content " * 20
            )
        
        conversational_context = context_assembler._get_conversational_buffer(session_id)
        retrieved_knowledge = []
        
        # Prune context
        pruned_conv, pruned_knowledge = context_assembler._prune_context(
            conversational_context,
            retrieved_knowledge,
            target_tokens=500
        )
        
        # Should have fewer messages after pruning
        assert len(pruned_conv) < len(conversational_context)
        
        # Should keep most recent messages
        if pruned_conv:
            assert pruned_conv[-1].content == conversational_context[-1].content
    
    def test_context_quality_scoring(self, context_assembler, sample_query_transformation):
        """Test context quality scoring."""
        session_id = "test_session"
        
        # Add recent conversation
        context_assembler.add_message_to_buffer(
            session_id=session_id,
            role="user",
            content="Tell me about Python"
        )
        context_assembler.add_message_to_buffer(
            session_id=session_id,
            role="assistant",
            content="Python is a programming language"
        )
        
        # Update session state
        updates = {
            "current_topic": "Python Programming",
            "discussed_concepts": ["variables", "functions"]
        }
        context_assembler.update_session_state(session_id, updates)
        
        # Get components
        conversational_context = context_assembler._get_conversational_buffer(session_id)
        session_state = context_assembler.session_states[session_id]
        retrieved_knowledge = [
            RAGSearchResult(
                content="Python is interpreted",
                source_id="test_1",
                locus_type="foundational_concept",
                uue_stage="understand",
                final_score=0.9,
                metadata={}
            ),
            RAGSearchResult(
                content="Python has simple syntax",
                source_id="test_2",
                locus_type="use_case",
                uue_stage="use",
                final_score=0.8,
                metadata={}
            )
        ]
        
        # Calculate quality score
        quality_score = context_assembler._calculate_context_quality(
            conversational_context,
            session_state,
            retrieved_knowledge,
            sample_query_transformation
        )
        
        assert 0.0 <= quality_score <= 1.0
        assert quality_score > 0  # Should have some quality score
    
    def test_context_for_prompt_formatting(self, context_assembler):
        """Test context formatting for prompt inclusion."""
        session_id = "test_session"
        user_id = "test_user"
        
        # Add conversation
        context_assembler.add_message_to_buffer(
            session_id=session_id,
            role="user",
            content="What is Python?"
        )
        context_assembler.add_message_to_buffer(
            session_id=session_id,
            role="assistant",
            content="Python is a programming language"
        )
        
        # Update session state
        updates = {
            "current_topic": "Python Programming",
            "discussed_concepts": ["variables", "functions"]
        }
        context_assembler.update_session_state(session_id, updates)
        
        # Create assembled context
        assembled_context = AssembledContext(
            conversational_context=context_assembler._get_conversational_buffer(session_id),
            session_context=context_assembler.session_states[session_id],
            retrieved_knowledge=[
                RAGSearchResult(
                    content="Python is interpreted",
                    source_id="test_1",
                    locus_type="foundational_concept",
                    uue_stage="understand",
                    final_score=0.9,
                    metadata={}
                )
            ],
            cognitive_profile=context_assembler._get_or_create_cognitive_profile(user_id),
            context_summary="Test summary",
            total_tokens=100,
            assembly_time_ms=50.0,
            context_quality_score=0.8
        )
        
        # Format for prompt
        prompt_context = context_assembler.get_context_for_prompt(assembled_context)
        
        assert "=== CONVERSATION HISTORY ===" in prompt_context
        assert "=== CURRENT TOPIC ===" in prompt_context
        assert "=== DISCUSSED CONCEPTS ===" in prompt_context
        assert "=== RELEVANT KNOWLEDGE ===" in prompt_context
        assert "=== USER PROFILE ===" in prompt_context
        assert "What is Python?" in prompt_context
        assert "Python Programming" in prompt_context
        assert "variables, functions" in prompt_context
    
    def test_session_updates_extraction(self, context_assembler):
        """Test session state updates extraction from conversation."""
        user_message = "What is machine learning? I'm confused about neural networks."
        assistant_response = "Machine learning is a concept in AI that uses algorithms to learn from data. Neural networks are a type of algorithm."
        
        updates = context_assembler.extract_session_updates(user_message, assistant_response)
        
        # Should extract questions
        assert "user_questions" in updates
        assert len(updates["user_questions"]) > 0
        
        # Should extract concepts (simplified extraction)
        if "discussed_concepts" in updates:
            assert len(updates["discussed_concepts"]) > 0
    
    @pytest.mark.asyncio
    async def test_empty_context_handling(self, context_assembler, sample_query_transformation):
        """Test handling of empty context scenarios."""
        user_id = "empty_user"
        session_id = "empty_session"
        
        # Assemble context with no prior data
        assembled_context = await context_assembler.assemble_context(
            user_id=user_id,
            session_id=session_id,
            current_query="Hello",
            query_transformation=sample_query_transformation
        )
        
        # Should handle gracefully
        assert isinstance(assembled_context, AssembledContext)
        assert len(assembled_context.conversational_context) == 0
        assert assembled_context.session_context.session_id == session_id
        assert assembled_context.session_context.user_id == user_id
        assert assembled_context.cognitive_profile.user_id == user_id
        assert assembled_context.context_quality_score >= 0
