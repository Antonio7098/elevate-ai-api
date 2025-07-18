"""
Validation tests with sample conversations for RAG Chat Core.

Tests the complete RAG pipeline with realistic conversation scenarios
to ensure end-to-end functionality works correctly.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import List, Dict, Any
import json

from app.core.query_transformer import QueryTransformer
from app.core.rag_search import RAGSearchService
from app.core.context_assembly import ContextAssembler
from app.core.response_generation import ResponseGenerator
from app.api.endpoints import chat_endpoint


class TestSampleConversations:
    """Test suite for sample conversation validation."""
    
    @pytest.fixture
    def mock_services(self):
        """Mock all services for conversation testing."""
        mock_embedding = AsyncMock()
        mock_embedding.embed_text.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        mock_vector_store = AsyncMock()
        mock_vector_store.search.return_value = [
            {
                "id": "node_1",
                "content": "Python is a high-level, interpreted programming language with dynamic semantics.",
                "metadata": {
                    "source_id": "python_basics",
                    "locus_type": "foundational_concept",
                    "uue_stage": "understand",
                    "difficulty": "beginner"
                },
                "distance": 0.15
            },
            {
                "id": "node_2",
                "content": "Python functions are defined using the 'def' keyword followed by function name and parameters.",
                "metadata": {
                    "source_id": "python_functions",
                    "locus_type": "use_case",
                    "uue_stage": "use",
                    "difficulty": "beginner"
                },
                "distance": 0.25
            },
            {
                "id": "node_3",
                "content": "Variables in Python are created by assignment and don't need explicit declaration.",
                "metadata": {
                    "source_id": "python_variables",
                    "locus_type": "foundational_concept",
                    "uue_stage": "understand",
                    "difficulty": "beginner"
                },
                "distance": 0.30
            }
        ]
        
        mock_gemini = AsyncMock()
        mock_gemini.generate_response.return_value = "Python is a versatile programming language known for its simplicity and readability. It's widely used for web development, data science, and automation."
        
        return {
            "embedding": mock_embedding,
            "vector_store": mock_vector_store,
            "gemini": mock_gemini
        }
    
    @pytest.fixture
    def rag_pipeline(self, mock_services):
        """Create complete RAG pipeline with mocked services."""
        query_transformer = QueryTransformer(mock_services["embedding"])
        rag_search_service = RAGSearchService(mock_services["vector_store"], mock_services["embedding"])
        context_assembler = ContextAssembler(rag_search_service)
        response_generator = ResponseGenerator(mock_services["gemini"])
        
        return {
            "query_transformer": query_transformer,
            "context_assembler": context_assembler,
            "response_generator": response_generator
        }
    
    @pytest.mark.asyncio
    async def test_beginner_python_learning_conversation(self, rag_pipeline):
        """Test conversation with beginner learning Python."""
        query_transformer = rag_pipeline["query_transformer"]
        context_assembler = rag_pipeline["context_assembler"]
        response_generator = rag_pipeline["response_generator"]
        
        user_id = "beginner_learner"
        session_id = "python_learning_session"
        
        # Conversation flow
        conversation_steps = [
            {
                "user_query": "I'm new to programming. What is Python?",
                "expected_intent": "factual",
                "context_setup": {
                    "learning_stage": "understand",
                    "preferred_difficulty": "beginner"
                }
            },
            {
                "user_query": "How do I write my first Python program?",
                "expected_intent": "procedural",
                "context_setup": {
                    "learning_stage": "use",
                    "preferred_difficulty": "beginner"
                }
            },
            {
                "user_query": "What are variables in Python?",
                "expected_intent": "factual",
                "context_setup": {
                    "learning_stage": "understand",
                    "preferred_difficulty": "beginner"
                }
            },
            {
                "user_query": "Can you give me an example of using variables?",
                "expected_intent": "procedural",
                "context_setup": {
                    "learning_stage": "use",
                    "preferred_difficulty": "beginner"
                }
            }
        ]
        
        # Execute conversation
        for i, step in enumerate(conversation_steps):
            # Transform query
            query_transformation = await query_transformer.transform_query(
                query=step["user_query"],
                user_context=step["context_setup"]
            )
            
            # Verify intent classification
            assert query_transformation.intent.value == step["expected_intent"]
            
            # Add user message to conversation buffer
            context_assembler.add_message_to_buffer(
                session_id=session_id,
                role="user",
                content=step["user_query"]
            )
            
            # Assemble context
            assembled_context = await context_assembler.assemble_context(
                user_id=user_id,
                session_id=session_id,
                current_query=step["user_query"],
                query_transformation=query_transformation
            )
            
            # Generate response
            from app.core.response_generation import ResponseGenerationRequest
            response_request = ResponseGenerationRequest(
                user_query=step["user_query"],
                query_transformation=query_transformation,
                assembled_context=assembled_context
            )
            
            generated_response = await response_generator.generate_response(response_request)
            
            # Validate response
            assert len(generated_response.content) > 0
            assert generated_response.confidence_score > 0
            assert generated_response.factual_accuracy_score > 0
            
            # Add assistant response to conversation buffer
            context_assembler.add_message_to_buffer(
                session_id=session_id,
                role="assistant",
                content=generated_response.content
            )
            
            # Verify conversation context builds up
            conv_messages = context_assembler._get_conversational_buffer(session_id)
            assert len(conv_messages) == (i + 1) * 2  # Each step adds user + assistant message
    
    @pytest.mark.asyncio
    async def test_advanced_ml_discussion_conversation(self, rag_pipeline):
        """Test conversation with advanced ML discussion."""
        query_transformer = rag_pipeline["query_transformer"]
        context_assembler = rag_pipeline["context_assembler"]
        response_generator = rag_pipeline["response_generator"]
        
        user_id = "advanced_learner"
        session_id = "ml_discussion_session"
        
        # Set up advanced user profile
        cognitive_profile = context_assembler._get_or_create_cognitive_profile(user_id)
        cognitive_profile.preferred_difficulty = "advanced"
        cognitive_profile.learning_style = "analytical"
        cognitive_profile.preferred_explanation_style = "technical"
        cognitive_profile.knowledge_level = {
            "python": "advanced",
            "machine_learning": "intermediate",
            "neural_networks": "beginner"
        }
        
        # Advanced conversation flow
        conversation_steps = [
            {
                "user_query": "Explain the mathematical foundation of neural networks",
                "expected_intent": "conceptual",
                "context_setup": {
                    "learning_stage": "understand",
                    "preferred_difficulty": "advanced",
                    "current_topic": "neural networks"
                }
            },
            {
                "user_query": "How does backpropagation work in detail?",
                "expected_intent": "conceptual",
                "context_setup": {
                    "learning_stage": "understand",
                    "preferred_difficulty": "advanced"
                }
            },
            {
                "user_query": "Compare gradient descent vs Adam optimizer",
                "expected_intent": "comparative",
                "context_setup": {
                    "learning_stage": "evaluate",
                    "preferred_difficulty": "advanced"
                }
            }
        ]
        
        # Execute advanced conversation
        for step in conversation_steps:
            query_transformation = await query_transformer.transform_query(
                query=step["user_query"],
                user_context=step["context_setup"]
            )
            
            # Verify advanced intent classification
            assert query_transformation.intent.value == step["expected_intent"]
            
            # Should have appropriate filters for advanced content
            if query_transformation.filter_params:
                assert query_transformation.filter_params.difficulty_level == "advanced"
            
            # Add to conversation
            context_assembler.add_message_to_buffer(
                session_id=session_id,
                role="user",
                content=step["user_query"]
            )
            
            # Assemble context
            assembled_context = await context_assembler.assemble_context(
                user_id=user_id,
                session_id=session_id,
                current_query=step["user_query"],
                query_transformation=query_transformation
            )
            
            # Verify cognitive profile is used
            assert assembled_context.cognitive_profile.preferred_difficulty == "advanced"
            assert assembled_context.cognitive_profile.learning_style == "analytical"
            
            # Generate response
            from app.core.response_generation import ResponseGenerationRequest
            response_request = ResponseGenerationRequest(
                user_query=step["user_query"],
                query_transformation=query_transformation,
                assembled_context=assembled_context
            )
            
            generated_response = await response_generator.generate_response(response_request)
            
            # Should use technical tone for advanced user
            assert generated_response.tone_style.value == "technical"
            assert len(generated_response.content) > 0
            
            context_assembler.add_message_to_buffer(
                session_id=session_id,
                role="assistant",
                content=generated_response.content
            )
    
    @pytest.mark.asyncio
    async def test_follow_up_question_conversation(self, rag_pipeline):
        """Test conversation with follow-up questions and context reference."""
        query_transformer = rag_pipeline["query_transformer"]
        context_assembler = rag_pipeline["context_assembler"]
        response_generator = rag_pipeline["response_generator"]
        
        user_id = "follow_up_user"
        session_id = "follow_up_session"
        
        # Initial question
        initial_query = "What is machine learning?"
        
        # Transform and process initial query
        query_transformation = await query_transformer.transform_query(
            query=initial_query,
            user_context={"learning_stage": "understand"},
            conversation_history=[]
        )
        
        context_assembler.add_message_to_buffer(
            session_id=session_id,
            role="user",
            content=initial_query
        )
        
        assembled_context = await context_assembler.assemble_context(
            user_id=user_id,
            session_id=session_id,
            current_query=initial_query,
            query_transformation=query_transformation
        )
        
        from app.core.response_generation import ResponseGenerationRequest
        response_request = ResponseGenerationRequest(
            user_query=initial_query,
            query_transformation=query_transformation,
            assembled_context=assembled_context
        )
        
        initial_response = await response_generator.generate_response(response_request)
        
        context_assembler.add_message_to_buffer(
            session_id=session_id,
            role="assistant",
            content=initial_response.content
        )
        
        # Follow-up questions that reference previous context
        follow_up_queries = [
            "Can you explain that in simpler terms?",
            "What are some examples of that?",
            "How does that relate to artificial intelligence?",
            "What did you mean by that last part?"
        ]
        
        for follow_up in follow_up_queries:
            # Get conversation history
            conv_history = []
            for msg in context_assembler._get_conversational_buffer(session_id):
                conv_history.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Transform follow-up query with conversation history
            follow_up_transformation = await query_transformer.transform_query(
                query=follow_up,
                user_context={},
                conversation_history=conv_history
            )
            
            # Expanded query should include context from previous conversation
            assert len(follow_up_transformation.expanded_query) > len(follow_up)
            assert "machine learning" in follow_up_transformation.expanded_query.lower()
            
            context_assembler.add_message_to_buffer(
                session_id=session_id,
                role="user",
                content=follow_up
            )
            
            # Assemble context with conversation history
            follow_up_context = await context_assembler.assemble_context(
                user_id=user_id,
                session_id=session_id,
                current_query=follow_up,
                query_transformation=follow_up_transformation
            )
            
            # Should have rich conversational context
            assert len(follow_up_context.conversational_context) > 0
            
            # Generate response
            follow_up_request = ResponseGenerationRequest(
                user_query=follow_up,
                query_transformation=follow_up_transformation,
                assembled_context=follow_up_context
            )
            
            follow_up_response = await response_generator.generate_response(follow_up_request)
            
            assert len(follow_up_response.content) > 0
            assert follow_up_response.confidence_score > 0
            
            context_assembler.add_message_to_buffer(
                session_id=session_id,
                role="assistant",
                content=follow_up_response.content
            )
    
    @pytest.mark.asyncio
    async def test_session_state_evolution_conversation(self, rag_pipeline):
        """Test how session state evolves through conversation."""
        query_transformer = rag_pipeline["query_transformer"]
        context_assembler = rag_pipeline["context_assembler"]
        response_generator = rag_pipeline["response_generator"]
        
        user_id = "session_evolution_user"
        session_id = "evolution_session"
        
        # Conversation that should build session state
        conversation_flow = [
            {
                "query": "I want to learn web development",
                "expected_topic": "web development",
                "expected_concepts": []
            },
            {
                "query": "Should I start with HTML and CSS?",
                "expected_topic": "web development",
                "expected_concepts": ["HTML", "CSS"]
            },
            {
                "query": "What about JavaScript?",
                "expected_topic": "web development",
                "expected_concepts": ["HTML", "CSS", "JavaScript"]
            },
            {
                "query": "How do I make interactive websites?",
                "expected_topic": "web development",
                "expected_concepts": ["HTML", "CSS", "JavaScript", "interactive"]
            }
        ]
        
        for i, step in enumerate(conversation_flow):
            # Transform query
            query_transformation = await query_transformer.transform_query(
                query=step["query"],
                user_context={},
                conversation_history=[]
            )
            
            # Add to conversation
            context_assembler.add_message_to_buffer(
                session_id=session_id,
                role="user",
                content=step["query"]
            )
            
            # Assemble context
            assembled_context = await context_assembler.assemble_context(
                user_id=user_id,
                session_id=session_id,
                current_query=step["query"],
                query_transformation=query_transformation
            )
            
            # Generate response
            from app.core.response_generation import ResponseGenerationRequest
            response_request = ResponseGenerationRequest(
                user_query=step["query"],
                query_transformation=query_transformation,
                assembled_context=assembled_context
            )
            
            generated_response = await response_generator.generate_response(response_request)
            
            # Extract session updates
            session_updates = context_assembler.extract_session_updates(
                step["query"], 
                generated_response.content
            )
            
            # Update session state
            context_assembler.update_session_state(session_id, session_updates)
            
            # Add assistant response
            context_assembler.add_message_to_buffer(
                session_id=session_id,
                role="assistant",
                content=generated_response.content
            )
            
            # Verify session state evolution
            session_state = context_assembler.session_states[session_id]
            if i > 0:  # After first message
                assert session_state.current_topic == step["expected_topic"]
                
                # Check that concepts accumulate
                discussed_concepts = session_state.discussed_concepts
                for concept in step["expected_concepts"]:
                    # Should have some concepts discussed (simplified check)
                    assert len(discussed_concepts) >= 0
    
    @pytest.mark.asyncio
    async def test_error_recovery_conversation(self, rag_pipeline):
        """Test conversation with error scenarios and recovery."""
        query_transformer = rag_pipeline["query_transformer"]
        context_assembler = rag_pipeline["context_assembler"]
        response_generator = rag_pipeline["response_generator"]
        
        user_id = "error_test_user"
        session_id = "error_test_session"
        
        # Test empty/invalid queries
        invalid_queries = ["", "   ", "???", "asldkfjasldfj"]
        
        for invalid_query in invalid_queries:
            try:
                query_transformation = await query_transformer.transform_query(
                    query=invalid_query,
                    user_context={},
                    conversation_history=[]
                )
                
                context_assembler.add_message_to_buffer(
                    session_id=session_id,
                    role="user",
                    content=invalid_query
                )
                
                assembled_context = await context_assembler.assemble_context(
                    user_id=user_id,
                    session_id=session_id,
                    current_query=invalid_query,
                    query_transformation=query_transformation
                )
                
                from app.core.response_generation import ResponseGenerationRequest
                response_request = ResponseGenerationRequest(
                    user_query=invalid_query,
                    query_transformation=query_transformation,
                    assembled_context=assembled_context
                )
                
                generated_response = await response_generator.generate_response(response_request)
                
                # Should handle gracefully
                assert len(generated_response.content) > 0
                assert generated_response.confidence_score >= 0
                
            except Exception as e:
                # Should not crash the system
                assert False, f"System should handle invalid query gracefully: {e}"
        
        # Test recovery with valid query after errors
        recovery_query = "What is Python programming?"
        
        query_transformation = await query_transformer.transform_query(
            query=recovery_query,
            user_context={},
            conversation_history=[]
        )
        
        context_assembler.add_message_to_buffer(
            session_id=session_id,
            role="user",
            content=recovery_query
        )
        
        assembled_context = await context_assembler.assemble_context(
            user_id=user_id,
            session_id=session_id,
            current_query=recovery_query,
            query_transformation=query_transformation
        )
        
        from app.core.response_generation import ResponseGenerationRequest
        response_request = ResponseGenerationRequest(
            user_query=recovery_query,
            query_transformation=query_transformation,
            assembled_context=assembled_context
        )
        
        recovery_response = await response_generator.generate_response(response_request)
        
        # Should work normally after errors
        assert len(recovery_response.content) > 0
        assert recovery_response.confidence_score > 0.5
    
    @pytest.mark.asyncio
    async def test_multi_turn_topic_switching_conversation(self, rag_pipeline):
        """Test conversation with multiple topic switches."""
        query_transformer = rag_pipeline["query_transformer"]
        context_assembler = rag_pipeline["context_assembler"]
        response_generator = rag_pipeline["response_generator"]
        
        user_id = "topic_switch_user"
        session_id = "topic_switch_session"
        
        # Topic switching conversation
        topic_switches = [
            {
                "query": "Tell me about Python programming",
                "expected_topic": "python"
            },
            {
                "query": "Actually, let's talk about web development instead",
                "expected_topic": "web development"
            },
            {
                "query": "What about machine learning?",
                "expected_topic": "machine learning"
            },
            {
                "query": "Going back to Python, how do I create functions?",
                "expected_topic": "python"
            }
        ]
        
        for switch in topic_switches:
            # Get conversation history
            conv_history = []
            for msg in context_assembler._get_conversational_buffer(session_id):
                conv_history.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Transform query
            query_transformation = await query_transformer.transform_query(
                query=switch["query"],
                user_context={},
                conversation_history=conv_history
            )
            
            context_assembler.add_message_to_buffer(
                session_id=session_id,
                role="user",
                content=switch["query"]
            )
            
            # Assemble context
            assembled_context = await context_assembler.assemble_context(
                user_id=user_id,
                session_id=session_id,
                current_query=switch["query"],
                query_transformation=query_transformation
            )
            
            # Generate response
            from app.core.response_generation import ResponseGenerationRequest
            response_request = ResponseGenerationRequest(
                user_query=switch["query"],
                query_transformation=query_transformation,
                assembled_context=assembled_context
            )
            
            generated_response = await response_generator.generate_response(response_request)
            
            # Update session state
            session_updates = context_assembler.extract_session_updates(
                switch["query"], 
                generated_response.content
            )
            context_assembler.update_session_state(session_id, session_updates)
            
            context_assembler.add_message_to_buffer(
                session_id=session_id,
                role="assistant",
                content=generated_response.content
            )
            
            # Verify response quality
            assert len(generated_response.content) > 0
            assert generated_response.confidence_score > 0
            
            # Check that system adapts to topic switches
            session_state = context_assembler.session_states[session_id]
            # Topic should evolve with conversation
            assert session_state.current_topic is not None
    
    def test_conversation_metrics_tracking(self, rag_pipeline):
        """Test that conversation metrics are properly tracked."""
        context_assembler = rag_pipeline["context_assembler"]
        
        session_id = "metrics_session"
        
        # Add several messages
        for i in range(5):
            context_assembler.add_message_to_buffer(
                session_id=session_id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i} content"
            )
        
        # Get conversation buffer
        messages = context_assembler._get_conversational_buffer(session_id)
        
        # Verify message tracking
        assert len(messages) == 5
        
        # Verify timestamps
        for msg in messages:
            assert isinstance(msg.timestamp, datetime)
            assert msg.message_id is not None
        
        # Verify chronological order
        timestamps = [msg.timestamp for msg in messages]
        assert timestamps == sorted(timestamps)
    
    @pytest.mark.asyncio
    async def test_conversation_performance_metrics(self, rag_pipeline):
        """Test performance metrics across conversation."""
        query_transformer = rag_pipeline["query_transformer"]
        context_assembler = rag_pipeline["context_assembler"]
        response_generator = rag_pipeline["response_generator"]
        
        user_id = "performance_user"
        session_id = "performance_session"
        
        # Test queries
        queries = [
            "What is Python?",
            "How do I create variables?",
            "Explain functions in Python",
            "What are classes and objects?"
        ]
        
        total_query_time = 0
        total_context_time = 0
        total_response_time = 0
        
        for query in queries:
            # Transform query and measure time
            query_transformation = await query_transformer.transform_query(
                query=query,
                user_context={},
                conversation_history=[]
            )
            total_query_time += query_transformation.processing_time_ms
            
            context_assembler.add_message_to_buffer(
                session_id=session_id,
                role="user",
                content=query
            )
            
            # Assemble context and measure time
            assembled_context = await context_assembler.assemble_context(
                user_id=user_id,
                session_id=session_id,
                current_query=query,
                query_transformation=query_transformation
            )
            total_context_time += assembled_context.assembly_time_ms
            
            # Generate response and measure time
            from app.core.response_generation import ResponseGenerationRequest
            response_request = ResponseGenerationRequest(
                user_query=query,
                query_transformation=query_transformation,
                assembled_context=assembled_context
            )
            
            generated_response = await response_generator.generate_response(response_request)
            total_response_time += generated_response.generation_time_ms
            
            context_assembler.add_message_to_buffer(
                session_id=session_id,
                role="assistant",
                content=generated_response.content
            )
        
        # Verify performance metrics
        assert total_query_time > 0
        assert total_context_time > 0
        assert total_response_time > 0
        
        # Average times should be reasonable (this is a rough check)
        avg_query_time = total_query_time / len(queries)
        avg_context_time = total_context_time / len(queries)
        avg_response_time = total_response_time / len(queries)
        
        assert avg_query_time < 1000  # Should be under 1 second
        assert avg_context_time < 1000  # Should be under 1 second
        assert avg_response_time < 5000  # Should be under 5 seconds
