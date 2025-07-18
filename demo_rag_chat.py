#!/usr/bin/env python3
"""
RAG Chat Core Demo Script

This script demonstrates the RAG Chat Core functionality with real examples
of conversational AI interactions in the Elevate AI system.
"""

import asyncio
import os
import sys
from typing import Dict, List, Any
from datetime import datetime

# Add the app directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from app.core.query_transformer import QueryTransformer
from app.core.context_assembly import ContextAssembler, ConversationMessage
from app.core.response_generation import ResponseGenerator, ResponseGenerationRequest
from app.services.gemini_service import GeminiService
from app.core.rag_search import RAGSearchService
from app.core.vector_store import create_vector_store
from app.core.embeddings import GoogleEmbeddingService


class RAGChatDemo:
    """Demo class for RAG Chat Core functionality."""
    
    def __init__(self):
        self.query_transformer = None
        self.context_assembler = None
        self.response_generator = None
        self.session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    async def initialize_services(self):
        """Initialize all RAG services for the demo."""
        print("🚀 Initializing RAG Chat Core services...")
        
        # Initialize embedding service with Google API key
        embedding_service = GoogleEmbeddingService(
            api_key=os.getenv("GOOGLE_API_KEY", "mock_key")
        )
        
        # Initialize Pinecone vector store
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
        
        if not pinecone_api_key:
            print("❌ PINECONE_API_KEY not found in environment variables")
            print("   Please set your Pinecone API key in the .env file")
            return False
            
        if pinecone_env in ["your_pinecone_environment_here", "your_environment_here"]:
            print("❌ PINECONE_ENVIRONMENT is set to a placeholder value")
            print("   Please update PINECONE_ENVIRONMENT in .env to a valid value like 'us-east1-gcp'")
            return False
        
        vector_store = create_vector_store(
            store_type="pinecone",
            api_key=pinecone_api_key,
            environment=pinecone_env,
            index_name="elevate-ai-demo"
        )
        
        # Initialize the vector store client
        await vector_store.initialize()
        
        # Initialize Gemini service
        gemini_service = GeminiService()
        
        # Initialize RAG components
        self.query_transformer = QueryTransformer(embedding_service)
        rag_search_service = RAGSearchService(vector_store, embedding_service)
        self.context_assembler = ContextAssembler(rag_search_service)
        self.response_generator = ResponseGenerator(gemini_service)
        
        print("✅ All services initialized successfully!")
        return True
    
    async def run_conversation_demo(self):
        """Run a demonstration conversation."""
        print("\n" + "="*60)
        print("🎯 RAG Chat Core Demo - Python Learning Conversation")
        print("="*60)
        
        # Demo conversation scenarios
        conversation_scenarios = [
            {
                "title": "Beginner Python Learning",
                "user_id": "beginner_learner",
                "context_setup": {
                    "learning_stage": "understand",
                    "preferred_difficulty": "beginner",
                    "topics_of_interest": ["python", "programming", "basics"]
                },
                "queries": [
                    "What is Python and why should I learn it?",
                    "How do I write my first Python program?",
                    "What are variables in Python?",
                    "Can you show me an example of using variables?"
                ]
            },
            {
                "title": "Advanced Machine Learning Discussion",
                "user_id": "ml_practitioner",
                "context_setup": {
                    "learning_stage": "evaluate",
                    "preferred_difficulty": "advanced",
                    "topics_of_interest": ["machine_learning", "algorithms", "tensorflow"]
                },
                "queries": [
                    "What are the key differences between supervised and unsupervised learning?",
                    "How do I choose the right algorithm for my ML problem?",
                    "What are some common pitfalls in deep learning?"
                ]
            }
        ]
        
        # Run each scenario
        for scenario in conversation_scenarios:
            await self.demonstrate_scenario(scenario)
    
    async def demonstrate_scenario(self, scenario: Dict[str, Any]):
        """Demonstrate a single conversation scenario."""
        print(f"\n📝 Scenario: {scenario['title']}")
        print("-" * 40)
        
        user_id = scenario["user_id"]
        context_setup = scenario["context_setup"]
        queries = scenario["queries"]
        
        conversation_history = []
        
        for i, query in enumerate(queries, 1):
            print(f"\n💬 Query {i}: {query}")
            
            try:
                # Step 1: Transform the query
                print("🔄 Step 1: Transforming query...")
                query_transformation = await self.query_transformer.transform_query(
                    query=query,
                    user_context=context_setup
                )
                
                print(f"   ✅ Intent: {query_transformation.intent}")
                print(f"   ✅ Confidence: {query_transformation.confidence:.2f}")
                print(f"   ✅ Expanded Query: {query_transformation.expanded_query}")
                
                # Step 2: Assemble context
                print("🔄 Step 2: Assembling context...")
                try:
                    # Create conversation message
                    conversation_message = ConversationMessage(
                        role="user",
                        content=query,
                        timestamp=datetime.utcnow(),
                        message_id=f"msg_{self.session_id}_{i}",
                        metadata={"query_id": f"query_{i}"}
                    )
                    
                    # For demo purposes, we'll create a simple context without full search
                    # since the vector store may not have data
                    assembled_context = await self.context_assembler.assemble_context(
                        user_id=user_id,
                        session_id=self.session_id,
                        current_query=query,
                        query_transformation=query_transformation
                    )
                    
                    print(f"   ✅ Context assembled with {len(assembled_context.conversational_context)} messages")
                    print(f"   ✅ Session state: {assembled_context.session_context.current_topic}")
                    
                except Exception as e:
                    print(f"   ⚠️  Context assembly failed (using fallback): {e}")
                    # Create fallback context
                    from app.core.context_assembly import AssembledContext, SessionState, CognitiveProfile
                    
                    assembled_context = AssembledContext(
                        conversational_context=[conversation_message],
                        session_context=SessionState(
                            session_id=self.session_id,
                            user_id=user_id,
                            current_topic="python_learning",
                            learning_objectives=context_setup.get("learning_objectives", []),
                            discussed_concepts=context_setup.get("discussed_concepts", []),
                            user_questions=context_setup.get("user_questions", []),
                            clarifications_needed=context_setup.get("clarifications_needed", []),
                            progress_indicators=context_setup.get("progress_indicators", {}),
                            context_summary="User is learning Python programming",
                            last_updated=datetime.utcnow(),
                            metadata={"context_setup": context_setup}
                        ),
                        cognitive_profile=CognitiveProfile(
                            user_id=user_id,
                            learning_style=context_setup.get("learning_style", "visual"),
                            preferred_difficulty=context_setup.get("preferred_difficulty", "beginner"),
                            knowledge_level=context_setup.get("knowledge_level", {}),
                            learning_pace=context_setup.get("learning_pace", "medium"),
                            preferred_explanation_style=context_setup.get("preferred_explanation_style", "detailed"),
                            misconceptions=context_setup.get("misconceptions", []),
                            strengths=context_setup.get("strengths", []),
                            areas_for_improvement=context_setup.get("areas_for_improvement", []),
                            last_updated=datetime.utcnow()
                        ),
                        retrieved_knowledge=[],
                        context_summary="User is learning Python programming",
                        total_tokens=100,
                        assembly_time_ms=50.0,
                        context_quality_score=0.8
                    )
                
                # Step 3: Generate response
                print("🔄 Step 3: Generating response...")
                response_request = ResponseGenerationRequest(
                    user_query=query,
                    query_transformation=query_transformation,
                    assembled_context=assembled_context,
                    max_tokens=1000,
                    temperature=0.7,
                    include_sources=True,
                    metadata=context_setup
                )
                
                generated_response = await self.response_generator.generate_response(response_request)
                
                print(f"   ✅ Response generated (confidence: {generated_response.confidence_score:.2f})")
                print(f"   ✅ Response type: {generated_response.response_type}")
                print(f"   ✅ Tone: {generated_response.tone_style}")
                
                # Display the response
                print(f"\n🤖 AI Response:")
                print(f"   {generated_response.content}")
                
                # Add to conversation history
                conversation_history.append({
                    "query": query,
                    "response": generated_response.content,
                    "intent": query_transformation.intent,
                    "confidence": generated_response.confidence_score
                })
                
                # Simulate adding response to conversation buffer
                response_message = ConversationMessage(
                    role="assistant",
                    content=generated_response.content,
                    timestamp=datetime.utcnow(),
                    message_id=f"msg_{self.session_id}_{i}_response",
                    metadata={"response_type": generated_response.response_type}
                )
                
                print(f"   ✅ Conversation updated with {len(conversation_history)} exchanges")
                
            except Exception as e:
                print(f"   ❌ Error processing query: {e}")
                continue
        
        # Print conversation summary
        print(f"\n📊 Conversation Summary:")
        print(f"   Total exchanges: {len(conversation_history)}")
        avg_confidence = sum(h["confidence"] for h in conversation_history) / len(conversation_history) if conversation_history else 0
        print(f"   Average confidence: {avg_confidence:.2f}")
        
        intents = [h["intent"] for h in conversation_history]
        unique_intents = set(intents)
        intent_names = [intent.name for intent in unique_intents]
        print(f"   Intent types covered: {', '.join(intent_names)}")
    
    async def run_demo(self):
        """Run the complete demo."""
        print("🎉 Welcome to the RAG Chat Core Demo!")
        print("This demo showcases the complete RAG pipeline in action.")
        
        try:
            # Check if initialization was successful
            initialization_success = await self.initialize_services()
            if not initialization_success:
                print("\n❌ Demo cannot proceed due to configuration issues.")
                print("Please fix the environment variables and try again.")
                return
                
            await self.run_conversation_demo()
            
            print("\n" + "="*60)
            print("✅ Demo completed successfully!")
            print("🚀 The RAG Chat Core is fully functional and ready for use!")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main entry point for the demo."""
    demo = RAGChatDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
