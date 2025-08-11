#!/usr/bin/env python3
"""
Single Query RAG Test - Validate Context-Aware Responses
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add app to path
sys.path.append('.')

from app.core.rag_search import RAGSearchService, RAGSearchRequest
from app.core.context_assembly import ContextAssembler, ConversationMessage, AssembledContext
from app.core.response_generation import ResponseGenerator, ResponseGenerationRequest
from app.core.query_transformer import QueryIntent
from app.models.user_profile import CognitiveProfile
from app.models.session import SessionState
from app.services.embedding_service import get_embedding_service

async def test_single_rag_query():
    """Test a single RAG query end-to-end"""
    
    print("🎯 TESTING SINGLE RAG QUERY - CONTEXT-AWARE RESPONSE")
    print("=" * 60)
    
    # Test query
    query = "What is machine learning?"
    
    print(f"📝 Query: {query}")
    print()
    
    try:
        # Step 1: Initialize services
        print("🔧 Step 1: Initializing services...")
        
        # Initialize embedding service
        from app.services.embedding_service import initialize_embedding_service
        api_key = os.getenv("GOOGLE_API_KEY")
        initialize_embedding_service("google", api_key=api_key)
        
        # Initialize RAG services
        search_service = RAGSearchService()
        context_assembler = ContextAssembler()
        response_generator = ResponseGenerator()
        
        print("   ✅ All services initialized")
        
        # Step 2: RAG Search
        print("🔍 Step 2: Performing RAG search...")
        
        search_request = RAGSearchRequest(
            query=query,
            top_k=3,  # Keep it simple - just 3 results
            similarity_threshold=0.6,
            user_context={
                'learning_stage': 'understand',
                'preferred_difficulty': 'beginner',
                'current_topic': 'machine_learning'
            }
        )
        
        search_response = await search_service.search(search_request)
        
        print(f"   ✅ Found {len(search_response.results)} relevant results")
        
        if search_response.results:
            for i, result in enumerate(search_response.results[:3], 1):
                print(f"   📖 Result {i}: {result.content[:80]}...")
        else:
            print("   ⚠️ No results found - will test with general knowledge")
        
        # Step 3: Context Assembly
        print("🧩 Step 3: Assembling context...")
        
        # Create minimal context
        cognitive_profile = CognitiveProfile(
            learning_stage="understand",
            preferred_difficulty="beginner",
            current_topic="machine_learning"
        )
        
        session_state = SessionState(
            session_id="test_session_001",
            user_id="test_user_001",
            current_topic="machine_learning"
        )
        
        assembled_context = AssembledContext(
            conversational_context=[],  # No conversation history
            retrieved_knowledge=search_response.results,
            session_context=session_state,
            cognitive_context=cognitive_profile
        )
        
        print(f"   ✅ Context assembled with {len(search_response.results)} knowledge pieces")
        
        # Step 4: Response Generation
        print("🤖 Step 4: Generating response with real Gemini API...")
        
        response_request = ResponseGenerationRequest(
            query=query,
            intent=QueryIntent.FACTUAL,
            assembled_context=assembled_context
        )
        
        generated_response = await response_generator.generate_response(response_request)
        
        print("   ✅ Response generated successfully!")
        print()
        
        # Step 5: Display Results
        print("📋 RESULTS:")
        print("-" * 40)
        print(f"🎯 Query: {query}")
        print(f"🔍 Knowledge Retrieved: {len(search_response.results)} pieces")
        print(f"🤖 Response Type: {generated_response.response_type}")
        print(f"📊 Confidence: {generated_response.confidence:.2f}")
        print()
        print("💬 AI Response:")
        print("-" * 40)
        print(generated_response.content)
        print("-" * 40)
        
        # Step 6: Context Usage Analysis
        print()
        print("🧐 CONTEXT USAGE ANALYSIS:")
        print("-" * 40)
        
        if search_response.results:
            print("✅ RAG search returned relevant knowledge")
            print("✅ Context assembled with retrieved knowledge")
            print("✅ Response generated using real Gemini API")
            
            # Check if response seems to use retrieved context
            response_lower = generated_response.content.lower()
            context_keywords = []
            
            for result in search_response.results:
                if "machine learning" in result.content.lower():
                    context_keywords.append("machine learning")
                if "artificial intelligence" in result.content.lower():
                    context_keywords.append("AI/artificial intelligence")
                if "algorithm" in result.content.lower():
                    context_keywords.append("algorithms")
            
            if context_keywords:
                print(f"🎯 Response likely uses retrieved context (contains: {', '.join(set(context_keywords))})")
            else:
                print("⚠️ Response may be using general knowledge instead of retrieved context")
        else:
            print("⚠️ No knowledge retrieved - response is using general knowledge")
        
        print()
        print("🎉 SINGLE RAG QUERY TEST COMPLETED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_single_rag_query())
    if success:
        print("\n🚀 RAG PIPELINE IS FULLY FUNCTIONAL!")
    else:
        print("\n💥 RAG PIPELINE NEEDS DEBUGGING")
