#!/usr/bin/env python3
"""
Simple RAG Test - Just test RAG search and response generation
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add app to path
import sys
sys.path.append('.')

async def test_simple_rag():
    """Simple test of RAG pipeline"""
    
    print("🎯 SIMPLE RAG PIPELINE TEST")
    print("=" * 40)
    
    try:
        # Check environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ No GOOGLE_API_KEY found")
            return False
        
        print(f"✅ API key found: {api_key[:10]}...")
        
        # Initialize embedding service
        from app.core.embeddings import initialize_embedding_service
        initialize_embedding_service("google", api_key=api_key)
        print("✅ Embedding service initialized")
        
        # Test RAG search
        from app.core.rag_search import RAGSearchService, RAGSearchRequest
        
        search_service = RAGSearchService()
        search_request = RAGSearchRequest(
            query="What is machine learning?",
            top_k=3,
            similarity_threshold=0.6
        )
        
        print("🔍 Testing RAG search...")
        search_response = await search_service.search(search_request)
        
        print(f"✅ RAG search completed - found {len(search_response.results)} results")
        
        if search_response.results:
            for i, result in enumerate(search_response.results, 1):
                print(f"   📖 Result {i}: {result.content[:60]}...")
        
        # Test Gemini directly
        print("🤖 Testing Gemini API directly...")
        from app.services.gemini_service import GeminiService
        
        gemini = GeminiService()
        if gemini.mock_mode:
            print("❌ Gemini is in mock mode!")
            return False
        
        # Create a simple prompt with retrieved context
        if search_response.results:
            context_text = "\n".join([f"- {r.content[:100]}" for r in search_response.results[:2]])
            prompt = f"""Based on the following retrieved knowledge:

{context_text}

Please answer this question: What is machine learning?

Provide a concise answer based on the context above."""
        else:
            prompt = "What is machine learning? Provide a concise explanation."
        
        print("📝 Generating response...")
        response = await gemini.generate_response(prompt)
        
        print("✅ Response generated!")
        print()
        print("🎯 RESULTS:")
        print("-" * 40)
        print(f"Query: What is machine learning?")
        print(f"Retrieved: {len(search_response.results)} knowledge pieces")
        print()
        print("🤖 AI Response:")
        print(response)
        print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_simple_rag())
    if success:
        print("\n🚀 RAG PIPELINE VALIDATION SUCCESSFUL!")
        print("✅ Environment loading working")
        print("✅ Embedding service working") 
        print("✅ RAG search working")
        print("✅ Gemini API working")
        print("✅ Context-aware responses possible")
    else:
        print("\n💥 RAG PIPELINE NEEDS FIXES")
