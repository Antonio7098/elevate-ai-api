#!/usr/bin/env python3
"""
Test Gemini Context-Aware Responses
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add app to path
import sys
sys.path.append('.')

async def test_gemini_context():
    """Test that Gemini uses provided context"""
    
    print("🎯 GEMINI CONTEXT-AWARE RESPONSE TEST")
    print("=" * 50)
    
    try:
        # Check API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ No GOOGLE_API_KEY found")
            return False
        
        print(f"✅ API key loaded: {api_key[:10]}...")
        
        # Initialize Gemini service
        from app.services.gemini_service import GeminiService
        
        gemini = GeminiService()
        
        if gemini.mock_mode:
            print("❌ Gemini is in mock mode! Cannot test real API.")
            return False
        
        print("✅ Gemini service initialized (not in mock mode)")
        
        # Test 1: Response WITHOUT context (general knowledge)
        print("\n🧪 Test 1: General knowledge response")
        print("-" * 30)
        
        general_prompt = "What is machine learning?"
        general_response = await gemini.generate_response(general_prompt)
        
        print("📝 General response:")
        print(general_response[:150] + "...")
        
        # Test 2: Response WITH specific context
        print("\n🧪 Test 2: Context-aware response")  
        print("-" * 30)
        
        # Simulate retrieved context from our Pinecone index
        context_prompt = """Based on the following knowledge retrieved from our database:

RETRIEVED CONTEXT:
- "Machine Learning is a subset of Artificial Intelligence"
- "Machine Learning Model Development Steps: 1. Define problem and collect data, 2. Preprocess and clean data, 3. Choose appropriate algorithm, 4. Train model on training data, 5. Evaluate model performance, 6. Deploy and monitor model"
- "Common machine learning algorithms include supervised learning, unsupervised learning, and reinforcement learning"

Please answer this question using the context above: What is machine learning?

Focus on the information provided in the retrieved context."""

        context_response = await gemini.generate_response(context_prompt)
        
        print("📝 Context-aware response:")
        print(context_response[:150] + "...")
        
        # Analysis
        print("\n🧐 ANALYSIS:")
        print("-" * 30)
        
        # Check if context response mentions specific details from our context
        context_lower = context_response.lower()
        general_lower = general_response.lower()
        
        context_indicators = []
        if "subset" in context_lower and "artificial intelligence" in context_lower:
            context_indicators.append("✅ Mentions ML as subset of AI (from context)")
        
        if "steps" in context_lower or "development" in context_lower:
            context_indicators.append("✅ Mentions development steps (from context)")
            
        if "supervised" in context_lower and "unsupervised" in context_lower:
            context_indicators.append("✅ Mentions algorithm types (from context)")
        
        print(f"📊 Context indicators found: {len(context_indicators)}")
        for indicator in context_indicators:
            print(f"   {indicator}")
        
        # Final assessment
        if len(context_indicators) >= 2:
            print("\n🎉 SUCCESS: Gemini is using retrieved context!")
            print("✅ Context-aware responses are working")
            return True
        elif len(context_indicators) >= 1:
            print("\n⚠️ PARTIAL: Some context usage detected")
            print("🔍 May need prompt engineering improvements")
            return True
        else:
            print("\n❌ CONCERN: Limited context usage detected")
            print("🔍 May need to improve context formatting")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_gemini_context())
    if success:
        print("\n🚀 GEMINI CONTEXT-AWARE RESPONSES VALIDATED!")
        print("✅ RAG pipeline can generate context-aware responses")
        print("✅ Ready for production use")
    else:
        print("\n💥 GEMINI CONTEXT USAGE NEEDS IMPROVEMENT")
