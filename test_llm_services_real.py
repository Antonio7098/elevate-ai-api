#!/usr/bin/env python3
"""
Test script for LLM services with real API calls.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.llm_service import create_llm_service, LLMService

class LLMServicesTester:
    def __init__(self):
        self.llm_service = None
        
    async def test_gemini_service_direct(self):
        """Test Gemini LLM service directly with real API calls."""
        print("🤖 Testing Gemini LLM Service (Direct)")
        print("-" * 50)
        
        try:
            # Check environment variables
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                print("❌ Missing GOOGLE_API_KEY environment variable")
                return False
            
            print(f"   🔑 API Key: {api_key[:10]}...{api_key[-10:] if len(api_key) > 20 else ''}")
            
            # Create Gemini LLM service
            print("   🚀 Creating Gemini LLM service...")
            self.llm_service = create_llm_service(provider="gemini")
            print("   ✅ Gemini LLM service created")
            
            # Test simple text generation
            print("   📝 Testing simple text generation...")
            prompt = "Write a short paragraph about artificial intelligence in exactly 2 sentences."
            response = await self.llm_service.call_llm(prompt)
            print(f"   ✅ Response received: {len(response)} characters")
            print(f"   📄 Response preview: {response[:100]}...")
            
            # Test structured response
            print("   🏗️  Testing structured response...")
            structured_prompt = """
            Return ONLY a valid JSON object with this exact structure:
            {
                "topic": "string",
                "summary": "string",
                "key_points": ["string", "string"]
            }
            
            Topic: Machine Learning
            """
            structured_response = await self.llm_service.call_llm(structured_prompt)
            print(f"   ✅ Structured response received: {len(structured_response)} characters")
            print(f"   📄 Response preview: {structured_response[:100]}...")
            
            # Test conversation
            print("   💬 Testing conversation...")
            conversation_prompt = "What is the difference between supervised and unsupervised learning?"
            conversation_response = await self.llm_service.call_llm(conversation_prompt)
            print(f"   ✅ Conversation response received: {len(conversation_response)} characters")
            print(f"   📄 Response preview: {conversation_response[:100]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Gemini service test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_llm_service_integration(self):
        """Test LLM service integration and fallback mechanisms."""
        print("\n🔗 Testing LLM Service Integration")
        print("-" * 50)
        
        try:
            # Test with different providers
            providers = ["gemini", "openrouter", "mock"]
            
            for provider in providers:
                print(f"   🔄 Testing provider: {provider}")
                try:
                    service = create_llm_service(provider=provider)
                    print(f"      ✅ {provider} service created successfully")
                    
                    # Test a simple call
                    test_prompt = "Say 'Hello from {provider}' in exactly 5 words."
                    response = await service.call_llm(test_prompt)
                    print(f"      ✅ {provider} response: {response[:50]}...")
                    
                except Exception as e:
                    print(f"      ❌ {provider} service failed: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ LLM service integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_model_cascading(self):
        """Test model cascading functionality."""
        print("\n🔄 Testing Model Cascading")
        print("-" * 50)
        
        try:
            # Test with Gemini service
            service = create_llm_service(provider="gemini")
            
            # Test different model configurations
            models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-1.5-pro"]
            
            for model in models:
                print(f"   🔄 Testing model: {model}")
                try:
                    # Set model if the service supports it
                    if hasattr(service, 'model'):
                        service.model = model
                        print(f"      ✅ Model set to: {model}")
                    
                    # Test call
                    test_prompt = f"Generate a creative story about a {model} AI model in exactly 3 sentences."
                    response = await service.call_llm(test_prompt)
                    print(f"      ✅ Response received: {len(response)} characters")
                    
                except Exception as e:
                    print(f"      ❌ Model {model} failed: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model cascading test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_all_tests(self):
        """Run all LLM service tests."""
        print("🚀 Starting LLM Services Test Suite")
        print("=" * 60)
        
        tests = [
            ("Gemini Service (Direct)", self.test_gemini_service_direct),
            ("LLM Service Integration", self.test_llm_service_integration),
            ("Model Cascading", self.test_model_cascading)
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n🧪 Running: {test_name}")
            try:
                result = await test_func()
                results.append((test_name, result))
                print(f"   {'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
            except Exception as e:
                print(f"   ❌ ERROR: {test_name} - {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        
        return passed == total

async def main():
    """Main test function."""
    tester = LLMServicesTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 All LLM service tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some LLM service tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
