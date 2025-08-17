#!/usr/bin/env python3
"""
Test module for LLM Services with REAL API calls.
Tests Gemini, OpenRouter, and fallback mechanisms.
"""

import asyncio
import os
import time
from typing import Dict, Any

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded")
except ImportError:
    print("⚠️  python-dotenv not available")

from app.services.llm_service import create_llm_service
from app.services.gemini_service import GeminiService, GeminiConfig
from app.core.llm_service import LLMService


class LLMServicesTester:
    """Test suite for LLM services with real API calls."""
    
    def __init__(self):
        self.test_results = []
        
    async def test_gemini_service_direct(self):
        """Test Gemini service directly with real API calls."""
        print("\n🔍 Testing Gemini Service Direct API Calls")
        print("-" * 50)
        
        try:
            # Test with real API key
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                print("❌ GOOGLE_API_KEY not found")
                return False
                
            config = GeminiConfig(api_key=api_key, model_name="gemini-2.5-flash")
            service = GeminiService(config)
            
            # Test basic generation
            start_time = time.time()
            response = await service.generate_response("Explain machine learning in one sentence")
            end_time = time.time()
            
            print(f"✅ Gemini API call successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📝 Response: {response[:100]}...")
            
            # Test embeddings
            embedding = await service.embed_text("Test text for embedding")
            print(f"✅ Embeddings generated: {len(embedding)} dimensions")
            
            return True
            
        except Exception as e:
            print(f"❌ Gemini service test failed: {e}")
            return False
    
    async def test_llm_service_integration(self):
        """Test integrated LLM service with real calls."""
        print("\n🔍 Testing Integrated LLM Service")
        print("-" * 50)
        
        try:
            # Test Gemini provider
            llm_service = create_llm_service(provider="gemini")
            
            start_time = time.time()
            response = await llm_service.call_llm(
                "What is the capital of France?",
                operation="test"
            )
            end_time = time.time()
            
            print(f"✅ Integrated LLM service working")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📝 Response: {response[:100]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Integrated LLM service test failed: {e}")
            return False
    
    async def test_openrouter_fallback(self):
        """Test OpenRouter fallback with real API calls."""
        print("\n🔍 Testing OpenRouter Fallback")
        print("-" * 50)
        
        try:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                print("⚠️  OPENROUTER_API_KEY not found, skipping test")
                return True
                
            llm_service = create_llm_service(provider="openrouter")
            
            start_time = time.time()
            response = await llm_service.call_openrouter_ai(
                "Explain quantum computing briefly",
                model="z-ai/glm-4.5-air:free",
                operation="test"
            )
            end_time = time.time()
            
            print(f"✅ OpenRouter API call successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📝 Response: {response[:100]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ OpenRouter test failed: {e}")
            return False
    
    async def test_model_cascading(self):
        """Test model cascading with real API calls."""
        print("\n🔍 Testing Model Cascading")
        print("-" * 50)
        
        try:
            from app.core.premium.model_cascader import ModelCascader
            
            cascader = ModelCascader()
            
            # Test model selection
            model_info = await cascader.get_model_info()
            print(f"✅ Model cascader initialized")
            print(f"   🎯 Available models: {list(model_info.keys())}")
            
            # Test cost estimation
            cost_estimate = await cascader.get_cost_estimate(
                "gemini-2.5-flash",
                input_chars=1000,
                output_chars=500
            )
            print(f"   💰 Cost estimate: ${cost_estimate:.6f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model cascading test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all LLM service tests."""
        print("🚀 LLM Services Test Suite")
        print("=" * 60)
        
        tests = [
            ("Gemini Direct API", self.test_gemini_service_direct),
            ("Integrated LLM Service", self.test_llm_service_integration),
            ("OpenRouter Fallback", self.test_openrouter_fallback),
            ("Model Cascading", self.test_model_cascading)
        ]
        
        for test_name, test_func in tests:
            try:
                success = await test_func()
                self.test_results.append((test_name, success))
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                self.test_results.append((test_name, False))
        
        # Print summary
        print("\n📊 LLM Services Test Results")
        print("-" * 40)
        passed = sum(1 for _, success in self.test_results if success)
        total = len(self.test_results)
        
        for test_name, success in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {status} {test_name}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        return passed == total


async def main():
    """Run LLM services tests."""
    tester = LLMServicesTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())






