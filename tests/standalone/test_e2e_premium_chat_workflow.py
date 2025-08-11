#!/usr/bin/env python3
"""
E2E Test: Premium Chat Workflow
Tests the complete premium chat workflow with real API calls and Core API integration.
"""

import asyncio
import sys
import os
import httpx
import json
from datetime import datetime
from typing import Dict, Any

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Test configuration
BASE_URL = "http://localhost:8000/api/v1"
AUTH_HEADER = {"Authorization": "Bearer test-token"}
CORE_API_URL = "http://localhost:3000"  # Core API URL
TEST_USER_ID = "test-premium-user-123"

class PremiumChatE2ETester:
    """E2E tester for premium chat workflow"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.test_results = []
    
    async def test_premium_health_check(self):
        """Test premium API health check"""
        print("\n🧪 Testing Premium API Health Check...")
        
        try:
            response = await self.client.get(f"{BASE_URL}/premium/health", headers=AUTH_HEADER)
            
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Health check passed")
                print(f"   - Status: {health_data.get('status')}")
                print(f"   - Services: {len(health_data.get('services', {}))} active")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    async def test_premium_chat_advanced(self):
        """Test premium advanced chat endpoint"""
        print("\n🧪 Testing Premium Advanced Chat...")
        
        try:
            payload = {
                "query": "Explain the fundamentals of machine learning with practical examples",
                "user_id": TEST_USER_ID,
                "user_context": {
                    "learning_level": "intermediate",
                    "preferred_style": "practical",
                    "previous_topics": ["python", "statistics"]
                }
            }
            
            response = await self.client.post(
                f"{BASE_URL}/premium/chat/advanced",
                json=payload,
                headers=AUTH_HEADER,
            )
            
            if response.status_code == 200:
                chat_data = response.json()
                print(f"✅ Advanced chat successful")
                print(f"   - Response length: {len(chat_data.get('response', ''))} chars")
                print(f"   - Experts used: {len(chat_data.get('experts_used', []))}")
                print(f"   - Confidence: {chat_data.get('confidence_score', 0):.3f}")
                return True
            else:
                print(f"❌ Advanced chat failed: {response.status_code}")
                print(f"   - Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Advanced chat error: {e}")
            return False
    
    async def test_langgraph_chat(self):
        """Test LangGraph chat endpoint"""
        print("\n🧪 Testing LangGraph Chat...")
        
        try:
            payload = {
                "query": "Create a learning plan for understanding neural networks",
                "user_id": TEST_USER_ID,
                "user_context": {
                    "learning_goal": "master neural networks",
                    "time_available": "2 hours per day",
                    "background": "basic programming"
                }
            }
            
            response = await self.client.post(
                f"{BASE_URL}/premium/chat/langgraph",
                json=payload,
                headers=AUTH_HEADER,
            )
            
            if response.status_code == 200:
                langgraph_data = response.json()
                print(f"✅ LangGraph chat successful")
                print(f"   - Response length: {len(langgraph_data.get('response', ''))} chars")
                print(f"   - Agents used: {len(langgraph_data.get('agents_used', []))}")
                print(f"   - Workflow status: {langgraph_data.get('workflow_status')}")
                return True
            else:
                print(f"❌ LangGraph chat failed: {response.status_code}")
                print(f"   - Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ LangGraph chat error: {e}")
            return False
    
    async def test_context_assembly(self):
        """Test Context Assembly Agent"""
        print("\n🧪 Testing Context Assembly Agent...")
        
        try:
            payload = {
                "query": "Explain deep learning concepts with examples",
                "user_id": TEST_USER_ID,
                "mode": "deep_dive",
                "session_context": {
                    "previous_queries": ["What is AI?", "Machine learning basics"],
                    "user_preferences": {"detail_level": "comprehensive"}
                },
                "hints": ["focus on practical applications"],
                "token_budget": 4000,
                "latency_budget_ms": 5000
            }
            
            response = await self.client.post(
                f"{BASE_URL}/premium/context/assemble",
                json=payload,
                headers=AUTH_HEADER,
            )
            
            if response.status_code == 200:
                caa_data = response.json()
                print(f"✅ Context assembly successful")
                print(f"   - Assembled context: {len(caa_data.get('assembled_context', ''))} chars")
                print(f"   - Sufficiency score: {caa_data.get('sufficiency_score', 0):.3f}")
                print(f"   - Token count: {caa_data.get('token_count', 0)}")
                print(f"   - Knowledge primitives: {len(caa_data.get('knowledge_primitives', []))}")
                return True
            else:
                print(f"❌ Context assembly failed: {response.status_code}")
                print(f"   - Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Context assembly error: {e}")
            return False
    
    async def test_learning_workflow(self):
        """Test learning workflow endpoint"""
        print("\n🧪 Testing Learning Workflow...")
        
        try:
            payload = {
                "learning_goal": "Master Python for data science",
                "user_id": TEST_USER_ID,
                "user_context": {
                    "current_skills": ["basic programming"],
                    "time_commitment": "1 hour daily",
                    "preferred_style": "hands-on"
                }
            }
            
            response = await self.client.post(
                f"{BASE_URL}/premium/learning/workflow",
                json=payload,
                headers=AUTH_HEADER,
            )
            
            if response.status_code == 200:
                workflow_data = response.json()
                print(f"✅ Learning workflow successful")
                print(f"   - Learning plan: {len(workflow_data.get('learning_plan', ''))} chars")
                print(f"   - Workflow status: {workflow_data.get('workflow_status')}")
                print(f"   - Progress evaluation: {len(workflow_data.get('progress_evaluation', ''))} chars")
                return True
            else:
                print(f"❌ Learning workflow failed: {response.status_code}")
                print(f"   - Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Learning workflow error: {e}")
            return False
    
    async def test_core_api_integration(self):
        """Test Core API integration"""
        print("\n🧪 Testing Core API Integration...")
        
        try:
            # Test Core API health
            core_response = await self.client.get(f"{CORE_API_URL}/health")
            
            if core_response.status_code == 200:
                print(f"✅ Core API health check passed")
                
                # Test user data retrieval
                user_response = await self.client.get(f"{CORE_API_URL}/users/{TEST_USER_ID}")
                
                if user_response.status_code == 200:
                    user_data = user_response.json()
                    print(f"✅ User data retrieved")
                    print(f"   - User ID: {user_data.get('id')}")
                    print(f"   - Premium status: {user_data.get('premium', False)}")
                    return True
                else:
                    print(f"⚠️  User data not found (expected for test user)")
                    return True  # This is expected for test users
            else:
                print(f"❌ Core API health check failed: {core_response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Core API integration error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all premium chat workflow tests"""
        print("🚀 Starting E2E Premium Chat Workflow Tests")
        print(f"📅 Test started at: {datetime.utcnow()}")
        print(f"🌐 AI API URL: {BASE_URL}")
        print(f"🌐 Core API URL: {CORE_API_URL}")
        
        tests = [
            ("Premium API Health Check", self.test_premium_health_check),
            ("Core API Integration", self.test_core_api_integration),
            ("Premium Advanced Chat", self.test_premium_chat_advanced),
            ("LangGraph Chat", self.test_langgraph_chat),
            ("Context Assembly Agent", self.test_context_assembly),
            ("Learning Workflow", self.test_learning_workflow)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            try:
                success = await test_func()
                results.append((test_name, success))
                self.test_results.append({
                    "test": test_name,
                    "success": success,
                    "timestamp": datetime.utcnow()
                })
            except Exception as e:
                print(f"❌ {test_name} test failed with exception: {e}")
                results.append((test_name, False))
        
        # Print summary
        print("\n" + "="*60)
        print("📊 E2E PREMIUM CHAT WORKFLOW TEST SUMMARY")
        print("="*60)
        
        passed = 0
        total = len(results)
        
        for test_name, success in results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status} {test_name}")
            if success:
                passed += 1
        
        print(f"\n🎯 Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Premium chat workflow is working correctly.")
        else:
            print("⚠️  Some tests failed. Please check the implementation.")
        
        print(f"📅 Test completed at: {datetime.utcnow()}")
        
        return results
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()

async def main():
    """Main test runner"""
    tester = PremiumChatE2ETester()
    
    try:
        results = await tester.run_all_tests()
        
        # Save test results
        with open("e2e_premium_chat_results.json", "w") as f:
            json.dump(tester.test_results, f, indent=2, default=str)
        
        print(f"\n📄 Test results saved to: e2e_premium_chat_results.json")
        
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

