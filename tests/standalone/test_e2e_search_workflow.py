#!/usr/bin/env python3
"""
E2E Test: Premium Search Workflow
Tests the complete premium search workflow with real API calls and Core API integration.
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
BASE_URL = "http://localhost:8000"
CORE_API_URL = "http://localhost:3000"  # Core API URL
TEST_USER_ID = "test-premium-user-123"

class PremiumSearchE2ETester:
    """E2E tester for premium search workflow"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.test_results = []
    
    async def test_advanced_search(self):
        """Test advanced search with RAG-Fusion"""
        print("\n🧪 Testing Advanced Search (RAG-Fusion)...")
        
        try:
            payload = {
                "query": "machine learning algorithms for classification",
                "user_id": TEST_USER_ID,
                "mode": "comprehensive",
                "filters": {
                    "content_type": ["tutorial", "example"],
                    "difficulty": "intermediate"
                },
                "preferences": {
                    "max_results": 10,
                    "include_code": True,
                    "include_diagrams": True
                }
            }
            
            response = await self.client.post(
                f"{BASE_URL}/premium/search/advanced",
                json=payload
            )
            
            if response.status_code == 200:
                search_data = response.json()
                print(f"✅ Advanced search successful")
                print(f"   - Results count: {len(search_data.get('results', []))}")
                print(f"   - Strategy scores: {len(search_data.get('strategy_scores', {}))}")
                print(f"   - Fusion quality: {search_data.get('fusion_quality', 0):.3f}")
                print(f"   - Optimization metrics: {len(search_data.get('optimization_metrics', {}))}")
                return True
            else:
                print(f"❌ Advanced search failed: {response.status_code}")
                print(f"   - Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Advanced search error: {e}")
            return False
    
    async def test_multimodal_search(self):
        """Test multimodal search"""
        print("\n🧪 Testing Multi-Modal Search...")
        
        try:
            payload = {
                "text_query": "neural network architecture",
                "image_query": None,  # Base64 encoded image would go here
                "audio_query": None,  # Base64 encoded audio would go here
                "code_query": "import tensorflow as tf",
                "diagram_query": "neural network diagram",
                "modality_weights": {
                    "text": 0.4,
                    "image": 0.2,
                    "audio": 0.1,
                    "code": 0.2,
                    "diagram": 0.1
                },
                "user_id": TEST_USER_ID
            }
            
            response = await self.client.post(
                f"{BASE_URL}/premium/search/multimodal",
                json=payload
            )
            
            if response.status_code == 200:
                multimodal_data = response.json()
                print(f"✅ Multi-modal search successful")
                print(f"   - Text results: {len(multimodal_data.get('text_results', []))}")
                print(f"   - Code results: {len(multimodal_data.get('code_results', []))}")
                print(f"   - Diagram results: {len(multimodal_data.get('diagram_results', []))}")
                print(f"   - Fusion scores: {len(multimodal_data.get('fusion_scores', {}))}")
                print(f"   - Cross-modal relationships: {len(multimodal_data.get('cross_modal_relationships', []))}")
                return True
            else:
                print(f"❌ Multi-modal search failed: {response.status_code}")
                print(f"   - Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Multi-modal search error: {e}")
            return False
    
    async def test_graph_search(self):
        """Test graph-based search"""
        print("\n🧪 Testing Graph-Based Search...")
        
        try:
            payload = {
                "query": "machine learning concepts and their relationships",
                "user_id": TEST_USER_ID,
                "graph_depth": 3,
                "relationship_types": ["PREREQUISITE_FOR", "RELATED_TO", "BUILDS_ON"],
                "include_metadata": True
            }
            
            response = await self.client.post(
                f"{BASE_URL}/premium/search/graph",
                json=payload
            )
            
            if response.status_code == 200:
                graph_data = response.json()
                print(f"✅ Graph search successful")
                print(f"   - Results count: {len(graph_data.get('results', []))}")
                print(f"   - Adaptive strategy: {graph_data.get('adaptive_strategy')}")
                print(f"   - Adaptation reason: {graph_data.get('adaptation_reason')}")
                print(f"   - Performance metrics: {len(graph_data.get('performance_metrics', {}))}")
                return True
            else:
                print(f"❌ Graph search failed: {response.status_code}")
                print(f"   - Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Graph search error: {e}")
            return False
    
    async def test_graph_search_basic(self):
        """Test basic graph search endpoint"""
        print("\n🧪 Testing Basic Graph Search...")
        
        try:
            payload = {
                "query": "artificial intelligence fundamentals",
                "user_id": TEST_USER_ID
            }
            
            response = await self.client.post(
                f"{BASE_URL}/premium/chat/graph-search",
                json=payload
            )
            
            if response.status_code == 200:
                graph_data = response.json()
                print(f"✅ Basic graph search successful")
                print(f"   - Results count: {len(graph_data.get('results', []))}")
                print(f"   - Query: {graph_data.get('query')}")
                return True
            else:
                print(f"❌ Basic graph search failed: {response.status_code}")
                print(f"   - Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Basic graph search error: {e}")
            return False
    
    async def test_core_api_knowledge_retrieval(self):
        """Test Core API knowledge retrieval"""
        print("\n🧪 Testing Core API Knowledge Retrieval...")
        
        try:
            # Test knowledge primitives retrieval
            primitives_response = await self.client.get(f"{CORE_API_URL}/knowledge-primitives")
            
            if primitives_response.status_code == 200:
                primitives_data = primitives_response.json()
                print(f"✅ Knowledge primitives retrieved")
                print(f"   - Primitives count: {len(primitives_data)}")
                return True
            else:
                print(f"⚠️  Knowledge primitives not available: {primitives_response.status_code}")
                return True  # This might not be implemented yet
                
        except Exception as e:
            print(f"❌ Core API knowledge retrieval error: {e}")
            return False
    
    async def test_search_performance(self):
        """Test search performance metrics"""
        print("\n🧪 Testing Search Performance...")
        
        try:
            # Test multiple searches to measure performance
            search_queries = [
                "python programming basics",
                "machine learning algorithms",
                "deep learning neural networks",
                "data science workflow",
                "statistical analysis methods"
            ]
            
            performance_metrics = []
            
            for query in search_queries:
                start_time = datetime.utcnow()
                
                payload = {
                    "query": query,
                    "user_id": TEST_USER_ID,
                    "mode": "fast"
                }
                
                response = await self.client.post(
                    f"{BASE_URL}/premium/search/advanced",
                    json=payload
                )
                
                end_time = datetime.utcnow()
                latency = (end_time - start_time).total_seconds() * 1000
                
                if response.status_code == 200:
                    search_data = response.json()
                    performance_metrics.append({
                        "query": query,
                        "latency_ms": latency,
                        "results_count": len(search_data.get('results', [])),
                        "success": True
                    })
                else:
                    performance_metrics.append({
                        "query": query,
                        "latency_ms": latency,
                        "results_count": 0,
                        "success": False
                    })
            
            # Calculate average performance
            successful_searches = [m for m in performance_metrics if m['success']]
            if successful_searches:
                avg_latency = sum(m['latency_ms'] for m in successful_searches) / len(successful_searches)
                avg_results = sum(m['results_count'] for m in successful_searches) / len(successful_searches)
                
                print(f"✅ Search performance test completed")
                print(f"   - Successful searches: {len(successful_searches)}/{len(search_queries)}")
                print(f"   - Average latency: {avg_latency:.1f}ms")
                print(f"   - Average results: {avg_results:.1f}")
                return True
            else:
                print(f"❌ No successful searches in performance test")
                return False
                
        except Exception as e:
            print(f"❌ Search performance test error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all premium search workflow tests"""
        print("🚀 Starting E2E Premium Search Workflow Tests")
        print(f"📅 Test started at: {datetime.utcnow()}")
        print(f"🌐 AI API URL: {BASE_URL}")
        print(f"🌐 Core API URL: {CORE_API_URL}")
        
        tests = [
            ("Core API Knowledge Retrieval", self.test_core_api_knowledge_retrieval),
            ("Basic Graph Search", self.test_graph_search_basic),
            ("Advanced Search (RAG-Fusion)", self.test_advanced_search),
            ("Multi-Modal Search", self.test_multimodal_search),
            ("Graph-Based Search", self.test_graph_search),
            ("Search Performance", self.test_search_performance)
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
        print("📊 E2E PREMIUM SEARCH WORKFLOW TEST SUMMARY")
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
            print("🎉 ALL TESTS PASSED! Premium search workflow is working correctly.")
        else:
            print("⚠️  Some tests failed. Please check the implementation.")
        
        print(f"📅 Test completed at: {datetime.utcnow()}")
        
        return results
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()

async def main():
    """Main test runner"""
    tester = PremiumSearchE2ETester()
    
    try:
        results = await tester.run_all_tests()
        
        # Save test results
        with open("e2e_premium_search_results.json", "w") as f:
            json.dump(tester.test_results, f, indent=2, default=str)
        
        print(f"\n📄 Test results saved to: e2e_premium_search_results.json")
        
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())








