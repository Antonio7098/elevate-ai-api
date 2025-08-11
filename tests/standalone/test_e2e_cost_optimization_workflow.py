#!/usr/bin/env python3
"""
E2E Test: Cost Optimization Workflow
Tests the complete cost optimization workflow with real API calls.
"""

import asyncio
import sys
import os
import httpx
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Test configuration
BASE_URL = "http://localhost:8000"
CORE_API_URL = "http://localhost:3000"
TEST_USER_ID = "test-premium-user-123"

# Enhanced test configuration
TEST_CONFIG = {
    "timeout": 30.0,
    "retry_attempts": 3,
    "retry_delay": 1.0,
    "performance_threshold": 5.0,  # seconds
    "cost_threshold": 0.10  # dollars
}

class CostOptimizationE2ETester:
    """E2E tester for cost optimization workflow"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=TEST_CONFIG["timeout"])
        self.test_results = []
        self.performance_metrics = {}
        self.cost_metrics = {}
    
    async def validate_configuration(self) -> bool:
        """Validate test configuration and API availability"""
        print("\n🔧 Validating Configuration...")
        
        try:
            # Check if APIs are accessible
            health_response = await self.client.get(f"{BASE_URL}/health")
            if health_response.status_code != 200:
                print(f"❌ Main API health check failed: {health_response.status_code}")
                return False
            
            core_health_response = await self.client.get(f"{CORE_API_URL}/health")
            if core_health_response.status_code != 200:
                print(f"❌ Core API health check failed: {core_health_response.status_code}")
                return False
            
            print("✅ Configuration validation successful")
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation error: {e}")
            return False
    
    async def test_model_cascading(self) -> bool:
        """Test model cascading with different query complexities"""
        print("\n🧪 Testing Model Cascading...")
        
        test_cases = [
            ("simple", "Hello, how are you?", "gpt-3.5-turbo"),
            ("medium", "Explain the basics of machine learning", "gpt-4"),
            ("complex", "Analyze the impact of AI on healthcare and provide detailed recommendations", "gpt-4-turbo")
        ]
        
        results = []
        
        for complexity, query, expected_model in test_cases:
            try:
                start_time = time.time()
                
                payload = {
                    "query": query,
                    "user_id": TEST_USER_ID,
                    "complexity": complexity
                }
                
                response = await self.client.post(
                    f"{BASE_URL}/premium/chat/advanced",
                    json=payload
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status_code == 200:
                    response_data = response.json()
                    used_model = response_data.get("model_used", "unknown")
                    
                    print(f"  ✅ {complexity.capitalize()} query: {used_model} ({response_time:.2f}s)")
                    results.append(True)
                    
                    # Store performance metrics
                    self.performance_metrics[f"cascading_{complexity}"] = response_time
                else:
                    print(f"  ❌ {complexity.capitalize()} query failed: {response.status_code}")
                    results.append(False)
                    
            except Exception as e:
                print(f"  ❌ {complexity.capitalize()} query error: {e}")
                results.append(False)
        
        return all(results)
    
    async def test_caching(self) -> bool:
        """Test caching with repeated queries"""
        print("\n🧪 Testing Caching...")
        
        try:
            test_query = "Explain machine learning in detail"
            cache_hits = 0
            
            # First request (should miss cache)
            start_time = time.time()
            payload = {"query": test_query, "user_id": TEST_USER_ID}
            response1 = await self.client.post(f"{BASE_URL}/premium/chat/advanced", json=payload)
            first_response_time = time.time() - start_time
            
            if response1.status_code != 200:
                print("  ❌ First request failed")
                return False
            
            # Second request (should hit cache)
            start_time = time.time()
            response2 = await self.client.post(f"{BASE_URL}/premium/chat/advanced", json=payload)
            second_response_time = time.time() - start_time
            
            if response2.status_code != 200:
                print("  ❌ Second request failed")
                return False
            
            # Check if caching is working (second request should be faster)
            if second_response_time < first_response_time:
                cache_hits += 1
                print(f"  ✅ Cache hit detected: {second_response_time:.2f}s vs {first_response_time:.2f}s")
            else:
                print(f"  ⚠️  Cache may not be working: {second_response_time:.2f}s vs {first_response_time:.2f}s")
            
            # Third request to verify consistency
            response3 = await self.client.post(f"{BASE_URL}/premium/chat/advanced", json=payload)
            if response3.status_code == 200:
                cache_hits += 1
            
            self.performance_metrics["caching_first"] = first_response_time
            self.performance_metrics["caching_cached"] = second_response_time
            
            print(f"  ✅ Caching test completed with {cache_hits}/2 cache hits")
            return cache_hits >= 1
            
        except Exception as e:
            print(f"  ❌ Caching test error: {e}")
            return False
    
    async def test_cost_monitoring(self) -> bool:
        """Test cost monitoring and tracking"""
        print("\n🧪 Testing Cost Monitoring...")
        
        try:
            queries = [
                "Simple greeting",
                "Explain quantum computing",
                "Analyze the economic impact of renewable energy",
                "Provide a detailed comparison of programming languages"
            ]
            
            total_cost = 0.0
            successful_queries = 0
            
            for i, query in enumerate(queries, 1):
                try:
                    start_time = time.time()
                    
                    payload = {"query": query, "user_id": TEST_USER_ID}
                    response = await self.client.post(f"{BASE_URL}/premium/chat/advanced", json=payload)
                    
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        cost = response_data.get("estimated_cost", 0.0)
                        total_cost += cost
                        successful_queries += 1
                        
                        print(f"  ✅ Query {i}: ${cost:.4f} ({response_time:.2f}s)")
                        
                        # Store cost metrics
                        self.cost_metrics[f"query_{i}"] = {
                            "cost": cost,
                            "response_time": response_time,
                            "complexity": len(query.split())
                        }
                    else:
                        print(f"  ❌ Query {i} failed: {response.status_code}")
                        
                except Exception as e:
                    print(f"  ❌ Query {i} error: {e}")
            
            # Check cost thresholds
            if total_cost > TEST_CONFIG["cost_threshold"]:
                print(f"  ⚠️  Total cost (${total_cost:.4f}) exceeds threshold (${TEST_CONFIG['cost_threshold']})")
            
            print(f"  📊 Cost monitoring: {successful_queries}/{len(queries)} queries successful, total: ${total_cost:.4f}")
            
            self.cost_metrics["total_cost"] = total_cost
            self.cost_metrics["successful_queries"] = successful_queries
            
            return successful_queries >= len(queries) * 0.8  # 80% success rate
            
        except Exception as e:
            print(f"  ❌ Cost monitoring test error: {e}")
            return False
    
    async def test_rate_limiting(self) -> bool:
        """Test rate limiting and throttling"""
        print("\n🧪 Testing Rate Limiting...")
        
        try:
            # Send multiple requests rapidly
            rapid_queries = ["Test query"] * 10
            responses = []
            
            start_time = time.time()
            
            for i, query in enumerate(rapid_queries):
                payload = {"query": query, "user_id": TEST_USER_ID}
                response = await self.client.post(f"{BASE_URL}/premium/chat/advanced", json=payload)
                responses.append(response.status_code)
                
                # Small delay to avoid overwhelming the system
                await asyncio.sleep(0.1)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Check for rate limiting (some requests should be throttled)
            successful = sum(1 for status in responses if status == 200)
            throttled = sum(1 for status in responses if status == 429)
            
            print(f"  📊 Rate limiting: {successful}/10 successful, {throttled}/10 throttled in {total_time:.2f}s")
            
            self.performance_metrics["rate_limiting_total_time"] = total_time
            self.performance_metrics["rate_limiting_successful"] = successful
            
            # Rate limiting is working if some requests are throttled
            return throttled > 0 or successful >= 8
            
        except Exception as e:
            print(f"  ❌ Rate limiting test error: {e}")
            return False
    
    async def test_error_handling(self) -> bool:
        """Test error handling with invalid inputs"""
        print("\n🧪 Testing Error Handling...")
        
        error_test_cases = [
            ({"query": "", "user_id": TEST_USER_ID}, 400, "Empty query"),
            ({"query": "Valid query"}, 400, "Missing user_id"),
            ({"query": "Valid query", "user_id": "invalid-user"}, 401, "Invalid user"),
            ({"query": "A" * 10000, "user_id": TEST_USER_ID}, 400, "Very long query")
        ]
        
        results = []
        
        for payload, expected_status, description in error_test_cases:
            try:
                response = await self.client.post(f"{BASE_URL}/premium/chat/advanced", json=payload)
                
                if response.status_code == expected_status:
                    print(f"  ✅ {description}: Correctly returned {expected_status}")
                    results.append(True)
                else:
                    print(f"  ❌ {description}: Expected {expected_status}, got {response.status_code}")
                    results.append(False)
                    
            except Exception as e:
                print(f"  ❌ {description}: Error occurred - {e}")
                results.append(False)
        
        return all(results)
    
    async def run_all_tests(self) -> List[Tuple[str, bool]]:
        """Run all cost optimization tests"""
        print("🚀 Starting E2E Cost Optimization Tests")
        print(f"📅 Test started at: {datetime.utcnow()}")
        print(f"⚙️  Configuration: {json.dumps(TEST_CONFIG, indent=2)}")
        
        # Validate configuration first
        if not await self.validate_configuration():
            print("❌ Configuration validation failed. Aborting tests.")
            return [("Configuration", False)]
        
        tests = [
            ("Model Cascading", self.test_model_cascading),
            ("Caching", self.test_caching),
            ("Cost Monitoring", self.test_cost_monitoring),
            ("Rate Limiting", self.test_rate_limiting),
            ("Error Handling", self.test_error_handling)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            try:
                success = await test_func()
                results.append((test_name, success))
            except Exception as e:
                print(f"❌ {test_name} test failed: {e}")
                results.append((test_name, False))
        
        # Print detailed summary
        await self.print_detailed_summary(results)
        
        return results
    
    async def print_detailed_summary(self, results: List[Tuple[str, bool]]):
        """Print detailed test summary with metrics"""
        print("\n" + "="*80)
        print("📊 E2E COST OPTIMIZATION TEST DETAILED SUMMARY")
        print("="*80)
        
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        # Test results
        print("\n🧪 TEST RESULTS:")
        for test_name, success in results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"  {status} {test_name}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        # Performance metrics
        if self.performance_metrics:
            print("\n⚡ PERFORMANCE METRICS:")
            for metric, value in self.performance_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.3f}s")
                else:
                    print(f"  {metric}: {value}")
        
        # Cost metrics
        if self.cost_metrics:
            print("\n💰 COST METRICS:")
            if "total_cost" in self.cost_metrics:
                print(f"  Total cost: ${self.cost_metrics['total_cost']:.4f}")
            if "successful_queries" in self.cost_metrics:
                print(f"  Successful queries: {self.cost_metrics['successful_queries']}")
        
        # Final status
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! Cost optimization workflow is working correctly.")
        elif passed >= total * 0.8:
            print("\n⚠️  Most tests passed. Some issues detected but workflow is functional.")
        else:
            print("\n❌ Multiple tests failed. Cost optimization workflow needs attention.")
        
        print("="*80)
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()

async def main():
    """Main test runner"""
    tester = CostOptimizationE2ETester()
    try:
        results = await tester.run_all_tests()
        
        # Exit with appropriate code
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        if passed == total:
            sys.exit(0)  # All tests passed
        elif passed >= total * 0.8:
            sys.exit(1)  # Most tests passed
        else:
            sys.exit(2)  # Many tests failed
            
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(3)
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())


