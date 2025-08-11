#!/usr/bin/env python3
"""
E2E Test: Real LLM Performance Testing
Tests the complete AI API workflow with REAL LLM service calls to measure actual performance.
"""

import asyncio
import sys
import os
import httpx
import json
import time
import statistics
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Test configuration
BASE_URL = "http://localhost:8000"
CORE_API_URL = "http://localhost:3000"
TEST_USER_ID = "test-premium-user-123"
API_KEY = "test-token"  # For testing purposes

# Enhanced test configuration for real LLM testing
REAL_LLM_CONFIG = {
    "timeout": 60.0,  # Longer timeout for real LLM calls
    "retry_attempts": 3,
    "retry_delay": 2.0,
    "performance_threshold": 30.0,  # seconds for real LLM calls
    "cost_threshold": 1.0,  # dollars for real testing
    "concurrent_requests": 5,  # Test concurrent performance
    "test_iterations": 3,  # Multiple iterations for reliable metrics
}

@dataclass
class PerformanceMetrics:
    """Performance metrics for a test run"""
    response_time: float
    token_count: int
    cost: float
    model_used: str
    success: bool
    error_message: Optional[str] = None

@dataclass
class TestResult:
    """Result of a test run"""
    test_name: str
    success: bool
    metrics: List[PerformanceMetrics]
    total_cost: float
    avg_response_time: float
    success_rate: float
    error_details: Optional[str] = None

class RealLLMPerformanceTester:
    """E2E tester for real LLM performance with actual API calls"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=REAL_LLM_CONFIG["timeout"],
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        self.test_results = []
        self.performance_metrics = {}
        self.cost_metrics = {}
        self.llm_models_tested = set()
        
    async def validate_environment(self) -> bool:
        """Validate that the environment is ready for real LLM testing"""
        print("\n🔧 Validating Environment for Real LLM Testing...")
        
        try:
            # Check if AI API is accessible
            health_response = await self.client.get(f"{BASE_URL}/health")
            if health_response.status_code != 200:
                print(f"❌ AI API health check failed: {health_response.status_code}")
                return False
            
            # Check if Core API is accessible
            core_health_response = await self.client.get(f"{CORE_API_URL}/health")
            if core_health_response.status_code != 200:
                print(f"❌ Core API health check failed: {core_health_response.status_code}")
                return False
            
            # Check premium API health
            premium_health_response = await self.client.get(f"{BASE_URL}/api/v1/premium/health")
            if premium_health_response.status_code != 200:
                print(f"❌ Premium API health check failed: {premium_health_response.status_code}")
                return False
            
            print("✅ Environment validation successful")
            return True
            
        except Exception as e:
            print(f"❌ Environment validation error: {e}")
            return False
    
    async def test_real_llm_chat_performance(self) -> TestResult:
        """Test real LLM chat performance with actual API calls"""
        print("\n🧪 Testing Real LLM Chat Performance...")
        
        test_queries = [
            "Explain the concept of machine learning in simple terms",
            "What are the key differences between supervised and unsupervised learning?",
            "How does a neural network work? Explain with examples.",
            "What is the impact of AI on modern healthcare?",
            "Explain the concept of transfer learning in deep learning"
        ]
        
        all_metrics = []
        total_cost = 0.0
        successful_queries = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"  🔄 Testing query {i}/{len(test_queries)}: {query[:50]}...")
            
            for iteration in range(REAL_LLM_CONFIG["test_iterations"]):
                try:
                    start_time = time.time()
                    
                    payload = {
                        "query": query,
                        "user_id": TEST_USER_ID,
                        "complexity": "medium"
                    }
                    
                    response = await self.client.post(
                        f"{BASE_URL}/api/v1/premium/chat/advanced",
                        json=payload
                    )
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        
                        # Extract metrics from response
                        model_used = response_data.get("model_used", "unknown")
                        estimated_cost = response_data.get("estimated_cost", 0.0)
                        token_count = response_data.get("token_count", 0)
                        
                        # Store model information
                        self.llm_models_tested.add(model_used)
                        
                        # Create metrics
                        metrics = PerformanceMetrics(
                            response_time=response_time,
                            token_count=token_count,
                            cost=estimated_cost,
                            model_used=model_used,
                            success=True
                        )
                        
                        all_metrics.append(metrics)
                        total_cost += estimated_cost
                        successful_queries += 1
                        
                        print(f"    ✅ Iteration {iteration + 1}: {model_used} - {response_time:.2f}s - ${estimated_cost:.4f}")
                        
                    else:
                        print(f"    ❌ Iteration {iteration + 1} failed: {response.status_code}")
                        error_metrics = PerformanceMetrics(
                            response_time=response_time,
                            token_count=0,
                            cost=0.0,
                            model_used="unknown",
                            success=False,
                            error_message=f"HTTP {response.status_code}"
                        )
                        all_metrics.append(error_metrics)
                        
                except Exception as e:
                    print(f"    ❌ Iteration {iteration + 1} error: {e}")
                    error_metrics = PerformanceMetrics(
                        response_time=0.0,
                        token_count=0,
                        cost=0.0,
                        model_used="unknown",
                        success=False,
                        error_message=str(e)
                    )
                    all_metrics.append(error_metrics)
                
                # Small delay between iterations
                await asyncio.sleep(1)
        
        # Calculate aggregate metrics
        successful_metrics = [m for m in all_metrics if m.success]
        avg_response_time = statistics.mean([m.response_time for m in successful_metrics]) if successful_metrics else 0.0
        success_rate = len(successful_metrics) / len(all_metrics) if all_metrics else 0.0
        
        # Store performance data
        self.performance_metrics["real_llm_chat"] = {
            "avg_response_time": avg_response_time,
            "total_cost": total_cost,
            "success_rate": success_rate,
            "models_used": list(self.llm_models_tested)
        }
        
        result = TestResult(
            test_name="Real LLM Chat Performance",
            success=success_rate >= 0.8,
            metrics=all_metrics,
            total_cost=total_cost,
            avg_response_time=avg_response_time,
            success_rate=success_rate
        )
        
        print(f"  📊 Results: {successful_queries}/{len(all_metrics)} successful, "
              f"avg time: {avg_response_time:.2f}s, total cost: ${total_cost:.4f}")
        
        return result
    
    async def test_model_cascading_performance(self) -> TestResult:
        """Test model cascading performance with different complexity levels"""
        print("\n🧪 Testing Model Cascading Performance...")
        
        complexity_test_cases = [
            ("simple", "Hello, how are you?", "gpt-3.5-turbo"),
            ("medium", "Explain the basics of machine learning", "gpt-4"),
            ("complex", "Analyze the impact of AI on healthcare and provide detailed recommendations with examples", "gpt-4-turbo")
        ]
        
        all_metrics = []
        total_cost = 0.0
        successful_queries = 0
        
        for complexity, query, expected_model in complexity_test_cases:
            print(f"  🔄 Testing {complexity} complexity: {query[:50]}...")
            
            for iteration in range(REAL_LLM_CONFIG["test_iterations"]):
                try:
                    start_time = time.time()
                    
                    payload = {
                        "query": query,
                        "user_id": TEST_USER_ID,
                        "complexity": complexity
                    }
                    
                    response = await self.client.post(
                        f"{BASE_URL}/api/v1/premium/chat/cascade",
                        json=payload
                    )
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        
                        model_used = response_data.get("model_used", "unknown")
                        estimated_cost = response_data.get("estimated_cost", 0.0)
                        token_count = response_data.get("token_count", 0)
                        
                        self.llm_models_tested.add(model_used)
                        
                        metrics = PerformanceMetrics(
                            response_time=response_time,
                            token_count=token_count,
                            cost=estimated_cost,
                            model_used=model_used,
                            success=True
                        )
                        
                        all_metrics.append(metrics)
                        total_cost += estimated_cost
                        successful_queries += 1
                        
                        print(f"    ✅ {complexity.capitalize()} - {model_used}: {response_time:.2f}s - ${estimated_cost:.4f}")
                        
                    else:
                        print(f"    ❌ {complexity.capitalize()} failed: {response.status_code}")
                        error_metrics = PerformanceMetrics(
                            response_time=response_time,
                            token_count=0,
                            cost=0.0,
                            model_used="unknown",
                            success=False,
                            error_message=f"HTTP {response.status_code}"
                        )
                        all_metrics.append(error_metrics)
                        
                except Exception as e:
                    print(f"    ❌ {complexity.capitalize()} error: {e}")
                    error_metrics = PerformanceMetrics(
                        response_time=0.0,
                        token_count=0,
                        cost=0.0,
                        model_used="unknown",
                        success=False,
                        error_message=str(e)
                    )
                    all_metrics.append(error_metrics)
                
                await asyncio.sleep(1)
        
        # Calculate aggregate metrics
        successful_metrics = [m for m in all_metrics if m.success]
        avg_response_time = statistics.mean([m.response_time for m in successful_metrics]) if successful_metrics else 0.0
        success_rate = len(successful_metrics) / len(all_metrics) if all_metrics else 0.0
        
        self.performance_metrics["model_cascading"] = {
            "avg_response_time": avg_response_time,
            "total_cost": total_cost,
            "success_rate": success_rate,
            "complexity_levels": len(complexity_test_cases)
        }
        
        result = TestResult(
            test_name="Model Cascading Performance",
            success=success_rate >= 0.8,
            metrics=all_metrics,
            total_cost=total_cost,
            avg_response_time=avg_response_time,
            success_rate=success_rate
        )
        
        print(f"  📊 Results: {successful_queries}/{len(all_metrics)} successful, "
              f"avg time: {avg_response_time:.2f}s, total cost: ${total_cost:.4f}")
        
        return result
    
    async def test_concurrent_llm_performance(self) -> TestResult:
        """Test concurrent LLM performance under load"""
        print("\n🧪 Testing Concurrent LLM Performance...")
        
        concurrent_queries = [
            "Explain quantum computing",
            "What is machine learning?",
            "How do neural networks work?",
            "Explain blockchain technology",
            "What is artificial intelligence?"
        ]
        
        all_metrics = []
        total_cost = 0.0
        successful_queries = 0
        
        async def make_concurrent_request(query: str, query_id: int) -> PerformanceMetrics:
            """Make a single concurrent request"""
            try:
                start_time = time.time()
                
                payload = {
                    "query": query,
                    "user_id": TEST_USER_ID,
                    "complexity": "medium"
                }
                
                response = await self.client.post(
                    f"{BASE_URL}/api/v1/premium/chat/advanced",
                    json=payload
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    model_used = response_data.get("model_used", "unknown")
                    estimated_cost = response_data.get("estimated_cost", 0.0)
                    token_count = response_data.get("token_count", 0)
                    
                    self.llm_models_tested.add(model_used)
                    
                    return PerformanceMetrics(
                        response_time=response_time,
                        token_count=token_count,
                        cost=estimated_cost,
                        model_used=model_used,
                        success=True
                    )
                else:
                    return PerformanceMetrics(
                        response_time=response_time,
                        token_count=0,
                        cost=0.0,
                        model_used="unknown",
                        success=False,
                        error_message=f"HTTP {response.status_code}"
                    )
                    
            except Exception as e:
                return PerformanceMetrics(
                    response_time=0.0,
                    token_count=0,
                    cost=0.0,
                    model_used="unknown",
                    success=False,
                    error_message=str(e)
                )
        
        # Run concurrent requests
        print(f"  🔄 Running {len(concurrent_queries)} concurrent requests...")
        start_time = time.time()
        
        tasks = [make_concurrent_request(query, i) for i, query in enumerate(concurrent_queries)]
        concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_concurrent_time = end_time - start_time
        
        # Process results
        for i, result in enumerate(concurrent_results):
            if isinstance(result, PerformanceMetrics):
                all_metrics.append(result)
                if result.success:
                    successful_queries += 1
                    total_cost += result.cost
                print(f"    ✅ Query {i + 1}: {result.model_used} - {result.response_time:.2f}s - ${result.cost:.4f}")
            else:
                error_metrics = PerformanceMetrics(
                    response_time=0.0,
                    token_count=0,
                    cost=0.0,
                    model_used="unknown",
                    success=False,
                    error_message=f"Exception: {result}"
                )
                all_metrics.append(error_metrics)
                print(f"    ❌ Query {i + 1}: Exception occurred")
        
        # Calculate aggregate metrics
        successful_metrics = [m for m in all_metrics if m.success]
        avg_response_time = statistics.mean([m.response_time for m in successful_metrics]) if successful_metrics else 0.0
        success_rate = len(successful_metrics) / len(all_metrics) if all_metrics else 0.0
        
        self.performance_metrics["concurrent_llm"] = {
            "avg_response_time": avg_response_time,
            "total_concurrent_time": total_concurrent_time,
            "total_cost": total_cost,
            "success_rate": success_rate,
            "concurrent_requests": len(concurrent_queries)
        }
        
        result = TestResult(
            test_name="Concurrent LLM Performance",
            success=success_rate >= 0.8,
            metrics=all_metrics,
            total_cost=total_cost,
            avg_response_time=avg_response_time,
            success_rate=success_rate
        )
        
        print(f"  📊 Results: {successful_queries}/{len(all_metrics)} successful, "
              f"total time: {total_concurrent_time:.2f}s, avg response: {avg_response_time:.2f}s")
        
        return result
    
    async def test_cost_optimization_workflow(self) -> TestResult:
        """Test the complete cost optimization workflow"""
        print("\n🧪 Testing Cost Optimization Workflow...")
        
        # Test different query types to trigger cost optimization
        optimization_queries = [
            "Simple greeting",
            "Explain machine learning concepts",
            "Provide a detailed analysis of AI ethics",
            "Generate a comprehensive guide to deep learning",
            "Explain quantum mechanics with examples"
        ]
        
        all_metrics = []
        total_cost = 0.0
        successful_queries = 0
        
        for i, query in enumerate(optimization_queries, 1):
            print(f"  🔄 Testing optimization query {i}/{len(optimization_queries)}: {query[:50]}...")
            
            try:
                start_time = time.time()
                
                payload = {
                    "query": query,
                    "user_id": TEST_USER_ID,
                    "optimize_cost": True
                }
                
                response = await self.client.post(
                    f"{BASE_URL}/api/v1/premium/chat/advanced",
                    json=payload
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    model_used = response_data.get("model_used", "unknown")
                    estimated_cost = response_data.get("estimated_cost", 0.0)
                    token_count = response_data.get("token_count", 0)
                    optimization_applied = response_data.get("optimization_applied", False)
                    
                    self.llm_models_tested.add(model_used)
                    
                    metrics = PerformanceMetrics(
                        response_time=response_time,
                        token_count=token_count,
                        cost=estimated_cost,
                        model_used=model_used,
                        success=True
                    )
                    
                    all_metrics.append(metrics)
                    total_cost += estimated_cost
                    successful_queries += 1
                    
                    opt_status = "✅" if optimization_applied else "⚠️"
                    print(f"    {opt_status} Query {i}: {model_used} - {response_time:.2f}s - ${estimated_cost:.4f}")
                    
                else:
                    print(f"    ❌ Query {i} failed: {response.status_code}")
                    error_metrics = PerformanceMetrics(
                        response_time=response_time,
                        token_count=0,
                        cost=0.0,
                        model_used="unknown",
                        success=False,
                        error_message=f"HTTP {response.status_code}"
                    )
                    all_metrics.append(error_metrics)
                    
            except Exception as e:
                print(f"    ❌ Query {i} error: {e}")
                error_metrics = PerformanceMetrics(
                    response_time=0.0,
                    token_count=0,
                    cost=0.0,
                    model_used="unknown",
                    success=False,
                    error_message=str(e)
                )
                all_metrics.append(error_metrics)
            
            await asyncio.sleep(1)
        
        # Calculate aggregate metrics
        successful_metrics = [m for m in all_metrics if m.success]
        avg_response_time = statistics.mean([m.response_time for m in successful_metrics]) if successful_metrics else 0.0
        success_rate = len(successful_metrics) / len(all_metrics) if all_metrics else 0.0
        
        self.performance_metrics["cost_optimization"] = {
            "avg_response_time": avg_response_time,
            "total_cost": total_cost,
            "success_rate": success_rate,
            "optimization_queries": len(optimization_queries)
        }
        
        result = TestResult(
            test_name="Cost Optimization Workflow",
            success=success_rate >= 0.8,
            metrics=all_metrics,
            total_cost=total_cost,
            avg_response_time=avg_response_time,
            success_rate=success_rate
        )
        
        print(f"  📊 Results: {successful_queries}/{len(all_metrics)} successful, "
              f"avg time: {avg_response_time:.2f}s, total cost: ${total_cost:.4f}")
        
        return result
    
    async def test_core_api_integration(self) -> TestResult:
        """Test integration with Core API"""
        print("\n🧪 Testing Core API Integration...")
        
        # Test creating a simple primitive through the AI API
        test_primitive_data = {
            "name": "Test Knowledge Primitive",
            "description": "A test primitive created during E2E testing",
            "type": "concept",
            "content": "This is test content for the primitive"
        }
        
        all_metrics = []
        total_cost = 0.0
        successful_operations = 0
        
        try:
            start_time = time.time()
            
            # Test primitive creation through AI API
            payload = {
                "query": f"Create a knowledge primitive: {json.dumps(test_primitive_data)}",
                "user_id": TEST_USER_ID,
                "operation": "create_primitive"
            }
            
            response = await self.client.post(
                f"{BASE_URL}/api/v1/premium/chat/advanced",
                json=payload
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                
                model_used = response_data.get("model_used", "unknown")
                estimated_cost = response_data.get("estimated_cost", 0.0)
                token_count = response_data.get("token_count", 0)
                
                self.llm_models_tested.add(model_used)
                
                metrics = PerformanceMetrics(
                    response_time=response_time,
                    token_count=token_count,
                    cost=estimated_cost,
                    model_used=model_used,
                    success=True
                )
                
                all_metrics.append(metrics)
                total_cost += estimated_cost
                successful_operations += 1
                
                print(f"  ✅ Core API integration successful: {model_used} - {response_time:.2f}s - ${estimated_cost:.4f}")
                
            else:
                print(f"  ❌ Core API integration failed: {response.status_code}")
                error_metrics = PerformanceMetrics(
                    response_time=response_time,
                    token_count=0,
                    cost=0.0,
                    model_used="unknown",
                    success=False,
                    error_message=f"HTTP {response.status_code}"
                )
                all_metrics.append(error_metrics)
                
        except Exception as e:
            print(f"  ❌ Core API integration error: {e}")
            error_metrics = PerformanceMetrics(
                response_time=0.0,
                token_count=0,
                cost=0.0,
                model_used="unknown",
                success=False,
                error_message=str(e)
            )
            all_metrics.append(error_metrics)
        
        # Calculate aggregate metrics
        successful_metrics = [m for m in all_metrics if m.success]
        avg_response_time = statistics.mean([m.response_time for m in successful_metrics]) if successful_metrics else 0.0
        success_rate = len(successful_metrics) / len(all_metrics) if all_metrics else 0.0
        
        self.performance_metrics["core_api_integration"] = {
            "avg_response_time": avg_response_time,
            "total_cost": total_cost,
            "success_rate": success_rate
        }
        
        result = TestResult(
            test_name="Core API Integration",
            success=success_rate >= 0.8,
            metrics=all_metrics,
            total_cost=total_cost,
            avg_response_time=avg_response_time,
            success_rate=success_rate
        )
        
        print(f"  📊 Results: {successful_operations}/{len(all_metrics)} successful, "
              f"avg time: {avg_response_time:.2f}s, total cost: ${total_cost:.4f}")
        
        return result
    
    async def run_all_tests(self) -> List[TestResult]:
        """Run all real LLM performance tests"""
        print("🚀 Starting Real LLM Performance E2E Tests")
        print(f"📅 Test started at: {datetime.utcnow()}")
        print(f"⚙️  Configuration: {json.dumps(REAL_LLM_CONFIG, indent=2)}")
        
        # Validate environment first
        if not await self.validate_environment():
            print("❌ Environment validation failed. Aborting tests.")
            return [TestResult(
                test_name="Environment Validation",
                success=False,
                metrics=[],
                total_cost=0.0,
                avg_response_time=0.0,
                success_rate=0.0,
                error_details="Environment validation failed"
            )]
        
        tests = [
            ("Real LLM Chat Performance", self.test_real_llm_chat_performance),
            ("Model Cascading Performance", self.test_model_cascading_performance),
            ("Concurrent LLM Performance", self.test_concurrent_llm_performance),
            ("Cost Optimization Workflow", self.test_cost_optimization_workflow)
            # ("Core API Integration", self.test_core_api_integration)  # Skipped - Core API not running
        ]
        
        results = []
        
        for test_name, test_func in tests:
            try:
                print(f"\n{'='*80}")
                print(f"🧪 Running: {test_name}")
                print(f"{'='*80}")
                
                result = await test_func()
                results.append(result)
                
            except Exception as e:
                print(f"❌ {test_name} test failed: {e}")
                error_result = TestResult(
                    test_name=test_name,
                    success=False,
                    metrics=[],
                    total_cost=0.0,
                    avg_response_time=0.0,
                    success_rate=0.0,
                    error_details=str(e)
                )
                results.append(error_result)
        
        # Print detailed summary
        await self.print_detailed_summary(results)
        
        return results
    
    async def print_detailed_summary(self, results: List[TestResult]):
        """Print detailed test summary with real performance metrics"""
        print("\n" + "="*100)
        print("📊 REAL LLM PERFORMANCE E2E TEST DETAILED SUMMARY")
        print("="*100)
        
        passed = sum(1 for result in results if result.success)
        total = len(results)
        
        # Test results
        print("\n🧪 TEST RESULTS:")
        for result in results:
            status = "✅ PASSED" if result.success else "❌ FAILED"
            print(f"  {status} {result.test_name}")
            if result.error_details:
                print(f"    Error: {result.error_details}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        # Performance metrics
        if self.performance_metrics:
            print("\n⚡ PERFORMANCE METRICS:")
            for test_name, metrics in self.performance_metrics.items():
                print(f"  📊 {test_name}:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        if "time" in key:
                            print(f"    {key}: {value:.3f}s")
                        elif "cost" in key:
                            print(f"    {key}: ${value:.4f}")
                        else:
                            print(f"    {key}: {value:.3f}")
                    else:
                        print(f"    {key}: {value}")
        
        # Cost analysis
        total_cost = sum(result.total_cost for result in results)
        print(f"\n💰 TOTAL COST ANALYSIS:")
        print(f"  Total cost across all tests: ${total_cost:.4f}")
        print(f"  Cost threshold: ${REAL_LLM_CONFIG['cost_threshold']}")
        
        if total_cost > REAL_LLM_CONFIG['cost_threshold']:
            print(f"  ⚠️  Total cost exceeds threshold!")
        
        # Models tested
        if self.llm_models_tested:
            print(f"\n🤖 LLM MODELS TESTED:")
            for model in sorted(self.llm_models_tested):
                print(f"  • {model}")
        
        # Final status
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! Real LLM performance is excellent.")
        elif passed >= total * 0.8:
            print("\n⚠️  Most tests passed. Some issues detected but performance is good.")
        else:
            print("\n❌ Multiple tests failed. Real LLM performance needs attention.")
        
        print("="*100)
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()

async def main():
    """Main test runner"""
    tester = RealLLMPerformanceTester()
    try:
        results = await tester.run_all_tests()
        
        # Exit with appropriate code
        passed = sum(1 for result in results if result.success)
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
