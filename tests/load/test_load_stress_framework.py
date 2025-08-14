#!/usr/bin/env python3
"""
Load and Stress Testing Framework for Blueprint Section Operations
Tests system performance under high concurrent load and stress conditions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import time
import statistics
import random
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import httpx
import json

# Test configuration
AI_API_BASE_URL = "http://localhost:8000"
CORE_API_BASE_URL = "http://localhost:3000"
API_KEY = "test-token"

@dataclass
class LoadTestResult:
    """Result of a load test"""
    test_name: str
    concurrent_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput: float  # requests per second
    success_rate: float
    test_duration: float
    errors: List[str]

@dataclass
class StressTestResult:
    """Result of a stress test"""
    test_name: str
    max_concurrent_users: int
    breaking_point: Optional[int]
    performance_degradation: Dict[str, float]
    error_threshold_reached: bool
    system_recovery: bool
    test_duration: float

class LoadTestWorker:
    """Individual worker for load testing"""
    
    def __init__(self, worker_id: int, api_key: str):
        self.worker_id = worker_id
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"Authorization": f"Bearer {api_key}"}
        )
        self.results: List[Dict[str, Any]] = []
    
    async def run_blueprint_operations(self, num_operations: int) -> List[Dict[str, Any]]:
        """Run a series of blueprint operations"""
        operations = []
        
        for i in range(num_operations):
            try:
                start_time = time.time()
                
                # Randomly choose operation type
                operation_type = random.choice([
                    "create_blueprint",
                    "get_sections",
                    "search_content",
                    "generate_primitives"
                ])
                
                if operation_type == "create_blueprint":
                    result = await self._create_test_blueprint()
                elif operation_type == "get_sections":
                    result = await self._get_sections()
                elif operation_type == "search_content":
                    result = await self._search_content()
                else:
                    result = await self._generate_primitives()
                
                end_time = time.time()
                response_time = end_time - start_time
                
                operations.append({
                    "operation_type": operation_type,
                    "success": True,
                    "response_time": response_time,
                    "worker_id": self.worker_id,
                    "operation_id": i
                })
                
            except Exception as e:
                end_time = time.time()
                response_time = end_time - start_time
                
                operations.append({
                    "operation_type": operation_type if 'operation_type' in locals() else "unknown",
                    "success": False,
                    "response_time": response_time,
                    "error": str(e),
                    "worker_id": self.worker_id,
                    "operation_id": i
                })
        
        self.results.extend(operations)
        return operations
    
    async def _create_test_blueprint(self) -> Dict[str, Any]:
        """Create a test blueprint"""
        blueprint_data = {
            "title": f"Load Test Blueprint {self.worker_id}-{int(time.time())}",
            "description": "Blueprint for load testing",
            "user_id": f"load-test-user-{self.worker_id}",
            "difficulty": "BEGINNER",
            "tags": ["load-test"]
        }
        
        response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/blueprints",
            json=blueprint_data
        )
        
        if response.status_code != 201:
            raise Exception(f"Blueprint creation failed: {response.status_code}")
        
        return response.json()
    
    async def _get_sections(self) -> Dict[str, Any]:
        """Get blueprint sections"""
        response = await self.client.get(
            f"{AI_API_BASE_URL}/api/v1/blueprint/sections",
            params={"user_id": f"load-test-user-{self.worker_id}"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Section retrieval failed: {response.status_code}")
        
        return response.json()
    
    async def _search_content(self) -> Dict[str, Any]:
        """Search content"""
        response = await self.client.post(
            f"{AI_API_BASE_URL}/api/v1/premium/search",
            json={
                "query": "machine learning basics",
                "user_id": f"load-test-user-{self.worker_id}",
                "complexity": "medium"
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Search failed: {response.status_code}")
        
        return response.json()
    
    async def _generate_primitives(self) -> Dict[str, Any]:
        """Generate primitives"""
        response = await self.client.post(
            f"{AI_API_BASE_URL}/api/v1/blueprint/sections/test-section/generate-primitives",
            json={
                "user_id": f"load-test-user-{self.worker_id}",
                "complexity": "medium"
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Primitive generation failed: {response.status_code}")
        
        return response.json()
    
    async def cleanup(self):
        """Clean up resources"""
        await self.client.aclose()

class LoadTestRunner:
    """Runs load tests with multiple concurrent workers"""
    
    def __init__(self):
        self.test_results: List[LoadTestResult] = []
    
    async def run_concurrent_load_test(
        self,
        concurrent_users: int,
        operations_per_user: int,
        test_name: str = "concurrent_blueprint_operations"
    ) -> LoadTestResult:
        """Run a load test with specified number of concurrent users"""
        print(f"🧪 Running load test: {concurrent_users} concurrent users, {operations_per_user} operations each")
        
        start_time = time.time()
        
        # Create workers
        workers = [
            LoadTestWorker(i, API_KEY) 
            for i in range(concurrent_users)
        ]
        
        # Run operations concurrently
        tasks = [
            worker.run_blueprint_operations(operations_per_user)
            for worker in workers
        ]
        
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_operations = []
        failed_operations = []
        all_operations = []
        
        for result in all_results:
            if isinstance(result, list):
                all_operations.extend(result)
                for op in result:
                    if op.get("success", False):
                        successful_operations.append(op)
                    else:
                        failed_operations.append(op)
            else:
                # Handle exceptions
                failed_operations.append({
                    "operation_type": "unknown",
                    "success": False,
                    "response_time": 0,
                    "error": str(result),
                    "worker_id": -1,
                    "operation_id": -1
                })
        
        # Calculate metrics
        total_requests = len(all_operations)
        successful_requests = len(successful_operations)
        failed_requests = len(failed_operations)
        
        if successful_operations:
            response_times = [op["response_time"] for op in successful_operations]
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            # Calculate percentiles
            sorted_times = sorted(response_times)
            p95_response_time = statistics.quantiles(sorted_times, n=20)[18] if len(sorted_times) >= 20 else max_response_time
            p99_response_time = statistics.quantiles(sorted_times, n=100)[98] if len(sorted_times) >= 100 else max_response_time
        else:
            avg_response_time = min_response_time = max_response_time = p95_response_time = p99_response_time = 0
        
        end_time = time.time()
        test_duration = end_time - start_time
        
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        throughput = total_requests / test_duration if test_duration > 0 else 0
        
        # Collect errors
        errors = []
        for op in failed_operations:
            if "error" in op:
                errors.append(op["error"])
        
        # Cleanup workers
        for worker in workers:
            await worker.cleanup()
        
        result = LoadTestResult(
            test_name=test_name,
            concurrent_users=concurrent_users,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            throughput=throughput,
            success_rate=success_rate,
            test_duration=test_duration,
            errors=errors
        )
        
        self.test_results.append(result)
        return result

class StressTestRunner:
    """Runs stress tests to find system breaking points"""
    
    def __init__(self):
        self.test_results: List[StressTestResult] = []
    
    async def run_stress_test(
        self,
        start_concurrent_users: int,
        max_concurrent_users: int,
        step_size: int,
        operations_per_user: int,
        test_name: str = "blueprint_operations_stress_test"
    ) -> StressTestResult:
        """Run a stress test to find the breaking point"""
        print(f"🧪 Running stress test: {start_concurrent_users} to {max_concurrent_users} users")
        
        start_time = time.time()
        breaking_point = None
        performance_degradation = {}
        error_threshold_reached = False
        system_recovery = False
        
        load_runner = LoadTestRunner()
        
        # Test with increasing load
        for concurrent_users in range(start_concurrent_users, max_concurrent_users + 1, step_size):
            print(f"  🔄 Testing with {concurrent_users} concurrent users...")
            
            try:
                result = await load_runner.run_concurrent_load_test(
                    concurrent_users=concurrent_users,
                    operations_per_user=operations_per_user,
                    test_name=f"{test_name}_{concurrent_users}_users"
                )
                
                # Check for breaking point
                if result.success_rate < 0.95:  # 95% success rate threshold
                    if breaking_point is None:
                        breaking_point = concurrent_users
                        print(f"    ⚠️  Breaking point detected at {concurrent_users} users")
                
                if result.success_rate < 0.8:  # 80% success rate threshold
                    error_threshold_reached = True
                    print(f"    ❌ Error threshold reached at {concurrent_users} users")
                    break
                
                # Track performance degradation
                if concurrent_users > start_concurrent_users:
                    baseline_result = load_runner.test_results[0]
                    degradation = {
                        "response_time": (result.avg_response_time - baseline_result.avg_response_time) / baseline_result.avg_response_time * 100,
                        "throughput": (baseline_result.throughput - result.throughput) / baseline_result.throughput * 100,
                        "success_rate": (baseline_result.success_rate - result.success_rate) * 100
                    }
                    performance_degradation[concurrent_users] = degradation
                
                # Small delay between tests
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"    ❌ Test failed with {concurrent_users} users: {e}")
                if breaking_point is None:
                    breaking_point = concurrent_users
                break
        
        # Test system recovery
        if breaking_point and breaking_point > start_concurrent_users:
            print(f"  🔄 Testing system recovery with {start_concurrent_users} users...")
            try:
                recovery_result = await load_runner.run_concurrent_load_test(
                    concurrent_users=start_concurrent_users,
                    operations_per_user=operations_per_user,
                    test_name=f"{test_name}_recovery_test"
                )
                
                system_recovery = recovery_result.success_rate >= 0.95
                print(f"    {'✅' if system_recovery else '❌'} System recovery: {recovery_result.success_rate*100:.1f}%")
                
            except Exception as e:
                print(f"    ❌ Recovery test failed: {e}")
                system_recovery = False
        
        end_time = time.time()
        test_duration = end_time - start_time
        
        result = StressTestResult(
            test_name=test_name,
            max_concurrent_users=max_concurrent_users,
            breaking_point=breaking_point,
            performance_degradation=performance_degradation,
            error_threshold_reached=error_threshold_reached,
            system_recovery=system_recovery,
            test_duration=test_duration
        )
        
        self.test_results.append(result)
        return result

class LoadStressTestFramework:
    """Main framework for running load and stress tests"""
    
    def __init__(self):
        self.load_runner = LoadTestRunner()
        self.stress_runner = StressTestRunner()
    
    async def run_comprehensive_load_test(self) -> Dict[str, Any]:
        """Run comprehensive load testing"""
        print("🚀 Starting Comprehensive Load Testing")
        print("=" * 60)
        
        load_scenarios = [
            {"users": 10, "operations": 5, "name": "light_load"},
            {"users": 25, "operations": 5, "name": "medium_load"},
            {"users": 50, "operations": 5, "name": "high_load"},
            {"users": 100, "operations": 3, "name": "very_high_load"}
        ]
        
        results = {}
        for scenario in load_scenarios:
            print(f"\n📊 Testing {scenario['name']}: {scenario['users']} users, {scenario['operations']} operations each")
            
            result = await self.load_runner.run_concurrent_load_test(
                concurrent_users=scenario["users"],
                operations_per_user=scenario["operations"],
                test_name=scenario["name"]
            )
            
            results[scenario["name"]] = result
            
            # Print results
            print(f"  ✅ Success Rate: {result.success_rate*100:.1f}%")
            print(f"  ⏱️  Avg Response Time: {result.avg_response_time*1000:.1f}ms")
            print(f"  📈 Throughput: {result.throughput:.2f} req/sec")
            
            if result.success_rate < 0.95:
                print(f"  ⚠️  Performance below target (95% success rate)")
        
        return results
    
    async def run_stress_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive stress testing"""
        print("\n🚀 Starting Comprehensive Stress Testing")
        print("=" * 60)
        
        stress_scenarios = [
            {
                "start_users": 10,
                "max_users": 200,
                "step_size": 25,
                "operations_per_user": 3,
                "name": "blueprint_operations_stress"
            }
        ]
        
        results = {}
        for scenario in stress_scenarios:
            print(f"\n📊 Testing {scenario['name']}")
            
            result = await self.stress_runner.run_stress_test(
                start_concurrent_users=scenario["start_users"],
                max_concurrent_users=scenario["max_users"],
                step_size=scenario["step_size"],
                operations_per_user=scenario["operations_per_user"],
                test_name=scenario["name"]
            )
            
            results[scenario["name"]] = result
            
            # Print results
            if result.breaking_point:
                print(f"  ⚠️  Breaking Point: {result.breaking_point} concurrent users")
            else:
                print(f"  ✅ No breaking point detected up to {scenario['max_users']} users")
            
            print(f"  🔄 System Recovery: {'✅' if result.system_recovery else '❌'}")
            print(f"  ❌ Error Threshold: {'✅' if result.error_threshold_reached else '❌'}")
        
        return results
    
    def print_comprehensive_summary(self, load_results: Dict[str, Any], stress_results: Dict[str, Any]):
        """Print comprehensive test summary"""
        print("\n📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        
        # Load test summary
        print("\n🔍 LOAD TEST RESULTS:")
        for test_name, result in load_results.items():
            print(f"  {test_name}:")
            print(f"    Users: {result.concurrent_users}")
            print(f"    Success Rate: {result.success_rate*100:.1f}%")
            print(f"    Avg Response: {result.avg_response_time*1000:.1f}ms")
            print(f"    Throughput: {result.throughput:.2f} req/sec")
            
            # Performance assessment
            if result.success_rate >= 0.95 and result.avg_response_time < 1.0:
                print(f"    Status: ✅ Excellent")
            elif result.success_rate >= 0.9 and result.avg_response_time < 2.0:
                print(f"    Status: ⚠️  Good")
            else:
                print(f"    Status: ❌ Needs Improvement")
        
        # Stress test summary
        print("\n🔍 STRESS TEST RESULTS:")
        for test_name, result in stress_results.items():
            print(f"  {test_name}:")
            if result.breaking_point:
                print(f"    Breaking Point: {result.breaking_point} users")
            else:
                print(f"    Breaking Point: Not reached")
            
            print(f"    System Recovery: {'✅' if result.system_recovery else '❌'}")
            print(f"    Error Threshold: {'✅' if result.error_threshold_reached else '❌'}")
            
            # Performance degradation analysis
            if result.performance_degradation:
                print(f"    Performance Degradation:")
                for user_count, degradation in result.performance_degradation.items():
                    print(f"      {user_count} users: RT +{degradation['response_time']:.1f}%, TP -{degradation['throughput']:.1f}%")
        
        # Overall assessment
        print("\n🎯 OVERALL ASSESSMENT:")
        
        # Calculate overall success rate
        all_load_results = list(load_results.values())
        overall_success_rate = sum(r.success_rate for r in all_load_results) / len(all_load_results) if all_load_results else 0
        
        # Calculate overall performance
        all_response_times = [r.avg_response_time for r in all_load_results]
        overall_avg_response = statistics.mean(all_response_times) if all_response_times else 0
        
        print(f"  Overall Success Rate: {overall_success_rate*100:.1f}%")
        print(f"  Overall Avg Response Time: {overall_avg_response*1000:.1f}ms")
        
        if overall_success_rate >= 0.95 and overall_avg_response < 1.0:
            print(f"  Overall Status: 🎉 EXCELLENT - System ready for production load")
        elif overall_success_rate >= 0.9 and overall_avg_response < 2.0:
            print(f"  Overall Status: ✅ GOOD - System performs well under load")
        elif overall_success_rate >= 0.8:
            print(f"  Overall Status: ⚠️  ACCEPTABLE - Some performance issues detected")
        else:
            print(f"  Overall Status: ❌ POOR - Significant performance issues need attention")
    
    async def save_test_results(self, load_results: Dict[str, Any], stress_results: Dict[str, Any]):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"load_stress_test_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = {
            "load_test_results": {
                name: {
                    "test_name": result.test_name,
                    "concurrent_users": result.concurrent_users,
                    "total_requests": result.total_requests,
                    "successful_requests": result.successful_requests,
                    "failed_requests": result.failed_requests,
                    "avg_response_time": result.avg_response_time,
                    "min_response_time": result.min_response_time,
                    "max_response_time": result.max_response_time,
                    "p95_response_time": result.p95_response_time,
                    "p99_response_time": result.p99_response_time,
                    "throughput": result.throughput,
                    "success_rate": result.success_rate,
                    "test_duration": result.test_duration,
                    "errors": result.errors
                }
                for name, result in load_results.items()
            },
            "stress_test_results": {
                name: {
                    "test_name": result.test_name,
                    "max_concurrent_users": result.max_concurrent_users,
                    "breaking_point": result.breaking_point,
                    "performance_degradation": result.performance_degradation,
                    "error_threshold_reached": result.error_threshold_reached,
                    "system_recovery": result.system_recovery,
                    "test_duration": result.test_duration
                }
                for name, result in stress_results.items()
            },
            "test_metadata": {
                "timestamp": timestamp,
                "total_load_tests": len(load_results),
                "total_stress_tests": len(stress_results)
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Test results saved to: {results_file}")

async def main():
    """Main function to run load and stress tests"""
    print("🚀 Load and Stress Testing Framework")
    print("=" * 60)
    
    framework = LoadStressTestFramework()
    
    try:
        # Run comprehensive load testing
        load_results = await framework.run_comprehensive_load_test()
        
        # Run stress testing
        stress_results = await framework.run_stress_test_suite()
        
        # Print comprehensive summary
        framework.print_comprehensive_summary(load_results, stress_results)
        
        # Save results
        await framework.save_test_results(load_results, stress_results)
        
        print("\n🎉 Load and stress testing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
