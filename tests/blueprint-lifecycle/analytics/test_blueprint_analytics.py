#!/usr/bin/env python3
"""
Comprehensive Blueprint Analytics E2E Test
Tests the complete analytics system including:
- Usage tracking and metrics collection
- Performance analytics
- User behavior analysis
- Insights generation
- Reporting and dashboards
- Data aggregation and processing
"""

import asyncio
import httpx
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Configuration
AI_API_BASE_URL = "http://localhost:8000"
CORE_API_BASE_URL = "http://localhost:3000"
API_KEY = "test_api_key_123"
TEST_USER_ID = "test-user-123"

@dataclass
class TestResult:
    step: str
    status: str  # PASS, FAIL, SKIP
    details: str = None
    error: Any = None
    duration: float = 0.0
    metadata: Dict[str, Any] = None

class BlueprintAnalyticsTester:
    def __init__(self):
        self.results: List[TestResult] = []
        self.test_data = {
            "blueprint_ids": [],
            "analytics_data": {},
            "performance_metrics": {}
        }
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0)
        )
        
    async def run(self) -> None:
        """Run the complete blueprint analytics test suite."""
        print("🚀 Starting Comprehensive Blueprint Analytics E2E Test\n")
        
        try:
            await self.run_step(self.test_environment_setup)
            await self.run_step(self.test_analytics_data_collection)
            await self.run_step(self.test_performance_metrics)
            await self.run_step(self.test_user_behavior_tracking)
            await self.run_step(self.test_insights_generation)
            await self.run_step(self.test_reporting_system)
            await self.run_step(self.test_data_aggregation)
            await self.run_step(self.test_analytics_api_endpoints)
        except Exception as error:
            print(f"\n❌ Test suite aborted due to critical failure: {error}")
        finally:
            await self.client.aclose()
            self.print_results()
    
    async def run_step(self, step_func, continue_on_error: bool = False) -> None:
        """Execute a test step with error handling and timing."""
        start_time = time.time()
        try:
            await step_func()
            duration = time.time() - start_time
            self.results.append(TestResult(
                step_func.__name__.replace('test_', '').replace('_', ' ').title(),
                "PASS",
                f"Completed successfully in {duration:.2f}s",
                duration=duration
            ))
        except Exception as error:
            duration = time.time() - start_time
            self.results.append(TestResult(
                step_func.__name__.replace('test_', '').replace('_', ' ').title(),
                "FAIL",
                f"Failed after {duration:.2f}s: {str(error)}",
                error,
                duration=duration
            ))
            if not continue_on_error:
                raise error

    async def test_environment_setup(self) -> None:
        """Test 1: Verify environment is ready for analytics testing."""
        print("🔧 Step 1: Environment Setup and Validation...")
        
        # Check AI API health
        response = await self.client.get(f"{AI_API_BASE_URL}/api/health")
        if response.status_code != 200:
            raise Exception(f"AI API health check failed: {response.status_code}")
        
        # Check Core API health
        response = await self.client.get(f"{CORE_API_BASE_URL}/health")
        if response.status_code != 200:
            raise Exception(f"Core API health check failed: {response.status_code}")
        
        # Check analytics endpoints
        response = await self.client.get(f"{CORE_API_BASE_URL}/api/analytics/health")
        if response.status_code != 200:
            raise Exception(f"Analytics health check failed: {response.status_code}")
        
        print("✅ Environment setup completed successfully")

    async def test_analytics_data_collection(self) -> None:
        """Test 2: Test analytics data collection system."""
        print("📊 Step 2: Testing Analytics Data Collection...")
        
        # Create test blueprints to generate analytics data
        test_blueprints = [
            {
                "title": "Test Analytics Blueprint 1",
                "content": "This is a test blueprint for analytics testing",
                "metadata": {"category": "test", "tags": ["analytics", "testing"]}
            },
            {
                "title": "Test Analytics Blueprint 2", 
                "content": "Another test blueprint for analytics validation",
                "metadata": {"category": "test", "tags": ["validation", "testing"]}
            }
        ]
        
        for blueprint in test_blueprints:
            response = await self.client.post(
                f"{CORE_API_BASE_URL}/api/blueprints",
                json=blueprint,
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            if response.status_code == 201:
                blueprint_id = response.json().get("id")
                self.test_data["blueprint_ids"].append(blueprint_id)
                print(f"✅ Created test blueprint: {blueprint_id}")
            else:
                raise Exception(f"Failed to create test blueprint: {response.status_code}")
        
        # Simulate user interactions to generate analytics data
        for blueprint_id in self.test_data["blueprint_ids"]:
            # Simulate view
            await self.client.post(
                f"{CORE_API_BASE_URL}/api/analytics/events",
                json={
                    "event_type": "blueprint_view",
                    "blueprint_id": blueprint_id,
                    "user_id": TEST_USER_ID,
                    "timestamp": datetime.utcnow().isoformat()
                },
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            # Simulate search
            await self.client.post(
                f"{CORE_API_BASE_URL}/api/analytics/events",
                json={
                    "event_type": "blueprint_search",
                    "blueprint_id": blueprint_id,
                    "user_id": TEST_USER_ID,
                    "query": "test analytics",
                    "timestamp": datetime.utcnow().isoformat()
                },
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
        
        print("✅ Analytics data collection testing completed")

    async def test_performance_metrics(self) -> None:
        """Test 3: Test performance metrics collection and analysis."""
        print("⚡ Step 3: Testing Performance Metrics...")
        
        # Test response time metrics
        start_time = time.time()
        response = await self.client.get(f"{CORE_API_BASE_URL}/api/blueprints")
        response_time = time.time() - start_time
        
        self.test_data["performance_metrics"]["blueprint_list_response_time"] = response_time
        
        # Test concurrent request handling
        async def make_concurrent_requests():
            tasks = []
            for i in range(10):
                task = self.client.get(f"{CORE_API_BASE_URL}/api/blueprints")
                tasks.append(task)
            return await asyncio.gather(*tasks)
        
        start_time = time.time()
        concurrent_responses = await make_concurrent_requests()
        concurrent_time = time.time() - start_time
        
        self.test_data["performance_metrics"]["concurrent_requests_time"] = concurrent_time
        self.test_data["performance_metrics"]["concurrent_requests_count"] = len(concurrent_responses)
        
        # Test memory usage metrics
        response = await self.client.get(f"{CORE_API_BASE_URL}/api/analytics/performance/memory")
        if response.status_code == 200:
            memory_data = response.json()
            self.test_data["performance_metrics"]["memory_usage"] = memory_data
            print("✅ Memory usage metrics retrieved")
        
        print("✅ Performance metrics testing completed")

    async def test_user_behavior_tracking(self) -> None:
        """Test 4: Test user behavior tracking and analysis."""
        print("👤 Step 4: Testing User Behavior Tracking...")
        
        # Test user session tracking
        session_data = {
            "user_id": TEST_USER_ID,
            "session_start": datetime.utcnow().isoformat(),
            "user_agent": "test-agent",
            "ip_address": "127.0.0.1"
        }
        
        response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/analytics/sessions",
            json=session_data,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 201:
            session_id = response.json().get("session_id")
            print(f"✅ User session created: {session_id}")
        else:
            raise Exception(f"Failed to create user session: {response.status_code}")
        
        # Test user interaction patterns
        interaction_data = {
            "user_id": TEST_USER_ID,
            "blueprint_id": self.test_data["blueprint_ids"][0],
            "interaction_type": "read",
            "duration": 30,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/analytics/interactions",
            json=interaction_data,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 201:
            print("✅ User interaction tracked successfully")
        else:
            raise Exception(f"Failed to track user interaction: {response.status_code}")
        
        # Test user preference analysis
        response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/analytics/users/{TEST_USER_ID}/preferences",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 200:
            preferences = response.json()
            self.test_data["analytics_data"]["user_preferences"] = preferences
            print("✅ User preferences retrieved successfully")
        
        print("✅ User behavior tracking testing completed")

    async def test_insights_generation(self) -> None:
        """Test 5: Test insights generation and analysis."""
        print("🧠 Step 5: Testing Insights Generation...")
        
        # Test trend analysis
        response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/analytics/insights/trends",
            params={"timeframe": "7d"},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 200:
            trends = response.json()
            self.test_data["analytics_data"]["trends"] = trends
            print("✅ Trend insights generated successfully")
        else:
            raise Exception(f"Failed to generate trend insights: {response.status_code}")
        
        # Test recommendation engine
        response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/analytics/recommendations",
            params={"user_id": TEST_USER_ID, "limit": 5},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 200:
            recommendations = response.json()
            self.test_data["analytics_data"]["recommendations"] = recommendations
            print("✅ Recommendations generated successfully")
        else:
            raise Exception(f"Failed to generate recommendations: {response.status_code}")
        
        # Test anomaly detection
        response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/analytics/insights/anomalies",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 200:
            anomalies = response.json()
            self.test_data["analytics_data"]["anomalies"] = anomalies
            print("✅ Anomaly detection completed successfully")
        
        print("✅ Insights generation testing completed")

    async def test_reporting_system(self) -> None:
        """Test 6: Test reporting and dashboard system."""
        print("📈 Step 6: Testing Reporting System...")
        
        # Test dashboard data retrieval
        response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/analytics/dashboard",
            params={"timeframe": "24h"},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 200:
            dashboard_data = response.json()
            self.test_data["analytics_data"]["dashboard"] = dashboard_data
            print("✅ Dashboard data retrieved successfully")
        else:
            raise Exception(f"Failed to retrieve dashboard data: {response.status_code}")
        
        # Test report generation
        report_request = {
            "report_type": "usage_summary",
            "timeframe": "7d",
            "format": "json",
            "filters": {"user_id": TEST_USER_ID}
        }
        
        response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/analytics/reports",
            json=report_request,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 202:
            report_id = response.json().get("report_id")
            print(f"✅ Report generation started: {report_id}")
            
            # Wait for report completion
            for _ in range(10):
                await asyncio.sleep(1)
                status_response = await self.client.get(
                    f"{CORE_API_BASE_URL}/api/analytics/reports/{report_id}",
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data.get("status") == "completed":
                        self.test_data["analytics_data"]["generated_report"] = status_data
                        print("✅ Report generation completed successfully")
                        break
        else:
            raise Exception(f"Failed to start report generation: {response.status_code}")
        
        print("✅ Reporting system testing completed")

    async def test_data_aggregation(self) -> None:
        """Test 7: Test data aggregation and processing."""
        print("🔢 Step 7: Testing Data Aggregation...")
        
        # Test real-time aggregation
        response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/analytics/aggregations/realtime",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 200:
            realtime_data = response.json()
            self.test_data["analytics_data"]["realtime_aggregation"] = realtime_data
            print("✅ Real-time aggregation data retrieved")
        else:
            raise Exception(f"Failed to retrieve real-time aggregation: {response.status_code}")
        
        # Test batch aggregation
        response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/analytics/aggregations/batch",
            params={"date": datetime.utcnow().date().isoformat()},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 200:
            batch_data = response.json()
            self.test_data["analytics_data"]["batch_aggregation"] = batch_data
            print("✅ Batch aggregation data retrieved")
        else:
            raise Exception(f"Failed to retrieve batch aggregation: {response.status_code}")
        
        # Test custom aggregation queries
        custom_query = {
            "metrics": ["total_views", "unique_users", "avg_session_duration"],
            "dimensions": ["blueprint_category", "user_type"],
            "filters": {"timeframe": "30d"}
        }
        
        response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/analytics/aggregations/custom",
            json=custom_query,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 200:
            custom_data = response.json()
            self.test_data["analytics_data"]["custom_aggregation"] = custom_data
            print("✅ Custom aggregation query executed successfully")
        else:
            raise Exception(f"Failed to execute custom aggregation: {response.status_code}")
        
        print("✅ Data aggregation testing completed")

    async def test_analytics_api_endpoints(self) -> None:
        """Test 8: Test all analytics API endpoints."""
        print("🔌 Step 8: Testing Analytics API Endpoints...")
        
        # Test analytics health endpoint
        response = await self.client.get(f"{CORE_API_BASE_URL}/api/analytics/health")
        if response.status_code != 200:
            raise Exception(f"Analytics health endpoint failed: {response.status_code}")
        
        # Test metrics endpoint
        response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/analytics/metrics",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        if response.status_code != 200:
            raise Exception(f"Metrics endpoint failed: {response.status_code}")
        
        # Test events endpoint
        response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/analytics/events",
            params={"limit": 10},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        if response.status_code != 200:
            raise Exception(f"Events endpoint failed: {response.status_code}")
        
        # Test users endpoint
        response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/analytics/users",
            params={"limit": 10},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        if response.status_code != 200:
            raise Exception(f"Users endpoint failed: {response.status_code}")
        
        print("✅ All analytics API endpoints tested successfully")

    def print_results(self) -> None:
        """Print comprehensive test results."""
        print("\n" + "="*80)
        print("📊 BLUEPRINT ANALYTICS E2E TEST RESULTS")
        print("="*80)
        
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == "PASS"])
        failed_tests = len([r for r in self.results if r.status == "FAIL"])
        total_duration = sum(r.duration for r in self.results)
        
        print(f"\n📈 SUMMARY:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   ⏱️  Total Duration: {total_duration:.2f}s")
        
        print(f"\n📋 DETAILED RESULTS:")
        for result in self.results:
            status_icon = "✅" if result.status == "PASS" else "❌"
            print(f"   {status_icon} {result.step}: {result.details}")
            if result.error:
                print(f"      Error: {result.error}")
        
        if self.test_data.get("performance_metrics"):
            print(f"\n⚡ PERFORMANCE METRICS:")
            for metric, value in self.test_data["performance_metrics"].items():
                if isinstance(value, float):
                    print(f"   {metric}: {value:.3f}s")
                else:
                    print(f"   {metric}: {value}")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        if failed_tests == 0:
            print("   🎉 All tests passed! Analytics system is working correctly.")
        else:
            print(f"   ⚠️  {failed_tests} test(s) failed. Review error details above.")
        
        print("="*80)

async def main():
    """Main entry point for the analytics test suite."""
    tester = BlueprintAnalyticsTester()
    await tester.run()

if __name__ == "__main__":
    asyncio.run(main())
