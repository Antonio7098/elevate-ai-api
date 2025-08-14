#!/usr/bin/env python3
"""
E2E Test Suite: Core API ↔ AI API Integration
Comprehensive testing of blueprint lifecycle, content generation, and synchronization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import httpx
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Test configuration
AI_API_BASE_URL = "http://localhost:8000"
CORE_API_BASE_URL = "http://localhost:3000"
TEST_USER_ID = "test-e2e-user-123"
API_KEY = "test-token"

@dataclass
class TestResult:
    """Result of an E2E test"""
    test_name: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class CoreAPIAIAPIIntegrationTester:
    """E2E tester for Core API ↔ AI API integration"""
    
    def __init__(self):
        self.ai_client = httpx.AsyncClient(
            timeout=60.0,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        self.core_client = httpx.AsyncClient(
            timeout=60.0,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        self.test_results: List[TestResult] = []
    
    async def validate_environment(self) -> bool:
        """Validate that both APIs are accessible"""
        print("🔧 Validating Environment...")
        
        try:
            # Check AI API health
            ai_health = await self.ai_client.get(f"{AI_API_BASE_URL}/health")
            if ai_health.status_code != 200:
                print(f"❌ AI API health check failed: {ai_health.status_code}")
                return False
            
            # Check Core API health
            core_health = await self.core_client.get(f"{CORE_API_BASE_URL}/health")
            if core_health.status_code != 200:
                print(f"❌ Core API health check failed: {core_health.status_code}")
                return False
            
            print("✅ Environment validation successful")
            return True
            
        except Exception as e:
            print(f"❌ Environment validation error: {e}")
            return False
    
    async def test_blueprint_lifecycle(self) -> TestResult:
        """Test complete blueprint lifecycle: create → index → update → delete"""
        print("\n🧪 Testing Blueprint Lifecycle...")
        
        start_time = time.time()
        
        try:
            # Create blueprint in Core API
            blueprint_data = {
                "title": "E2E Test Blueprint",
                "description": "Blueprint for E2E integration testing",
                "user_id": TEST_USER_ID,
                "difficulty": "BEGINNER",
                "tags": ["e2e-test", "integration"]
            }
            
            create_response = await self.core_client.post(
                f"{CORE_API_BASE_URL}/api/blueprints",
                json=blueprint_data
            )
            
            if create_response.status_code != 201:
                raise Exception(f"Blueprint creation failed: {create_response.status_code}")
            
            blueprint = create_response.json()
            blueprint_id = blueprint["id"]
            
            # Trigger AI API indexing
            index_response = await self.ai_client.post(
                f"{AI_API_BASE_URL}/api/v1/blueprint/lifecycle/index",
                json={
                    "blueprint_id": blueprint_id,
                    "user_id": TEST_USER_ID,
                    "force_reindex": True
                }
            )
            
            if index_response.status_code != 200:
                raise Exception(f"Indexing failed: {index_response.status_code}")
            
            # Cleanup
            await self.core_client.delete(f"{CORE_API_BASE_URL}/api/blueprints/{blueprint_id}")
            
            end_time = time.time()
            duration = end_time - start_time
            
            return TestResult(
                test_name="blueprint_lifecycle",
                success=True,
                duration=duration,
                details={"blueprint_id": blueprint_id}
            )
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            return TestResult(
                test_name="blueprint_lifecycle",
                success=False,
                duration=duration,
                error_message=str(e)
            )
    
    async def run_all_e2e_tests(self) -> List[TestResult]:
        """Run all E2E tests"""
        print("🚀 Starting Core API ↔ AI API Integration E2E Tests")
        print("=" * 70)
        
        # Validate environment first
        if not await self.validate_environment():
            print("❌ Environment validation failed. Aborting tests.")
            return []
        
        # Run all tests
        tests = [
            self.test_blueprint_lifecycle()
        ]
        
        for test in tests:
            result = await test
            self.test_results.append(result)
            
            if result.success:
                print(f"✅ {result.test_name}: PASSED ({result.duration:.2f}s)")
            else:
                print(f"❌ {result.test_name}: FAILED ({result.duration:.2f}s) - {result.error_message}")
        
        return self.test_results
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n📊 E2E TEST SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.success)
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 All E2E tests passed!")
        else:
            print("\n❌ Some E2E tests failed.")

async def main():
    """Main E2E testing function"""
    print("🚀 Core API ↔ AI API Integration E2E Testing")
    print("=" * 70)
    
    # Initialize tester
    tester = CoreAPIAIAPIIntegrationTester()
    
    try:
        # Run all E2E tests
        results = await tester.run_all_e2e_tests()
        
        # Print comprehensive summary
        tester.print_test_summary()
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"e2e_integration_results_{timestamp}.json"
        
        # Convert results to serializable format
        results_data = []
        for result in results:
            results_data.append({
                "test_name": result.test_name,
                "success": result.success,
                "duration": result.duration,
                "error_message": result.error_message,
                "details": result.details
            })
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 E2E test results saved to: {results_file}")
        
        # Return appropriate exit code
        if all(result.success for result in results):
            return 0
        else:
            return 1
            
    except Exception as e:
        print(f"❌ E2E testing failed: {e}")
        return 2

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
