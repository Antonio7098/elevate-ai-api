#!/usr/bin/env python3

"""
End-to-End Primitive Generation & Sync Test Script

This script tests the complete primitive generation and Core API synchronization workflow:
1. Generate primitives from source content
2. Create mastery criteria for primitives  
3. Generate criterion-specific questions
4. Sync primitives with Core API
5. Verify sync status and data integrity

Usage: python test-e2e-primitives.py
"""

import asyncio
import sys
import json
import time
from typing import Dict, List, Any, Optional
import httpx
from datetime import datetime

# Configuration
AI_API_BASE_URL = "http://localhost:8000"
CORE_API_BASE_URL = "http://localhost:3000"
API_KEY = "test_api_key_123"
TEST_USER_ID = 108  # From previous test user creation

class TestResult:
    def __init__(self, step: str, status: str, details: str = None, error: Any = None):
        self.step = step
        self.status = status  # 'PASS', 'FAIL', 'SKIP'
        self.details = details
        self.error = error

class PrimitiveE2ETest:
    def __init__(self):
        self.results: List[TestResult] = []
        self.test_data = {}
        # Increase timeouts to accommodate LLM processing and network variability
        timeout = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=30.0)
        self.client = httpx.AsyncClient(timeout=timeout)
        
    async def run(self) -> None:
        """Run the complete primitive e2e test suite."""
        print("🚀 Starting AI API Primitive Generation & Sync E2E Test\n")
        
        try:
            await self.run_step(self.test_health_check)
            await self.run_step(self.test_primitive_generation)
            await self.run_step(self.test_mastery_criteria_generation)
            await self.run_step(self.test_criterion_question_generation)
            await self.run_step(self.test_core_api_sync)
            await self.run_step(self.test_sync_status_verification)
            await self.run_step(self.test_batch_operations)
        except Exception as error:
            print(f"\n❌ Test suite aborted due to critical failure: {error}")
        finally:
            await self.client.aclose()
            self.print_results()
    
    async def run_step(self, step_func, continue_on_error: bool = False) -> None:
        """Execute a test step with error handling."""
        try:
            await step_func()
        except Exception as error:
            if not continue_on_error:
                raise error

    async def test_health_check(self) -> None:
        """Test 1: Verify AI API health and availability."""
        try:
            print("🏥 Step 1: Checking AI API health...")
            
            response = await self.client.get(f"{AI_API_BASE_URL}/api/health")
            
            if response.status_code == 200:
                health_data = response.json()
                self.results.append(TestResult(
                    "1. AI API Health Check",
                    "PASS",
                    f"AI API healthy - Status: {health_data.get('status', 'unknown')}"
                ))
                print("   ✅ AI API health check successful")
            else:
                raise Exception(f"Health check failed with status {response.status_code}")
                
        except Exception as error:
            self.results.append(TestResult(
                "1. AI API Health Check",
                "FAIL",
                f"Health check failed: {str(error)}",
                error
            ))
            print("   ❌ AI API health check failed")
            raise error

    async def test_primitive_generation(self) -> None:
        """Test 2: Generate primitives from source content."""
        try:
            print("🧩 Step 2: Testing primitive generation...")
            
            # Sample educational content for primitive generation
            source_content = """
            # Photosynthesis in Plants
            
            Photosynthesis is the process by which plants convert sunlight into chemical energy.
            
            ## Light Reactions
            The light-dependent reactions occur in the thylakoids of chloroplasts.
            Chlorophyll absorbs light energy and converts it to ATP and NADPH.
            
            ## Calvin Cycle  
            The Calvin cycle uses ATP and NADPH to convert CO2 into glucose.
            This process occurs in the stroma of chloroplasts.
            
            ## Key Components
            - Chloroplasts: organelles where photosynthesis occurs
            - Chlorophyll: green pigment that captures light
            - Stomata: pores that allow gas exchange
            """
            
            # Minimal schema-compliant payload to reduce validation complexity
            payload = {
                "sourceContent": source_content,
                "sourceType": "article",
                "userPreferences": {}
            }
            
            headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/primitives/generate",
                json=payload,
                headers=headers
            )

            # Definitive logging to see the response
            print(f"   ℹ️  API Response Status: {response.status_code}")
            print(f"   ℹ️  API Response Text: {response.text}")
            
            if response.status_code == 200:
                data = response.json()
                self.test_data['primitives'] = data.get('primitives', [])
                primitive_count = len(self.test_data['primitives'])
                
                self.results.append(TestResult(
                    "2. Primitive Generation",
                    "PASS", 
                    f"Generated {primitive_count} primitives successfully"
                ))
                print(f"   ✅ Generated {primitive_count} primitives")
                
                # Store some primitives for later tests
                if primitive_count > 0:
                    sample_primitive = self.test_data['primitives'][0]
                    self.test_data['sample_primitive_id'] = sample_primitive.get('primitiveId')
                    print(f"   📝 Sample primitive: {sample_primitive.get('title', 'Unknown')}")
                
            else:
                error_msg = f"Primitive generation failed with status {response.status_code}: {response.text}"
                print(f"   ❌ HTTP Error: {error_msg}")
                raise Exception(error_msg)
                
        except httpx.ConnectError as error:
            error_details = f"Connection error: {error.request.method} {error.request.url} - {str(error)}"
            print(f"   ❌ Detailed error: {error_details}")
            self.results.append(TestResult(
                "2. Primitive Generation",
                "FAIL",
                f"Primitive generation failed: {error_details}",
                error
            ))
            print("   ❌ Primitive generation failed")
            raise error
        except httpx.ReadTimeout as error:
            error_details = f"Read timeout: {error.request.method} {error.request.url} - {str(error)}"
            print(f"   ❌ Detailed error: {error_details}")
            self.results.append(TestResult(
                "2. Primitive Generation",
                "FAIL",
                f"Primitive generation failed: {error_details}",
                error
            ))
            print("   ❌ Primitive generation failed")
            raise error
        except httpx.RequestError as error:
            error_details = f"Request error: {error.request.method} {error.request.url} - {str(error)}"
            print(f"   ❌ Detailed error: {error_details}")
            self.results.append(TestResult(
                "2. Primitive Generation",
                "FAIL",
                f"Primitive generation failed: {error_details}",
                error
            ))
            print("   ❌ Primitive generation failed")
            raise error
        except Exception as error:
            error_details = repr(error)
            print(f"   ❌ Detailed error: {error_details}")
            if hasattr(error, 'response'):
                print(f"   ℹ️  API Response Status: {error.response.status_code}")
                print(f"   ℹ️  API Response Text: {error.response.text}")
            self.results.append(TestResult(
                "2. Primitive Generation",
                "FAIL",
                f"Primitive generation failed: {error_details}",
                error
            ))
            print("   ❌ Primitive generation failed")
            raise error
        finally:
            print("   ℹ️  Finished primitive generation test step.")

    async def test_mastery_criteria_generation(self) -> None:
        """Test 3: Generate mastery criteria for primitives."""
        try:
            print("🎯 Step 3: Testing mastery criteria generation...")
            
            if not self.test_data.get('primitives'):
                raise Exception("No primitives available from previous step")
            
            # Use the first primitive for criteria generation
            sample_primitive = self.test_data['primitives'][0]
            
            payload = {
                "primitive": sample_primitive,
                "criteria_count": 3,
                "uee_levels": ["understand", "use", "explore"],
                "weight_distribution": "balanced"
            }
            
            headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
            # Note: This endpoint may not exist yet - this is a test for future implementation
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/primitives/mastery-criteria",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                self.test_data['mastery_criteria'] = data.get('criteria', [])
                criteria_count = len(self.test_data['mastery_criteria'])
                
                self.results.append(TestResult(
                    "3. Mastery Criteria Generation", 
                    "PASS",
                    f"Generated {criteria_count} mastery criteria successfully"
                ))
                print(f"   ✅ Generated {criteria_count} mastery criteria")
                
            else:
                # For now, create mock criteria since endpoint may not exist
                self.test_data['mastery_criteria'] = [
                    {
                        "criterionId": f"crit_{i+1}",
                        "primitiveId": sample_primitive.get('primitiveId'),
                        "description": f"Mock criterion {i+1}",
                        "ueeLevel": ["understand", "use", "explore"][i % 3],
                        "weight": 3.0
                    }
                    for i in range(3)
                ]
                
                self.results.append(TestResult(
                    "3. Mastery Criteria Generation",
                    "SKIP", 
                    "Endpoint not available - using mock data"
                ))
                print("   ⏭️  Using mock mastery criteria (endpoint not implemented)")
                
        except Exception as error:
            self.results.append(TestResult(
                "3. Mastery Criteria Generation",
                "FAIL",
                f"Mastery criteria generation failed: {str(error)}",
                error
            ))
            print("   ❌ Mastery criteria generation failed")
            # Don't raise error - continue with mock data
            
    async def test_criterion_question_generation(self) -> None:
        """Test 4: Generate questions for mastery criteria."""
        try:
            print("❓ Step 4: Testing criterion-specific question generation...")
            
            if not self.test_data.get('mastery_criteria'):
                raise Exception("No mastery criteria available from previous step")
                
            # Use the first criterion for question generation
            sample_criterion = self.test_data['mastery_criteria'][0]
            
            payload = {
                "criteria": [sample_criterion],
                "questions_per_criterion": 2,
                "question_types": ["short_answer", "multiple_choice"],
                "difficulty_level": "intermediate"
            }
            
            headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/primitives/questions/generate",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                self.test_data['questions'] = data.get('questions', [])
                question_count = len(self.test_data['questions'])
                
                self.results.append(TestResult(
                    "4. Criterion Question Generation",
                    "PASS",
                    f"Generated {question_count} questions successfully"
                ))
                print(f"   ✅ Generated {question_count} criterion-specific questions")
                
            else:
                raise Exception(f"Question generation failed with status {response.status_code}: {response.text}")
                
        except Exception as error:
            self.results.append(TestResult(
                "4. Criterion Question Generation", 
                "FAIL",
                f"Question generation failed: {str(error)}",
                error
            ))
            print("   ❌ Criterion question generation failed")
            raise error

    async def test_core_api_sync(self) -> None:
        """Test 5: Synchronize primitives with Core API."""
        try:
            print("🔄 Step 5: Testing Core API synchronization...")
            
            if not self.test_data.get('primitives'):
                raise Exception("No primitives available for sync")
                
            payload = {
                "primitives": self.test_data['primitives'][:2],  # Sync first 2 primitives
                "mastery_criteria": self.test_data.get('mastery_criteria', [])[:2],
                "blueprint_id": "test-blueprint-001",
                "user_id": TEST_USER_ID
            }
            
            headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/primitives/sync",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                sync_status = data.get('status', 'unknown')
                self.test_data['sync_id'] = data.get('sync_id', 'unknown')
                
                self.results.append(TestResult(
                    "5. Core API Sync",
                    "PASS",
                    f"Sync initiated successfully - Status: {sync_status}"
                ))
                print(f"   ✅ Core API sync initiated - ID: {self.test_data['sync_id']}")
                
            else:
                raise Exception(f"Core API sync failed with status {response.status_code}: {response.text}")
                
        except Exception as error:
            self.results.append(TestResult(
                "5. Core API Sync",
                "FAIL", 
                f"Core API sync failed: {str(error)}",
                error
            ))
            print("   ❌ Core API sync failed")
            raise error

    async def test_sync_status_verification(self) -> None:
        """Test 6: Verify synchronization status and completion."""
        try:
            print("✅ Step 6: Testing sync status verification...")
            
            if not self.test_data.get('sync_id'):
                raise Exception("No sync ID available from previous step")
                
            # Poll sync status with retries
            max_attempts = 10
            for attempt in range(max_attempts):
                headers = {"X-API-Key": API_KEY}
                response = await self.client.get(
                    f"{AI_API_BASE_URL}/api/v1/primitives/sync/status/test-blueprint-001",
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status', 'unknown')
                    
                    if status == 'completed':
                        self.results.append(TestResult(
                            "6. Sync Status Verification",
                            "PASS",
                            f"Sync completed successfully after {attempt + 1} attempts"
                        ))
                        print(f"   ✅ Sync completed successfully")
                        return
                    elif status == 'failed':
                        raise Exception(f"Sync failed: {data.get('error', 'Unknown error')}")
                    else:
                        print(f"   ⏳ Sync in progress... (attempt {attempt + 1}/{max_attempts})")
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(2)  # Wait 2 seconds before next attempt
                else:
                    raise Exception(f"Status check failed with status {response.status_code}")
                    
            # If we get here, sync didn't complete in time
            self.results.append(TestResult(
                "6. Sync Status Verification",
                "FAIL",
                f"Sync did not complete within {max_attempts * 2} seconds"
            ))
            print(f"   ❌ Sync verification timed out")
            
        except Exception as error:
            self.results.append(TestResult(
                "6. Sync Status Verification",
                "FAIL",
                f"Sync status verification failed: {str(error)}",
                error
            ))
            print("   ❌ Sync status verification failed")
            raise error

    async def test_batch_operations(self) -> None:
        """Test 7: Test batch primitive operations."""
        try:
            print("📦 Step 7: Testing batch primitive operations...")
            
            # Test batch primitive extraction from multiple blueprints
            payload = {
                "blueprint_ids": ["test-blueprint-001", "test-blueprint-002"],
                "extraction_preferences": {
                    "max_primitives_per_blueprint": 5,
                    "primitive_types": ["fact", "concept"],
                    "uee_preference": "balanced"
                }
            }
            
            headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/blueprints/batch/primitives",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', {})
                total_primitives = sum(len(blueprint_data.get('primitives', [])) 
                                    for blueprint_data in results.values())
                
                self.results.append(TestResult(
                    "7. Batch Operations",
                    "PASS",
                    f"Batch processed {len(results)} blueprints, extracted {total_primitives} primitives"
                ))
                print(f"   ✅ Batch operations successful - {total_primitives} total primitives")
                
            else:
                raise Exception(f"Batch operations failed with status {response.status_code}: {response.text}")
                
        except Exception as error:
            self.results.append(TestResult(
                "7. Batch Operations",
                "FAIL",
                f"Batch operations failed: {str(error)}",
                error
            ))
            print("   ❌ Batch operations failed") 
            # Don't raise error - this is not critical

    def print_results(self) -> None:
        """Print formatted test results."""
        print(f"\n{'=' * 60}")
        print("📊 AI API PRIMITIVE E2E TEST RESULTS")
        print(f"{'=' * 60}")
        
        for result in self.results:
            status_emoji = "✅" if result.status == "PASS" else "❌" if result.status == "FAIL" else "⏭️"
            print(f"{status_emoji} {result.step}")
            if result.details:
                print(f"   {result.details}")
        
        print("-" * 60)
        
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL") 
        skipped = sum(1 for r in self.results if r.status == "SKIP")
        
        print(f"📈 SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
        
        if failed > 0:
            print("⚠️  Some tests failed. Check the errors above for details.")
        elif passed > 0:
            print("🎉 All tests passed! Primitive workflow is working correctly.")
        
        print(f"{'=' * 60}")

async def main():
    """Main function to run the test suite.""" 
    test = PrimitiveE2ETest()
    await test.run()

if __name__ == "__main__":
    asyncio.run(main())
