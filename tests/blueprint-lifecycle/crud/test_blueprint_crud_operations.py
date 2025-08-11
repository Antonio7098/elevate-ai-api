#!/usr/bin/env python3
"""
Comprehensive Blueprint CRUD Operations E2E Test
Tests all CRUD operations for blueprints including:
- Create multiple blueprints
- Read/retrieve blueprints
- Update blueprint content and metadata
- Delete blueprints
- List and search operations
- Bulk operations
- Versioning and history
"""

import asyncio
import httpx
import json
import time
from datetime import datetime
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

class BlueprintCRUDTester:
    def __init__(self):
        self.results: List[TestResult] = []
        self.test_data = {
            "blueprint_ids": [],
            "test_blueprints": []
        }
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0)
        )
        
    async def run(self) -> None:
        """Run the complete blueprint CRUD test suite."""
        print("🚀 Starting Comprehensive Blueprint CRUD Operations E2E Test\n")
        
        try:
            await self.run_step(self.test_environment_setup)
            await self.run_step(self.test_create_operations)
            await self.run_step(self.test_read_operations)
            await self.run_step(self.test_update_operations)
            await self.run_step(self.test_search_and_list_operations)
            await self.run_step(self.test_bulk_operations)
            await self.run_step(self.test_versioning_operations)
            await self.run_step(self.test_delete_operations)
            await self.run_step(self.test_performance_metrics)
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
        """Test 1: Verify environment is ready for CRUD operations."""
        print("🔧 Step 1: Environment Setup and Validation...")
        
        # Check AI API health
        response = await self.client.get(f"{AI_API_BASE_URL}/api/health")
        if response.status_code != 200:
            raise Exception(f"AI API health check failed: {response.status_code}")
        
        # Check Core API health
        response = await self.client.get(f"{CORE_API_BASE_URL}/health")
        if response.status_code != 200:
            raise Exception(f"Core API health check failed: {response.status_code}")
        
        # Check blueprint CRUD endpoints
        response = await self.client.get(
            f"{AI_API_BASE_URL}/api/v1/blueprints/health",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        if response.status_code != 200:
            raise Exception(f"Blueprint API health check failed: {response.status_code}")
        
        print("   ✅ Environment validation successful")

    async def test_create_operations(self) -> None:
        """Test 2: Test creating multiple blueprints with different content."""
        print("📝 Step 2: Testing Blueprint Creation Operations...")
        
        # Test blueprint templates
        blueprint_templates = [
            {
                "name": "Python Programming Basics",
                "content": """
                # Python Programming Fundamentals
                
                ## Variables and Data Types
                - Integers, floats, strings, booleans
                - Lists, tuples, dictionaries, sets
                
                ## Control Flow
                - If statements and loops
                - Functions and scope
                
                ## Object-Oriented Programming
                - Classes and objects
                - Inheritance and polymorphism
                """,
                "difficulty_level": "beginner",
                "estimated_duration": "4 weeks"
            },
            {
                "name": "Advanced Data Structures",
                "content": """
                # Advanced Data Structures and Algorithms
                
                ## Tree Structures
                - Binary trees, AVL trees, Red-black trees
                - Tree traversal algorithms
                
                ## Graph Algorithms
                - Depth-first search, breadth-first search
                - Shortest path algorithms
                
                ## Advanced Algorithms
                - Dynamic programming
                - Greedy algorithms
                """,
                "difficulty_level": "advanced",
                "estimated_duration": "6 weeks"
            },
            {
                "name": "Machine Learning Fundamentals",
                "content": """
                # Machine Learning Basics
                
                ## Supervised Learning
                - Linear regression, logistic regression
                - Decision trees, random forests
                
                ## Unsupervised Learning
                - Clustering algorithms
                - Dimensionality reduction
                
                ## Model Evaluation
                - Cross-validation
                - Performance metrics
                """,
                "difficulty_level": "intermediate",
                "estimated_duration": "8 weeks"
            }
        ]
        
        for i, template in enumerate(blueprint_templates):
            print(f"   Creating blueprint {i+1}: {template['name']}")
            
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/generate-notes-from-content",
                json={
                    "user_content": template["content"],
                    "content_format": "markdown",
                    "create_blueprint": True
                },
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            if response.status_code != 200:
                raise Exception(f"Blueprint creation failed for {template['name']}: {response.status_code}")
            
            blueprint_result = response.json()
            if not blueprint_result.get("success"):
                raise Exception(f"Blueprint creation failed for {template['name']}: {blueprint_result.get('message', 'Unknown error')}")
            
            blueprint_id = blueprint_result.get("blueprint_id")
            if not blueprint_id:
                raise Exception(f"Blueprint creation failed for {template['name']}: No blueprint_id returned")
            
            # Store blueprint data
            self.test_data["blueprint_ids"].append(blueprint_id)
            self.test_data["test_blueprints"].append({
                "id": blueprint_id,
                "name": template["name"],
                "difficulty": template["difficulty_level"],
                "template": template
            })
            
            print(f"   ✅ Created blueprint: {blueprint_id}")
        
        print(f"   ✅ Successfully created {len(blueprint_templates)} blueprints")

    async def test_read_operations(self) -> None:
        """Test 3: Test reading and retrieving blueprints."""
        print("📖 Step 3: Testing Blueprint Read Operations...")
        
        if not self.test_data["blueprint_ids"]:
            raise Exception("No blueprints available for read testing")
        
        for blueprint_info in self.test_data["test_blueprints"]:
            blueprint_id = blueprint_info["id"]
            expected_name = blueprint_info["name"]
            
            print(f"   Reading blueprint: {blueprint_id}")
            
            # Test individual blueprint retrieval
            response = await self.client.get(
                f"{AI_API_BASE_URL}/api/v1/blueprints/{blueprint_id}",
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            if response.status_code != 200:
                raise Exception(f"Blueprint retrieval failed for {blueprint_id}: {response.status_code}")
            
            blueprint = response.json()
            
            # Validate retrieved data
            if blueprint["name"] != expected_name:
                raise Exception(f"Blueprint name mismatch: expected {expected_name}, got {blueprint['name']}")
            
            required_fields = ["blueprint_id", "name", "description", "concepts", "created_at", "updated_at"]
            for field in required_fields:
                if field not in blueprint:
                    raise Exception(f"Missing required field in retrieved blueprint: {field}")
            
            print(f"   ✅ Successfully retrieved blueprint: {blueprint_id}")
        
        # Test bulk retrieval
        print("   Testing bulk blueprint retrieval...")
        response = await self.client.get(
            f"{AI_API_BASE_URL}/api/v1/blueprints/",
            params={"user_id": TEST_USER_ID, "limit": 10},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Bulk blueprint retrieval failed: {response.status_code}")
        
        blueprints_list = response.json()
        if "blueprints" not in blueprints_list:
            raise Exception("Bulk retrieval response missing 'blueprints' field")
        
        print(f"   ✅ Bulk retrieval successful: {len(blueprints_list['blueprints'])} blueprints")

    async def test_update_operations(self) -> None:
        """Test 4: Test updating blueprint content and metadata."""
        print("✏️ Step 4: Testing Blueprint Update Operations...")
        
        if not self.test_data["blueprint_ids"]:
            raise Exception("No blueprints available for update testing")
        
        # Test updating the first blueprint
        blueprint_id = self.test_data["blueprint_ids"][0]
        original_name = self.test_data["test_blueprints"][0]["name"]
        
        print(f"   Updating blueprint: {blueprint_id}")
        
        # Update blueprint metadata
        update_data = {
            "name": f"{original_name} - Updated",
            "description": "This blueprint has been updated during testing",
            "tags": ["test", "e2e", "updated", "beginner"],
            "difficulty_level": "intermediate",
            "estimated_duration": "5 weeks"
        }
        
        response = await self.client.put(
            f"{AI_API_BASE_URL}/api/v1/blueprints/{blueprint_id}",
            json=update_data,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Blueprint update failed: {response.status_code}")
        
        update_result = response.json()
        
        # Verify update was successful
        if update_result["name"] != update_data["name"]:
            raise Exception(f"Update verification failed: name not updated")
        
        # Test content update
        new_content = """
        # Updated Python Programming Fundamentals
        
        ## Enhanced Content
        - Updated variables and data types
        - Advanced control flow examples
        - Object-oriented programming patterns
        
        ## New Sections
        - Error handling and exceptions
        - File I/O operations
        - Module and package management
        """
        
        content_update_response = await self.client.put(
            f"{AI_API_BASE_URL}/api/v1/blueprints/{blueprint_id}/content",
            json={"content": new_content},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if content_update_response.status_code != 200:
            raise Exception(f"Content update failed: {content_update_response.status_code}")
        
        # Verify content was updated
        updated_blueprint = await self.client.get(
            f"{AI_API_BASE_URL}/api/v1/blueprints/{blueprint_id}",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if updated_blueprint.status_code != 200:
            raise Exception(f"Failed to retrieve updated blueprint: {updated_blueprint.status_code}")
        
        updated_data = updated_blueprint.json()
        if "content" not in updated_data or len(updated_data["content"]) < 100:
            raise Exception("Content update verification failed")
        
        print(f"   ✅ Successfully updated blueprint: {blueprint_id}")

    async def test_search_and_list_operations(self) -> None:
        """Test 5: Test search, filtering, and listing operations."""
        print("🔍 Step 5: Testing Search and List Operations...")
        
        # Test filtering by difficulty level
        print("   Testing difficulty level filtering...")
        response = await self.client.get(
            f"{AI_API_BASE_URL}/api/v1/blueprints/",
            params={"user_id": TEST_USER_ID, "difficulty_level": "beginner"},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Difficulty filtering failed: {response.status_code}")
        
        filtered_blueprints = response.json()
        beginner_blueprints = [bp for bp in filtered_blueprints.get("blueprints", []) 
                             if bp.get("difficulty_level") == "beginner"]
        
        if len(beginner_blueprints) == 0:
            raise Exception("No beginner blueprints found after filtering")
        
        print(f"   ✅ Difficulty filtering successful: {len(beginner_blueprints)} beginner blueprints")
        
        # Test search by name
        print("   Testing name search...")
        search_term = "Python"
        response = await self.client.get(
            f"{AI_API_BASE_URL}/api/v1/blueprints/search",
            params={"user_id": TEST_USER_ID, "query": search_term},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Name search failed: {response.status_code}")
        
        search_results = response.json()
        if "results" not in search_results:
            raise Exception("Search response missing 'results' field")
        
        python_blueprints = [bp for bp in search_results["results"] 
                           if "Python" in bp.get("name", "")]
        
        if len(python_blueprints) == 0:
            raise Exception("No Python blueprints found in search results")
        
        print(f"   ✅ Name search successful: {len(python_blueprints)} Python blueprints found")
        
        # Test pagination
        print("   Testing pagination...")
        response = await self.client.get(
            f"{AI_API_BASE_URL}/api/v1/blueprints/",
            params={"user_id": TEST_USER_ID, "limit": 2, "offset": 0},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Pagination failed: {response.status_code}")
        
        paginated_results = response.json()
        if len(paginated_results.get("blueprints", [])) > 2:
            raise Exception("Pagination limit not enforced")
        
        print("   ✅ Pagination testing successful")

    async def test_bulk_operations(self) -> None:
        """Test 6: Test bulk operations on multiple blueprints."""
        print("📦 Step 6: Testing Bulk Operations...")
        
        if len(self.test_data["blueprint_ids"]) < 2:
            raise Exception("Need at least 2 blueprints for bulk operations testing")
        
        # Test bulk status update
        print("   Testing bulk status update...")
        blueprint_ids = self.test_data["blueprint_ids"][:2]
        
        bulk_update_data = {
            "blueprint_ids": blueprint_ids,
            "updates": {
                "status": "active",
                "tags": ["test", "e2e", "bulk-updated"]
            }
        }
        
        response = await self.client.put(
            f"{AI_API_BASE_URL}/api/v1/blueprints/bulk-update",
            json=bulk_update_data,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Bulk update failed: {response.status_code}")
        
        bulk_result = response.json()
        if "updated_count" not in bulk_result:
            raise Exception("Bulk update response missing 'updated_count' field")
        
        if bulk_result["updated_count"] != len(blueprint_ids):
            raise Exception(f"Bulk update count mismatch: expected {len(blueprint_ids)}, got {bulk_result['updated_count']}")
        
        print(f"   ✅ Bulk update successful: {bulk_result['updated_count']} blueprints updated")
        
        # Test bulk retrieval
        print("   Testing bulk retrieval...")
        bulk_retrieve_data = {
            "blueprint_ids": blueprint_ids
        }
        
        response = await self.client.post(
            f"{AI_API_BASE_URL}/api/v1/blueprints/bulk-retrieve",
            json=bulk_retrieve_data,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Bulk retrieval failed: {response.status_code}")
        
        bulk_retrieve_result = response.json()
        if "blueprints" not in bulk_retrieve_result:
            raise Exception("Bulk retrieval response missing 'blueprints' field")
        
        if len(bulk_retrieve_result["blueprints"]) != len(blueprint_ids):
            raise Exception(f"Bulk retrieval count mismatch: expected {len(blueprint_ids)}, got {len(bulk_retrieve_result['blueprints'])}")
        
        print(f"   ✅ Bulk retrieval successful: {len(bulk_retrieve_result['blueprints'])} blueprints retrieved")

    async def test_versioning_operations(self) -> None:
        """Test 7: Test blueprint versioning and history."""
        print("📚 Step 7: Testing Versioning Operations...")
        
        if not self.test_data["blueprint_ids"]:
            raise Exception("No blueprints available for versioning testing")
        
        blueprint_id = self.test_data["blueprint_ids"][0]
        
        print(f"   Testing versioning for blueprint: {blueprint_id}")
        
        # Create a new version
        new_version_content = """
        # Version 2.0 - Enhanced Python Programming
        
        ## New Features
        - Advanced error handling
        - Decorators and context managers
        - Async programming basics
        
        ## Updated Content
        - Modern Python best practices
        - Type hints and annotations
        - Performance optimization tips
        """
        
        version_response = await self.client.post(
            f"{AI_API_BASE_URL}/api/v1/blueprints/{blueprint_id}/versions",
            json={
                "content": new_version_content,
                "version_notes": "Enhanced content with new features",
                "major_version": True
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if version_response.status_code != 200:
            raise Exception(f"Version creation failed: {version_response.status_code}")
        
        version_result = version_response.json()
        if "version_id" not in version_result:
            raise Exception("Version creation response missing 'version_id' field")
        
        print(f"   ✅ Version created: {version_result['version_id']}")
        
        # Test version history
        history_response = await self.client.get(
            f"{AI_API_BASE_URL}/api/v1/blueprints/{blueprint_id}/versions",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if history_response.status_code != 200:
            raise Exception(f"Version history retrieval failed: {history_response.status_code}")
        
        history_result = history_response.json()
        if "versions" not in history_result:
            raise Exception("Version history response missing 'versions' field")
        
        if len(history_result["versions"]) < 2:  # Original + new version
            raise Exception("Version history incomplete")
        
        print(f"   ✅ Version history retrieved: {len(history_result['versions'])} versions")

    async def test_delete_operations(self) -> None:
        """Test 8: Test blueprint deletion operations."""
        print("🗑️ Step 8: Testing Delete Operations...")
        
        if not self.test_data["blueprint_ids"]:
            raise Exception("No blueprints available for deletion testing")
        
        # Test soft delete (mark as deleted)
        blueprint_id = self.test_data["blueprint_ids"][0]
        
        print(f"   Testing soft delete for blueprint: {blueprint_id}")
        
        delete_response = await self.client.delete(
            f"{AI_API_BASE_URL}/api/v1/blueprints/{blueprint_id}",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if delete_response.status_code != 200:
            raise Exception(f"Soft delete failed: {delete_response.status_code}")
        
        delete_result = delete_response.json()
        if "deleted" not in delete_result or not delete_result["deleted"]:
            raise Exception("Soft delete verification failed")
        
        print(f"   ✅ Soft delete successful: {blueprint_id}")
        
        # Verify blueprint is marked as deleted
        verify_response = await self.client.get(
            f"{AI_API_BASE_URL}/api/v1/blueprints/{blueprint_id}",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if verify_response.status_code != 200:
            raise Exception(f"Failed to retrieve deleted blueprint: {verify_response.status_code}")
        
        deleted_blueprint = verify_response.json()
        if "deleted_at" not in deleted_blueprint:
            raise Exception("Blueprint not properly marked as deleted")
        
        print(f"   ✅ Deletion verification successful")
        
        # Test hard delete (permanent removal)
        print("   Testing hard delete...")
        hard_delete_response = await self.client.delete(
            f"{AI_API_BASE_URL}/api/v1/blueprints/{blueprint_id}/permanent",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if hard_delete_response.status_code != 200:
            raise Exception(f"Hard delete failed: {hard_delete_response.status_code}")
        
        # Verify blueprint is completely removed
        final_verify_response = await self.client.get(
            f"{AI_API_BASE_URL}/api/v1/blueprints/{blueprint_id}",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if final_verify_response.status_code != 404:
            raise Exception("Blueprint not completely removed after hard delete")
        
        print(f"   ✅ Hard delete successful: {blueprint_id} completely removed")
        
        # Remove from test data
        self.test_data["blueprint_ids"].remove(blueprint_id)
        self.test_data["test_blueprints"] = [bp for bp in self.test_data["test_blueprints"] 
                                           if bp["id"] != blueprint_id]

    async def test_performance_metrics(self) -> None:
        """Test 9: Test performance metrics for CRUD operations."""
        print("⏱️ Step 9: Testing Performance Metrics...")
        
        # Test CRUD operation performance
        performance_metrics = {}
        
        # Create performance test
        start_time = time.time()
        response = await self.client.post(
            f"{AI_API_BASE_URL}/api/v1/blueprints/generate",
            json={
                "content": "# Performance Test\nSimple content for performance testing.",
                "user_id": TEST_USER_ID,
                "blueprint_options": {"name": "Performance Test Blueprint"}
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        create_time = time.time() - start_time
        
        if response.status_code != 200:
            raise Exception(f"Performance test creation failed: {response.status_code}")
        
        performance_metrics["create_time"] = create_time
        
        # Read performance test
        blueprint_id = response.json()["blueprint_id"]
        start_time = time.time()
        read_response = await self.client.get(
            f"{AI_API_BASE_URL}/api/v1/blueprints/{blueprint_id}",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        read_time = time.time() - start_time
        
        if read_response.status_code != 200:
            raise Exception(f"Performance test read failed: {read_response.status_code}")
        
        performance_metrics["read_time"] = read_time
        
        # Update performance test
        start_time = time.time()
        update_response = await self.client.put(
            f"{AI_API_BASE_URL}/api/v1/blueprints/{blueprint_id}",
            json={"name": "Updated Performance Test"},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        update_time = time.time() - start_time
        
        if update_response.status_code != 200:
            raise Exception(f"Performance test update failed: {update_response.status_code}")
        
        performance_metrics["update_time"] = update_time
        
        # Delete performance test
        start_time = time.time()
        delete_response = await self.client.delete(
            f"{AI_API_BASE_URL}/api/v1/blueprints/{blueprint_id}/permanent",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        delete_time = time.time() - start_time
        
        if delete_response.status_code != 200:
            raise Exception(f"Performance test deletion failed: {delete_response.status_code}")
        
        performance_metrics["delete_time"] = delete_time
        
        # Performance thresholds
        thresholds = {
            "create_time": 30.0,  # 30 seconds max
            "read_time": 2.0,     # 2 seconds max
            "update_time": 5.0,   # 5 seconds max
            "delete_time": 3.0    # 3 seconds max
        }
        
        for operation, threshold in thresholds.items():
            if performance_metrics[operation] > threshold:
                raise Exception(f"{operation} too slow: {performance_metrics[operation]:.2f}s > {threshold}s")
        
        print(f"   ✅ Performance tests passed:")
        for operation, time_taken in performance_metrics.items():
            print(f"      {operation}: {time_taken:.2f}s")
        
        # Store performance metrics
        self.test_data["performance_metrics"] = performance_metrics

    def print_results(self) -> None:
        """Print comprehensive test results."""
        print("\n" + "="*60)
        print("📊 BLUEPRINT CRUD OPERATIONS E2E TEST RESULTS")
        print("="*60)
        
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == "PASS"])
        failed_tests = len([r for r in self.results if r.status == "FAIL"])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if "performance_metrics" in self.test_data:
            metrics = self.test_data["performance_metrics"]
            print(f"\nPerformance Metrics:")
            for operation, time_taken in metrics.items():
                print(f"  {operation.replace('_', ' ').title()}: {time_taken:.2f}s")
        
        print(f"\nTest Data Summary:")
        print(f"  Blueprints Created: {len(self.test_data['blueprint_ids'])}")
        print(f"  Test Blueprints: {len(self.test_data['test_blueprints'])}")
        
        print(f"\nDetailed Results:")
        for result in self.results:
            status_icon = "✅" if result.status == "PASS" else "❌"
            print(f"  {status_icon} {result.step}: {result.details}")
            if result.error:
                print(f"    Error: {result.error}")
        
        if failed_tests == 0:
            print(f"\n🎉 All tests passed! Blueprint CRUD operations are working correctly.")
        else:
            print(f"\n⚠️ {failed_tests} tests failed. Please review the errors above.")

async def main():
    """Main test execution function."""
    tester = BlueprintCRUDTester()
    await tester.run()

if __name__ == "__main__":
    asyncio.run(main())
