#!/usr/bin/env python3
"""
Comprehensive Blueprint Creation E2E Test
Tests the complete blueprint creation workflow including:
- Source content ingestion
- Content parsing and validation
- Blueprint generation
- Quality checks
- Error handling
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
class BlueprintTestResult:
    step: str
    status: str  # PASS, FAIL, SKIP
    details: str = None
    error: Any = None
    duration: float = 0.0
    metadata: Dict[str, Any] = None

class BlueprintCreationTester:
    def __init__(self):
        self.results: List[BlueprintTestResult] = []
        self.test_data = {}
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0)
        )
        # Artifact tracking for monitoring: save intermediate responses per run
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.artifacts_dir = Path("artifacts") / self.run_id
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
    async def run(self) -> None:
        """Run the complete blueprint creation test suite."""
        print("🚀 Starting Comprehensive Blueprint Creation E2E Test\n")
        
        try:
            await self.run_step(self.test_environment_setup)
            await self.run_step(self.test_source_content_ingestion)
            await self.run_step(self.test_content_parsing)
            await self.run_step(self.test_blueprint_generation)
            await self.run_step(self.test_blueprint_validation)
            await self.run_step(self.test_error_handling)
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
            self.results.append(BlueprintTestResult(
                step_func.__name__.replace('test_', '').replace('_', ' ').title(),
                "PASS",
                f"Completed successfully in {duration:.2f}s",
                duration=duration
            ))
        except Exception as error:
            duration = time.time() - start_time
            self.results.append(BlueprintTestResult(
                step_func.__name__.replace('test_', '').replace('_', ' ').title(),
                "FAIL",
                f"Failed after {duration:.2f}s: {str(error)}",
                error,
                duration=duration
            ))
            if not continue_on_error:
                raise error

    async def test_environment_setup(self) -> None:
        """Test 1: Verify environment is ready for blueprint creation."""
        print("🔧 Step 1: Environment Setup and Validation...")
        
        # Check AI API health
        response = await self.client.get(f"{AI_API_BASE_URL}/api/health")
        if response.status_code != 200:
            raise Exception(f"AI API health check failed: {response.status_code}")
        
        # Check Core API health
        response = await self.client.get(f"{CORE_API_BASE_URL}/health")
        if response.status_code != 200:
            raise Exception(f"Core API health check failed: {response.status_code}")
        
        # Check blueprint endpoints
        response = await self.client.get(
            f"{AI_API_BASE_URL}/api/v1/blueprints/health",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        if response.status_code != 200:
            raise Exception(f"Blueprint API health check failed: {response.status_code}")
        
        print("   ✅ Environment validation successful")

    async def test_source_content_ingestion(self) -> None:
        """Test 2: Test ingestion of various source content types."""
        print("📥 Step 2: Testing Source Content Ingestion...")
        
        # Test different content types
        content_types = [
            {
                "name": "Educational Text",
                "content": """
                # Photosynthesis in Plants
                
                Photosynthesis is the process by which plants convert sunlight into chemical energy.
                
                ## Light Reactions
                The light-dependent reactions occur in the thylakoids of chloroplasts.
                Chlorophyll absorbs light energy and converts it to ATP and NADPH.
                
                ## Calvin Cycle  
                The Calvin cycle uses ATP and NADPH to convert CO2 into glucose.
                This process occurs in the stroma of chloroplasts.
                """,
                "type": "text"
            },
            {
                "name": "Structured Data",
                "content": """
                {
                    "topic": "Machine Learning Basics",
                    "concepts": [
                        {"name": "Supervised Learning", "description": "Learning from labeled examples"},
                        {"name": "Unsupervised Learning", "description": "Finding patterns in unlabeled data"},
                        {"name": "Neural Networks", "description": "Computing systems inspired by biological neurons"}
                    ]
                }
                """,
                "type": "json"
            },
            {
                "name": "Code Example",
                "content": """
                def calculate_gradient_descent(learning_rate, iterations):
                    \"\"\"
                    Implements gradient descent optimization algorithm
                    \"\"\"
                    for i in range(iterations):
                        # Update parameters
                        params = params - learning_rate * gradient
                    return params
                """,
                "type": "code"
            }
        ]
        
        for content_type in content_types:
            print(f"   Testing {content_type['name']} ingestion...")
            
            # Test content validation
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/blueprints/validate-content",
                json={
                    "content": content_type["content"],
                    "content_type": content_type["type"],
                    "user_id": TEST_USER_ID
                },
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            if response.status_code != 200:
                raise Exception(f"Content validation failed for {content_type['name']}: {response.status_code}")
            
            validation_result = response.json()
            if not validation_result.get("valid"):
                raise Exception(f"Content validation failed for {content_type['name']}: {validation_result.get('errors')}")
            
            print(f"   ✅ {content_type['name']} validation successful")

    async def test_content_parsing(self) -> None:
        """Test 3: Test content parsing and structure extraction."""
        print("🔍 Step 3: Testing Content Parsing and Structure Extraction...")
        
        # Test parsing of complex content
        complex_content = """
        # Advanced Machine Learning Concepts
        
        ## Supervised Learning
        Supervised learning involves training a model on labeled data.
        
        ### Classification
        - Binary Classification: Two classes (e.g., spam/not spam)
        - Multi-class Classification: Multiple classes (e.g., image categories)
        
        ### Regression
        - Linear Regression: Predicts continuous values
        - Logistic Regression: Predicts probabilities
        
        ## Unsupervised Learning
        Finding hidden patterns in unlabeled data.
        
        ### Clustering
        - K-means: Groups similar data points
        - Hierarchical: Creates nested clusters
        
        ### Dimensionality Reduction
        - PCA: Principal Component Analysis
        - t-SNE: t-Distributed Stochastic Neighbor Embedding
        """
        
        response = await self.client.post(
            f"{AI_API_BASE_URL}/api/v1/blueprints/parse-content",
            json={
                "content": complex_content,
                "user_id": TEST_USER_ID,
                "parse_options": {
                    "extract_concepts": True,
                    "identify_relationships": True,
                    "generate_summary": True
                }
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Content parsing failed: {response.status_code}")
        
        parse_result = response.json()
        
        # Validate parsing results
        required_fields = ["concepts", "relationships", "summary", "structure"]
        for field in required_fields:
            if field not in parse_result:
                raise Exception(f"Missing required parsing field: {field}")
        
        if len(parse_result["concepts"]) < 5:
            raise Exception(f"Insufficient concepts extracted: {len(parse_result['concepts'])}")
        
        # Persist artifact for monitoring
        parse_path = self.artifacts_dir / "parse_result.json"
        with open(parse_path, "w", encoding="utf-8") as f:
            json.dump({
                "input_preview": complex_content.strip()[:500],
                "parse_result": parse_result
            }, f, ensure_ascii=False, indent=2)
        
        # Store for cross-step access
        self.test_data["parse_result_path"] = str(parse_path)
        
        print("   ✅ Content parsing successful")

    async def test_blueprint_generation(self) -> None:
        """Test 4: Test complete blueprint generation workflow."""
        print("🏗️ Step 4: Testing Blueprint Generation...")
        
        # Create a comprehensive test blueprint
        test_content = """
        # Software Development Lifecycle
        
        ## Planning Phase
        - Requirements gathering
        - Feasibility analysis
        - Resource planning
        
        ## Design Phase
        - System architecture
        - Database design
        - UI/UX design
        
        ## Development Phase
        - Coding standards
        - Version control
        - Code review process
        
        ## Testing Phase
        - Unit testing
        - Integration testing
        - User acceptance testing
        
        ## Deployment Phase
        - Production deployment
        - Monitoring setup
        - Documentation
        """
        
        # Generate blueprint
        response = await self.client.post(
            f"{AI_API_BASE_URL}/api/v1/blueprints/generate",
            json={
                "content": test_content,
                "user_id": TEST_USER_ID,
                "blueprint_options": {
                    "name": "Software Development Lifecycle",
                    "description": "Comprehensive guide to software development process",
                    "difficulty_level": "intermediate",
                    "estimated_duration": "8 weeks",
                    "prerequisites": ["Basic programming knowledge", "Understanding of project management"],
                    "learning_objectives": [
                        "Understand SDLC phases",
                        "Apply best practices",
                        "Manage development projects"
                    ]
                }
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Blueprint generation failed: {response.status_code}")
        
        blueprint_result = response.json()
        
        # Store blueprint ID for later tests
        self.test_data["blueprint_id"] = blueprint_result["blueprint_id"]
        
        # Persist artifact for monitoring
        blueprint_path = self.artifacts_dir / "blueprint_result.json"
        with open(blueprint_path, "w", encoding="utf-8") as f:
            json.dump({
                "input_preview": test_content.strip()[:500],
                "blueprint_result": blueprint_result
            }, f, ensure_ascii=False, indent=2)
        
        self.test_data["blueprint_result_path"] = str(blueprint_path)
        
        # Validate blueprint structure
        required_blueprint_fields = [
            "blueprint_id", "name", "description", "concepts", 
            "learning_objectives", "difficulty_level", "estimated_duration"
        ]
        
        for field in required_blueprint_fields:
            if field not in blueprint_result:
                raise Exception(f"Missing required blueprint field: {field}")
        
        print(f"   ✅ Blueprint generated successfully: {blueprint_result['blueprint_id']}")

    async def test_blueprint_validation(self) -> None:
        """Test 5: Test blueprint validation and quality checks."""
        print("✅ Step 5: Testing Blueprint Validation...")
        
        if "blueprint_id" not in self.test_data:
            raise Exception("No blueprint ID available for validation")
        
        blueprint_id = self.test_data["blueprint_id"]
        
        # Test blueprint retrieval
        response = await self.client.get(
            f"{AI_API_BASE_URL}/api/v1/blueprints/{blueprint_id}",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Blueprint retrieval failed: {response.status_code}")
        
        blueprint = response.json()
        
        # Validate blueprint quality
        quality_checks = [
            ("concepts", lambda x: len(x) >= 3, "At least 3 concepts"),
            ("learning_objectives", lambda x: len(x) >= 2, "At least 2 learning objectives"),
            ("difficulty_level", lambda x: x in ["beginner", "intermediate", "advanced"], "Valid difficulty level"),
            ("estimated_duration", lambda x: isinstance(x, str) and len(x) > 0, "Valid duration"),
            ("prerequisites", lambda x: isinstance(x, list), "Prerequisites list"),
        ]
        
        for field, validator, description in quality_checks:
            if field not in blueprint:
                raise Exception(f"Missing field: {field}")
            
            if not validator(blueprint[field]):
                raise Exception(f"Quality check failed for {field}: {description}")
        
        print("   ✅ Blueprint validation successful")

    async def test_error_handling(self) -> None:
        """Test 6: Test error handling for invalid inputs."""
        print("⚠️ Step 6: Testing Error Handling...")
        
        # Test invalid content
        invalid_content = ""
        response = await self.client.post(
            f"{AI_API_BASE_URL}/api/v1/blueprints/validate-content",
            json={
                "content": invalid_content,
                "content_type": "text",
                "user_id": TEST_USER_ID
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        # Should return 400 for invalid content
        if response.status_code != 400:
            raise Exception(f"Expected 400 for invalid content, got {response.status_code}")
        
        # Test invalid user ID
        response = await self.client.post(
            f"{AI_API_BASE_URL}/api/v1/blueprints/generate",
            json={
                "content": "Test content",
                "user_id": "",
                "blueprint_options": {"name": "Test"}
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        # Should return 400 for invalid user ID
        if response.status_code != 400:
            raise Exception(f"Expected 400 for invalid user ID, got {response.status_code}")
        
        print("   ✅ Error handling tests passed")

    async def test_performance_metrics(self) -> None:
        """Test 7: Test performance metrics and timing."""
        print("⏱️ Step 7: Testing Performance Metrics...")
        
        # Test blueprint generation performance
        start_time = time.time()
        
        response = await self.client.post(
            f"{AI_API_BASE_URL}/api/v1/blueprints/generate",
            json={
                "content": "# Simple Test\nThis is a simple test content for performance testing.",
                "user_id": TEST_USER_ID,
                "blueprint_options": {"name": "Performance Test"}
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code != 200:
            raise Exception(f"Performance test blueprint generation failed: {response.status_code}")
        
        # Performance thresholds
        if generation_time > 30.0:  # 30 seconds max
            raise Exception(f"Blueprint generation too slow: {generation_time:.2f}s")
        
        print(f"   ✅ Performance test passed: {generation_time:.2f}s")
        
        # Store performance metrics
        self.test_data["performance_metrics"] = {
            "blueprint_generation_time": generation_time,
            "total_test_duration": sum(r.duration for r in self.results if r.duration)
        }

    def print_results(self) -> None:
        """Print comprehensive test results."""
        print("\n" + "="*60)
        print("📊 BLUEPRINT CREATION E2E TEST RESULTS")
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
            print(f"  Blueprint Generation: {metrics['blueprint_generation_time']:.2f}s")
            print(f"  Total Test Duration: {metrics['total_test_duration']:.2f}s")
        
        print(f"\nDetailed Results:")
        for result in self.results:
            status_icon = "✅" if result.status == "PASS" else "❌"
            print(f"  {status_icon} {result.step}: {result.details}")
            if result.error:
                print(f"    Error: {result.error}")
        
        # Surface artifact locations for monitoring/troubleshooting
        if hasattr(self, "artifacts_dir"):
            print(f"\nArtifacts saved to: {self.artifacts_dir}")
            if "parse_result_path" in self.test_data:
                print(f"  - Parse result: {self.test_data['parse_result_path']}")
            if "blueprint_result_path" in self.test_data:
                print(f"  - Blueprint result: {self.test_data['blueprint_result_path']}")
        
        if failed_tests == 0:
            print(f"\n🎉 All tests passed! Blueprint creation system is working correctly.")
        else:
            print(f"\n⚠️ {failed_tests} tests failed. Please review the errors above.")

async def main():
    """Main test execution function."""
    tester = BlueprintCreationTester()
    await tester.run()

if __name__ == "__main__":
    asyncio.run(main())
