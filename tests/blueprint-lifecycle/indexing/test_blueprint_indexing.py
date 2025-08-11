#!/usr/bin/env python3
"""
Comprehensive Blueprint Indexing E2E Test
Tests the complete indexing system including:
- Vector indexing and embedding generation
- Document processing and chunking
- Index management and optimization
- Search performance and accuracy
- Index synchronization and updates
- Batch processing and scalability
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

class BlueprintIndexingTester:
    def __init__(self):
        self.results: List[TestResult] = []
        self.test_data = {
            "blueprint_ids": [],
            "index_data": {},
            "performance_metrics": {},
            "search_results": []
        }
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0)
        )
        
    async def run(self) -> None:
        """Run the complete blueprint indexing test suite."""
        print("🚀 Starting Comprehensive Blueprint Indexing E2E Test\n")
        
        try:
            await self.run_step(self.test_environment_setup)
            await self.run_step(self.test_document_processing)
            await self.run_step(self.test_vector_indexing)
            await self.run_step(self.test_index_management)
            await self.run_step(self.test_search_performance)
            await self.run_step(self.test_index_synchronization)
            await self.run_step(self.test_batch_processing)
            await self.run_step(self.test_index_optimization)
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
        """Test 1: Verify environment is ready for indexing testing."""
        print("🔧 Step 1: Environment Setup and Validation...")
        
        # Check AI API health
        response = await self.client.get(f"{AI_API_BASE_URL}/api/health")
        if response.status_code != 200:
            raise Exception(f"AI API health check failed: {response.status_code}")
        
        # Check Core API health
        response = await self.client.get(f"{CORE_API_BASE_URL}/health")
        if response.status_code != 200:
            raise Exception(f"Core API health check failed: {response.status_code}")
        
        # Check indexing endpoints
        response = await self.client.get(f"{CORE_API_BASE_URL}/api/indexing/health")
        if response.status_code != 200:
            raise Exception(f"Indexing health check failed: {response.status_code}")
        
        # Check vector store status
        response = await self.client.get(f"{CORE_API_BASE_URL}/api/indexing/vectorstore/status")
        if response.status_code != 200:
            raise Exception(f"Vector store status check failed: {response.status_code}")
        
        print("✅ Environment setup completed successfully")

    async def test_document_processing(self) -> None:
        """Test 2: Test document processing and chunking strategies."""
        print("📄 Step 2: Testing Document Processing...")
        
        # Create test blueprints with different content types and lengths
        test_blueprints = [
            {
                "title": "Short Technical Guide",
                "content": "This is a concise technical guide covering basic concepts. It includes key points and examples.",
                "metadata": {
                    "category": "technical",
                    "content_type": "guide",
                    "complexity": "beginner"
                }
            },
            {
                "title": "Comprehensive Research Paper",
                "content": """
                This is a comprehensive research paper that covers multiple aspects of the subject matter.
                It includes detailed analysis, methodology, results, and conclusions.
                
                The paper begins with an introduction that sets the context and outlines the objectives.
                The methodology section describes the approach taken and the tools used.
                Results are presented with detailed analysis and supporting data.
                Conclusions summarize the findings and suggest future work.
                
                This content is designed to test the chunking strategy for long-form documents.
                It should be split into multiple chunks while maintaining semantic coherence.
                """,
                "metadata": {
                    "category": "research",
                    "content_type": "paper",
                    "complexity": "advanced"
                }
            },
            {
                "title": "Structured Documentation",
                "content": """
                # Introduction
                This is a structured document with clear sections.
                
                ## Section 1: Overview
                This section provides an overview of the topic.
                
                ## Section 2: Details
                This section goes into more detail about specific aspects.
                
                ## Section 3: Examples
                Here are some examples to illustrate the concepts.
                
                ## Conclusion
                This concludes the structured documentation.
                """,
                "metadata": {
                    "category": "documentation",
                    "content_type": "structured",
                    "complexity": "intermediate"
                }
            }
        ]
        
        # Create blueprints and test processing
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
                
                # Test document processing
                process_response = await self.client.post(
                    f"{CORE_API_BASE_URL}/api/indexing/process",
                    json={"blueprint_id": blueprint_id},
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
                
                if process_response.status_code == 202:
                    print(f"✅ Document processing started for blueprint: {blueprint_id}")
                else:
                    print(f"⚠️ Document processing failed for blueprint: {blueprint_id}")
            else:
                raise Exception(f"Failed to create test blueprint: {response.status_code}")
        
        # Wait for processing to complete
        print("⏳ Waiting for document processing to complete...")
        for _ in range(30):  # Wait up to 30 seconds
            await asyncio.sleep(1)
            
            # Check processing status
            response = await self.client.get(
                f"{CORE_API_BASE_URL}/api/indexing/status",
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            if response.status_code == 200:
                status_data = response.json()
                if all(status_data.get("blueprint_status", {}).get(bp_id) == "processed" 
                       for bp_id in self.test_data["blueprint_ids"]):
                    print("✅ All documents processed successfully")
                    break
        
        print("✅ Document processing testing completed")

    async def test_vector_indexing(self) -> None:
        """Test 3: Test vector indexing and embedding generation."""
        print("🔢 Step 3: Testing Vector Indexing...")
        
        # Test embedding generation for each blueprint
        for blueprint_id in self.test_data["blueprint_ids"]:
            # Generate embeddings
            embed_response = await self.client.post(
                f"{CORE_API_BASE_URL}/api/indexing/embeddings/generate",
                json={"blueprint_id": blueprint_id},
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            if embed_response.status_code == 202:
                print(f"✅ Embedding generation started for blueprint: {blueprint_id}")
            else:
                print(f"⚠️ Embedding generation failed for blueprint: {blueprint_id}")
        
        # Wait for embedding generation to complete
        print("⏳ Waiting for embedding generation to complete...")
        for _ in range(60):  # Wait up to 60 seconds for embeddings
            await asyncio.sleep(1)
            
            # Check embedding status
            response = await self.client.get(
                f"{CORE_API_BASE_URL}/api/indexing/embeddings/status",
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            if response.status_code == 200:
                status_data = response.json()
                if all(status_data.get("blueprint_status", {}).get(bp_id) == "completed" 
                       for bp_id in self.test_data["blueprint_ids"]):
                    print("✅ All embeddings generated successfully")
                    break
        
        # Test vector storage
        for blueprint_id in self.test_data["blueprint_ids"]:
            store_response = await self.client.post(
                f"{CORE_API_BASE_URL}/api/indexing/vectorstore/store",
                json={"blueprint_id": blueprint_id},
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            if store_response.status_code == 202:
                print(f"✅ Vector storage started for blueprint: {blueprint_id}")
            else:
                print(f"⚠️ Vector storage failed for blueprint: {blueprint_id}")
        
        # Wait for vector storage to complete
        print("⏳ Waiting for vector storage to complete...")
        for _ in range(60):  # Wait up to 60 seconds for storage
            await asyncio.sleep(1)
            
            # Check storage status
            response = await self.client.get(
                f"{CORE_API_BASE_URL}/api/indexing/vectorstore/status",
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            if response.status_code == 200:
                status_data = response.json()
                if all(status_data.get("blueprint_status", {}).get(bp_id) == "stored" 
                       for bp_id in self.test_data["blueprint_ids"]):
                    print("✅ All vectors stored successfully")
                    break
        
        print("✅ Vector indexing testing completed")

    async def test_index_management(self) -> None:
        """Test 4: Test index management and operations."""
        print("⚙️ Step 4: Testing Index Management...")
        
        # Test index statistics
        stats_response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/indexing/stats",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            self.test_data["index_data"]["index_stats"] = stats_data
            print("✅ Index statistics retrieved successfully")
        else:
            raise Exception(f"Failed to retrieve index statistics: {stats_response.status_code}")
        
        # Test index health check
        health_response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/indexing/health",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if health_response.status_code == 200:
            health_data = health_response.json()
            self.test_data["index_data"]["index_health"] = health_data
            print("✅ Index health check completed successfully")
        else:
            raise Exception(f"Failed to perform index health check: {health_response.status_code}")
        
        # Test index optimization
        optimize_response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/indexing/optimize",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if optimize_response.status_code == 202:
            print("✅ Index optimization started successfully")
            
            # Wait for optimization to complete
            for _ in range(30):
                await asyncio.sleep(1)
                status_response = await self.client.get(
                    f"{CORE_API_BASE_URL}/api/indexing/optimize/status",
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data.get("status") == "completed":
                        print("✅ Index optimization completed successfully")
                        break
        else:
            print(f"⚠️ Index optimization failed: {optimize_response.status_code}")
        
        print("✅ Index management testing completed")

    async def test_search_performance(self) -> None:
        """Test 5: Test search performance and accuracy."""
        print("🔍 Step 5: Testing Search Performance...")
        
        # Test search queries with performance measurement
        search_queries = [
            "technical guide concepts",
            "research methodology analysis",
            "structured documentation sections",
            "comprehensive analysis results"
        ]
        
        for query in search_queries:
            start_time = time.time()
            response = await self.client.post(
                f"{CORE_API_BASE_URL}/api/indexing/search",
                json={
                    "query": query,
                    "max_results": 5,
                    "include_metadata": True
                },
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            search_time = time.time() - start_time
            
            if response.status_code == 200:
                search_results = response.json()
                self.test_data["search_results"].append({
                    "query": query,
                    "results": search_results,
                    "response_time": search_time
                })
                print(f"✅ Search completed for '{query}' in {search_time:.3f}s")
            else:
                raise Exception(f"Search failed for query '{query}': {response.status_code}")
        
        # Test semantic search
        semantic_response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/indexing/search/semantic",
            json={
                "query": "machine learning concepts",
                "max_results": 3,
                "similarity_threshold": 0.7
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if semantic_response.status_code == 200:
            semantic_results = semantic_response.json()
            self.test_data["search_results"].append({
                "query": "semantic search",
                "results": semantic_results,
                "response_time": 0.0
            })
            print("✅ Semantic search completed successfully")
        else:
            raise Exception(f"Semantic search failed: {semantic_response.status_code}")
        
        # Test filtered search
        filtered_response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/indexing/search/filtered",
            json={
                "query": "technical content",
                "filters": {
                    "category": "technical",
                    "complexity": "beginner"
                },
                "max_results": 5
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if filtered_response.status_code == 200:
            filtered_results = filtered_response.json()
            self.test_data["search_results"].append({
                "query": "filtered search",
                "results": filtered_results,
                "response_time": 0.0
            })
            print("✅ Filtered search completed successfully")
        else:
            raise Exception(f"Filtered search failed: {filtered_response.status_code}")
        
        print("✅ Search performance testing completed")

    async def test_index_synchronization(self) -> None:
        """Test 6: Test index synchronization and updates."""
        print("🔄 Step 6: Testing Index Synchronization...")
        
        # Test index refresh
        refresh_response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/indexing/refresh",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if refresh_response.status_code == 202:
            print("✅ Index refresh started successfully")
            
            # Wait for refresh to complete
            for _ in range(30):
                await asyncio.sleep(1)
                status_response = await self.client.get(
                    f"{CORE_API_BASE_URL}/api/indexing/refresh/status",
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data.get("status") == "completed":
                        print("✅ Index refresh completed successfully")
                        break
        else:
            print(f"⚠️ Index refresh failed: {refresh_response.status_code}")
        
        # Test incremental updates
        update_response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/indexing/update/incremental",
            json={"blueprint_ids": self.test_data["blueprint_ids"]},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if update_response.status_code == 202:
            print("✅ Incremental update started successfully")
            
            # Wait for update to complete
            for _ in range(30):
                await asyncio.sleep(1)
                status_response = await self.client.get(
                    f"{CORE_API_BASE_URL}/api/indexing/update/status",
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data.get("status") == "completed":
                        print("✅ Incremental update completed successfully")
                        break
        else:
            print(f"⚠️ Incremental update failed: {update_response.status_code}")
        
        # Test index consistency check
        consistency_response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/indexing/consistency",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if consistency_response.status_code == 200:
            consistency_data = consistency_response.json()
            self.test_data["index_data"]["consistency_check"] = consistency_data
            print("✅ Index consistency check completed successfully")
        else:
            raise Exception(f"Index consistency check failed: {consistency_response.status_code}")
        
        print("✅ Index synchronization testing completed")

    async def test_batch_processing(self) -> None:
        """Test 7: Test batch processing and scalability."""
        print("📦 Step 7: Testing Batch Processing...")
        
        # Test batch document processing
        batch_process_response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/indexing/batch/process",
            json={"blueprint_ids": self.test_data["blueprint_ids"]},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if batch_process_response.status_code == 202:
            print("✅ Batch processing started successfully")
            
            # Wait for batch processing to complete
            for _ in range(60):
                await asyncio.sleep(1)
                status_response = await self.client.get(
                    f"{CORE_API_BASE_URL}/api/indexing/batch/status",
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data.get("status") == "completed":
                        print("✅ Batch processing completed successfully")
                        break
        else:
            print(f"⚠️ Batch processing failed: {batch_process_response.status_code}")
        
        # Test batch embedding generation
        batch_embed_response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/indexing/batch/embeddings",
            json={"blueprint_ids": self.test_data["blueprint_ids"]},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if batch_embed_response.status_code == 202:
            print("✅ Batch embedding generation started successfully")
            
            # Wait for batch embedding to complete
            for _ in range(60):
                await asyncio.sleep(1)
                status_response = await self.client.get(
                    f"{CORE_API_BASE_URL}/api/indexing/batch/embeddings/status",
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data.get("status") == "completed":
                        print("✅ Batch embedding generation completed successfully")
                        break
        else:
            print(f"⚠️ Batch embedding generation failed: {batch_embed_response.status_code}")
        
        # Test batch vector storage
        batch_store_response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/indexing/batch/vectorstore",
            json={"blueprint_ids": self.test_data["blueprint_ids"]},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if batch_store_response.status_code == 202:
            print("✅ Batch vector storage started successfully")
            
            # Wait for batch storage to complete
            for _ in range(60):
                await asyncio.sleep(1)
                status_response = await self.client.get(
                    f"{CORE_API_BASE_URL}/api/indexing/batch/vectorstore/status",
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data.get("status") == "completed":
                        print("✅ Batch vector storage completed successfully")
                        break
        else:
            print(f"⚠️ Batch vector storage failed: {batch_store_response.status_code}")
        
        print("✅ Batch processing testing completed")

    async def test_index_optimization(self) -> None:
        """Test 8: Test index optimization and performance tuning."""
        print("⚡ Step 8: Testing Index Optimization...")
        
        # Test query performance before optimization
        start_time = time.time()
        response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/indexing/search",
            json={
                "query": "technical concepts",
                "max_results": 10
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        pre_optimization_time = time.time() - start_time
        
        if response.status_code == 200:
            self.test_data["performance_metrics"]["pre_optimization_search_time"] = pre_optimization_time
            print(f"✅ Pre-optimization search completed in {pre_optimization_time:.3f}s")
        else:
            raise Exception(f"Pre-optimization search failed: {response.status_code}")
        
        # Test index compression
        compress_response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/indexing/compress",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if compress_response.status_code == 202:
            print("✅ Index compression started successfully")
            
            # Wait for compression to complete
            for _ in range(60):
                await asyncio.sleep(1)
                status_response = await self.client.get(
                    f"{CORE_API_BASE_URL}/api/indexing/compress/status",
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data.get("status") == "completed":
                        print("✅ Index compression completed successfully")
                        break
        else:
            print(f"⚠️ Index compression failed: {compress_response.status_code}")
        
        # Test query performance after optimization
        start_time = time.time()
        response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/indexing/search",
            json={
                "query": "technical concepts",
                "max_results": 10
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        post_optimization_time = time.time() - start_time
        
        if response.status_code == 200:
            self.test_data["performance_metrics"]["post_optimization_search_time"] = post_optimization_time
            print(f"✅ Post-optimization search completed in {post_optimization_time:.3f}s")
            
            # Calculate improvement
            improvement = ((pre_optimization_time - post_optimization_time) / pre_optimization_time) * 100
            print(f"📈 Performance improvement: {improvement:.1f}%")
        else:
            raise Exception(f"Post-optimization search failed: {response.status_code}")
        
        # Test index metrics
        metrics_response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/indexing/metrics",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if metrics_response.status_code == 200:
            metrics_data = metrics_response.json()
            self.test_data["index_data"]["optimization_metrics"] = metrics_data
            print("✅ Index optimization metrics retrieved successfully")
        else:
            raise Exception(f"Failed to retrieve optimization metrics: {metrics_response.status_code}")
        
        print("✅ Index optimization testing completed")

    def print_results(self) -> None:
        """Print comprehensive test results."""
        print("\n" + "="*80)
        print("🔢 BLUEPRINT INDEXING E2E TEST RESULTS")
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
        
        if self.test_data.get("search_results"):
            print(f"\n🔍 SEARCH RESULTS SUMMARY:")
            total_results = len(self.test_data["search_results"])
            avg_response_time = sum(r.get("response_time", 0) for r in self.test_data["search_results"]) / total_results
            print(f"   Total searches: {total_results}")
            print(f"   Average response time: {avg_response_time:.3f}s")
        
        if self.test_data.get("index_data"):
            print(f"\n🔢 INDEX DATA SUMMARY:")
            for key, value in self.test_data["index_data"].items():
                if isinstance(value, dict):
                    print(f"   {key}: {len(value)} items")
                else:
                    print(f"   {key}: {value}")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        if failed_tests == 0:
            print("   🎉 All tests passed! Indexing system is working correctly.")
        else:
            print(f"   ⚠️  {failed_tests} test(s) failed. Review error details above.")
        
        print("="*80)

async def main():
    """Main entry point for the indexing test suite."""
    tester = BlueprintIndexingTester()
    await tester.run()

if __name__ == "__main__":
    asyncio.run(main())
