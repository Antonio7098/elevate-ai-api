#!/usr/bin/env python3
"""
Comprehensive Blueprint GraphRAG E2E Test
Tests the complete GraphRAG system including:
- Knowledge graph construction from blueprints
- Graph-based retrieval and reasoning
- Graph analytics and insights
- Graph traversal and path finding
- Graph embeddings and similarity
- Graph query optimization
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

class BlueprintGraphRAGTester:
    def __init__(self):
        self.results: List[TestResult] = []
        self.test_data = {
            "blueprint_ids": [],
            "graph_data": {},
            "performance_metrics": {},
            "graph_queries": []
        }
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0)
        )
        
    async def run(self) -> None:
        """Run the complete blueprint GraphRAG test suite."""
        print("🚀 Starting Comprehensive Blueprint GraphRAG E2E Test\n")
        
        try:
            await self.run_step(self.test_environment_setup)
            await self.run_step(self.test_knowledge_graph_construction)
            await self.run_step(self.test_graph_retrieval)
            await self.run_step(self.test_graph_reasoning)
            await self.run_step(self.test_graph_analytics)
            await self.run_step(self.test_graph_traversal)
            await self.run_step(self.test_graph_embeddings)
            await self.run_step(self.test_graph_query_optimization)
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
        """Test 1: Verify environment is ready for GraphRAG testing."""
        print("🔧 Step 1: Environment Setup and Validation...")
        
        # Check AI API health
        response = await self.client.get(f"{AI_API_BASE_URL}/api/health")
        if response.status_code != 200:
            raise Exception(f"AI API health check failed: {response.status_code}")
        
        # Check Core API health
        response = await self.client.get(f"{CORE_API_BASE_URL}/health")
        if response.status_code != 200:
            raise Exception(f"Core API health check failed: {response.status_code}")
        
        # Check GraphRAG endpoints
        response = await self.client.get(f"{CORE_API_BASE_URL}/api/graphrag/health")
        if response.status_code != 200:
            raise Exception(f"GraphRAG health check failed: {response.status_code}")
        
        print("✅ Environment setup completed successfully")

    async def test_knowledge_graph_construction(self) -> None:
        """Test 2: Test knowledge graph construction from blueprints."""
        print("🏗️ Step 2: Testing Knowledge Graph Construction...")
        
        # Create test blueprints with rich content for graph construction
        test_blueprints = [
            {
                "title": "Machine Learning Fundamentals",
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed. It includes supervised learning, unsupervised learning, and reinforcement learning.",
                "metadata": {
                    "category": "AI/ML",
                    "tags": ["machine learning", "artificial intelligence", "supervised learning", "unsupervised learning"],
                    "concepts": ["ML", "AI", "learning algorithms", "data science"]
                }
            },
            {
                "title": "Deep Learning Neural Networks",
                "content": "Deep learning uses neural networks with multiple layers to model complex patterns. It includes convolutional neural networks (CNNs) for image processing and recurrent neural networks (RNNs) for sequential data.",
                "metadata": {
                    "category": "AI/ML",
                    "tags": ["deep learning", "neural networks", "CNNs", "RNNs", "image processing"],
                    "concepts": ["neural networks", "deep learning", "computer vision", "natural language processing"]
                }
            },
            {
                "title": "Data Science Pipeline",
                "content": "The data science pipeline includes data collection, preprocessing, feature engineering, model training, evaluation, and deployment. Each step is crucial for building effective ML models.",
                "metadata": {
                    "category": "Data Science",
                    "tags": ["data science", "pipeline", "feature engineering", "model training", "deployment"],
                    "concepts": ["data pipeline", "ML workflow", "model lifecycle", "data preprocessing"]
                }
            }
        ]
        
        # Create blueprints and trigger graph construction
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
                
                # Trigger graph construction
                graph_response = await self.client.post(
                    f"{CORE_API_BASE_URL}/api/graphrag/construct",
                    json={"blueprint_id": blueprint_id},
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
                
                if graph_response.status_code == 202:
                    print(f"✅ Graph construction started for blueprint: {blueprint_id}")
                else:
                    print(f"⚠️ Graph construction failed for blueprint: {blueprint_id}")
            else:
                raise Exception(f"Failed to create test blueprint: {response.status_code}")
        
        # Wait for graph construction to complete
        print("⏳ Waiting for graph construction to complete...")
        for _ in range(30):  # Wait up to 30 seconds
            await asyncio.sleep(1)
            
            # Check graph status
            response = await self.client.get(
                f"{CORE_API_BASE_URL}/api/graphrag/status",
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            if response.status_code == 200:
                status_data = response.json()
                if all(status_data.get("blueprint_status", {}).get(bp_id) == "completed" 
                       for bp_id in self.test_data["blueprint_ids"]):
                    print("✅ All graph constructions completed")
                    break
        
        print("✅ Knowledge graph construction testing completed")

    async def test_graph_retrieval(self) -> None:
        """Test 3: Test graph-based retrieval system."""
        print("🔍 Step 3: Testing Graph-Based Retrieval...")
        
        # Test semantic search with graph context
        search_queries = [
            "machine learning algorithms",
            "neural network architectures",
            "data science workflow",
            "AI and ML concepts"
        ]
        
        for query in search_queries:
            response = await self.client.post(
                f"{CORE_API_BASE_URL}/api/graphrag/search",
                json={
                    "query": query,
                    "use_graph_context": True,
                    "max_results": 5
                },
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            if response.status_code == 200:
                search_results = response.json()
                self.test_data["graph_queries"].append({
                    "query": query,
                    "results": search_results
                })
                print(f"✅ Graph search completed for: '{query}'")
            else:
                raise Exception(f"Graph search failed for query '{query}': {response.status_code}")
        
        # Test concept-based retrieval
        concept_response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/graphrag/retrieve/concepts",
            json={
                "concepts": ["machine learning", "neural networks"],
                "relationship_depth": 2
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if concept_response.status_code == 200:
            concept_results = concept_response.json()
            self.test_data["graph_data"]["concept_retrieval"] = concept_results
            print("✅ Concept-based retrieval completed successfully")
        else:
            raise Exception(f"Concept-based retrieval failed: {concept_response.status_code}")
        
        print("✅ Graph-based retrieval testing completed")

    async def test_graph_reasoning(self) -> None:
        """Test 4: Test graph-based reasoning capabilities."""
        print("🧠 Step 4: Testing Graph-Based Reasoning...")
        
        # Test logical reasoning queries
        reasoning_queries = [
            {
                "type": "inference",
                "premise": "If a system uses neural networks, then it is a form of machine learning",
                "query": "What type of system is deep learning?"
            },
            {
                "type": "causality",
                "query": "What are the prerequisites for building a data science pipeline?"
            },
            {
                "type": "similarity",
                "query": "How are CNNs and RNNs related to neural networks?"
            }
        ]
        
        for query in reasoning_queries:
            response = await self.client.post(
                f"{CORE_API_BASE_URL}/api/graphrag/reason",
                json=query,
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            if response.status_code == 200:
                reasoning_result = response.json()
                print(f"✅ Reasoning completed for {query['type']} query")
            else:
                print(f"⚠️ Reasoning failed for {query['type']} query: {response.status_code}")
        
        # Test multi-hop reasoning
        multi_hop_response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/graphrag/reason/multi-hop",
            json={
                "start_concept": "machine learning",
                "target_concept": "data preprocessing",
                "max_hops": 3
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if multi_hop_response.status_code == 200:
            multi_hop_result = multi_hop_response.json()
            self.test_data["graph_data"]["multi_hop_reasoning"] = multi_hop_result
            print("✅ Multi-hop reasoning completed successfully")
        else:
            print(f"⚠️ Multi-hop reasoning failed: {multi_hop_response.status_code}")
        
        print("✅ Graph-based reasoning testing completed")

    async def test_graph_analytics(self) -> None:
        """Test 5: Test graph analytics and insights."""
        print("📊 Step 5: Testing Graph Analytics...")
        
        # Test graph statistics
        stats_response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/graphrag/analytics/stats",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            self.test_data["graph_data"]["graph_stats"] = stats_data
            print("✅ Graph statistics retrieved successfully")
        else:
            raise Exception(f"Failed to retrieve graph statistics: {stats_response.status_code}")
        
        # Test centrality analysis
        centrality_response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/graphrag/analytics/centrality",
            params={"top_k": 10},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if centrality_response.status_code == 200:
            centrality_data = centrality_response.json()
            self.test_data["graph_data"]["centrality_analysis"] = centrality_data
            print("✅ Centrality analysis completed successfully")
        else:
            raise Exception(f"Failed to perform centrality analysis: {centrality_response.status_code}")
        
        # Test community detection
        community_response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/graphrag/analytics/communities",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if community_response.status_code == 200:
            community_data = community_response.json()
            self.test_data["graph_data"]["community_detection"] = community_data
            print("✅ Community detection completed successfully")
        else:
            raise Exception(f"Failed to perform community detection: {community_response.status_code}")
        
        print("✅ Graph analytics testing completed")

    async def test_graph_traversal(self) -> None:
        """Test 6: Test graph traversal and path finding."""
        print("🛤️ Step 6: Testing Graph Traversal...")
        
        # Test shortest path finding
        path_response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/graphrag/traverse/path",
            json={
                "start_node": "machine learning",
                "end_node": "data preprocessing",
                "algorithm": "dijkstra"
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if path_response.status_code == 200:
            path_data = path_response.json()
            self.test_data["graph_data"]["shortest_path"] = path_data
            print("✅ Shortest path finding completed successfully")
        else:
            raise Exception(f"Failed to find shortest path: {path_response.status_code}")
        
        # Test breadth-first traversal
        bfs_response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/graphrag/traverse/bfs",
            json={
                "start_node": "neural networks",
                "max_depth": 3
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if bfs_response.status_code == 200:
            bfs_data = bfs_response.json()
            self.test_data["graph_data"]["bfs_traversal"] = bfs_data
            print("✅ BFS traversal completed successfully")
        else:
            raise Exception(f"Failed to perform BFS traversal: {bfs_response.status_code}")
        
        # Test depth-first traversal
        dfs_response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/graphrag/traverse/dfs",
            json={
                "start_node": "deep learning",
                "max_depth": 3
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if dfs_response.status_code == 200:
            dfs_data = dfs_response.json()
            self.test_data["graph_data"]["dfs_traversal"] = dfs_data
            print("✅ DFS traversal completed successfully")
        else:
            raise Exception(f"Failed to perform DFS traversal: {dfs_response.status_code}")
        
        print("✅ Graph traversal testing completed")

    async def test_graph_embeddings(self) -> None:
        """Test 7: Test graph embeddings and similarity."""
        print("🔢 Step 7: Testing Graph Embeddings...")
        
        # Test node embeddings
        embeddings_response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/graphrag/embeddings/nodes",
            params={"concepts": ["machine learning", "neural networks", "data science"]},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if embeddings_response.status_code == 200:
            embeddings_data = embeddings_response.json()
            self.test_data["graph_data"]["node_embeddings"] = embeddings_data
            print("✅ Node embeddings retrieved successfully")
        else:
            raise Exception(f"Failed to retrieve node embeddings: {embeddings_response.status_code}")
        
        # Test graph similarity
        similarity_response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/graphrag/similarity",
            json={
                "concept1": "machine learning",
                "concept2": "deep learning",
                "similarity_type": "cosine"
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if similarity_response.status_code == 200:
            similarity_data = similarity_response.json()
            self.test_data["graph_data"]["concept_similarity"] = similarity_data
            print("✅ Concept similarity calculated successfully")
        else:
            raise Exception(f"Failed to calculate concept similarity: {similarity_response.status_code}")
        
        # Test graph clustering
        clustering_response = await self.client.get(
            f"{CORE_API_BASE_URL}/api/graphrag/embeddings/clusters",
            params={"n_clusters": 3},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if clustering_response.status_code == 200:
            clustering_data = clustering_response.json()
            self.test_data["graph_data"]["graph_clustering"] = clustering_data
            print("✅ Graph clustering completed successfully")
        else:
            raise Exception(f"Failed to perform graph clustering: {clustering_response.status_code}")
        
        print("✅ Graph embeddings testing completed")

    async def test_graph_query_optimization(self) -> None:
        """Test 8: Test graph query optimization and performance."""
        print("⚡ Step 8: Testing Graph Query Optimization...")
        
        # Test query performance metrics
        start_time = time.time()
        response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/graphrag/search",
            json={
                "query": "machine learning algorithms",
                "use_graph_context": True,
                "max_results": 10
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        query_time = time.time() - start_time
        
        if response.status_code == 200:
            self.test_data["performance_metrics"]["graph_search_time"] = query_time
            print(f"✅ Graph search completed in {query_time:.3f}s")
        else:
            raise Exception(f"Graph search failed: {response.status_code}")
        
        # Test concurrent graph queries
        async def make_concurrent_graph_queries():
            queries = [
                {"query": "neural networks", "max_results": 5},
                {"query": "data science", "max_results": 5},
                {"query": "AI concepts", "max_results": 5}
            ]
            
            tasks = []
            for query in queries:
                task = self.client.post(
                    f"{CORE_API_BASE_URL}/api/graphrag/search",
                    json=query,
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
                tasks.append(task)
            
            return await asyncio.gather(*tasks)
        
        start_time = time.time()
        concurrent_responses = await make_concurrent_graph_queries()
        concurrent_time = time.time() - start_time
        
        self.test_data["performance_metrics"]["concurrent_graph_queries_time"] = concurrent_time
        self.test_data["performance_metrics"]["concurrent_graph_queries_count"] = len(concurrent_responses)
        
        print(f"✅ Concurrent graph queries completed in {concurrent_time:.3f}s")
        
        # Test query plan analysis
        plan_response = await self.client.post(
            f"{CORE_API_BASE_URL}/api/graphrag/query/plan",
            json={
                "query": "machine learning algorithms",
                "use_graph_context": True
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if plan_response.status_code == 200:
            plan_data = plan_response.json()
            self.test_data["graph_data"]["query_plan"] = plan_data
            print("✅ Query plan analysis completed successfully")
        else:
            print(f"⚠️ Query plan analysis failed: {plan_response.status_code}")
        
        print("✅ Graph query optimization testing completed")

    def print_results(self) -> None:
        """Print comprehensive test results."""
        print("\n" + "="*80)
        print("🏗️ BLUEPRINT GRAPHRAG E2E TEST RESULTS")
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
        
        if self.test_data.get("graph_data"):
            print(f"\n🏗️ GRAPH DATA SUMMARY:")
            for key, value in self.test_data["graph_data"].items():
                if isinstance(value, dict):
                    print(f"   {key}: {len(value)} items")
                else:
                    print(f"   {key}: {value}")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        if failed_tests == 0:
            print("   🎉 All tests passed! GraphRAG system is working correctly.")
        else:
            print(f"   ⚠️  {failed_tests} test(s) failed. Review error details above.")
        
        print("="*80)

async def main():
    """Main entry point for the GraphRAG test suite."""
    tester = BlueprintGraphRAGTester()
    await tester.run()

if __name__ == "__main__":
    asyncio.run(main())
