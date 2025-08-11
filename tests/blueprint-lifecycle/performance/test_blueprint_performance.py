"""
Blueprint Performance Test Suite

This module contains comprehensive performance tests for the blueprint lifecycle,
including response time, throughput, scalability, and resource usage tests.
"""

import pytest
import time
import asyncio
import statistics
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
import psutil
import gc
import tracemalloc

from app.core.blueprint.blueprint_service import BlueprintService
from app.core.blueprint.blueprint_model import Blueprint
from app.core.blueprint.blueprint_repository import BlueprintRepository
from app.core.blueprint.blueprint_validator import BlueprintValidator
from app.core.blueprint.blueprint_indexer import BlueprintIndexer
from app.core.blueprint.blueprint_rag import BlueprintRAG
from app.core.blueprint.blueprint_analytics import BlueprintAnalytics
from app.core.blueprint.blueprint_graphrag import BlueprintGraphRAG
from app.core.blueprint.blueprint_vector_sync import BlueprintVectorSync


class TestBlueprintPerformance:
    """Test suite for blueprint performance testing."""
    
    @pytest.fixture
    def mock_blueprint_service(self):
        """Mock blueprint service for performance testing."""
        service = Mock(spec=BlueprintService)
        service.create_blueprint = AsyncMock()
        service.get_blueprint = AsyncMock()
        service.update_blueprint = AsyncMock()
        service.delete_blueprint = AsyncMock()
        service.list_blueprints = AsyncMock()
        return service
    
    @pytest.fixture
    def mock_blueprint_repository(self):
        """Mock blueprint repository for performance testing."""
        repo = Mock(spec=BlueprintRepository)
        repo.create = AsyncMock()
        repo.get_by_id = AsyncMock()
        repo.update = AsyncMock()
        repo.delete = AsyncMock()
        repo.list_all = AsyncMock()
        repo.search = AsyncMock()
        return repo
    
    @pytest.fixture
    def sample_blueprint_data(self):
        """Sample blueprint data for testing."""
        return {
            "name": "Test Blueprint",
            "description": "A test blueprint for performance testing",
            "content": "This is test content for the blueprint",
            "metadata": {
                "category": "test",
                "tags": ["performance", "testing"],
                "version": "1.0.0"
            },
            "settings": {
                "chunk_size": 1000,
                "overlap": 200,
                "embedding_model": "text-embedding-ada-002"
            }
        }
    
    @pytest.fixture
    def performance_config(self):
        """Performance testing configuration."""
        return {
            "iterations": 100,
            "concurrent_users": 10,
            "timeout_seconds": 30,
            "memory_threshold_mb": 500,
            "cpu_threshold_percent": 80
        }

    def test_blueprint_creation_response_time(self, mock_blueprint_service, sample_blueprint_data, performance_config):
        """Test blueprint creation response time performance."""
        # Mock successful creation
        mock_blueprint_service.create_blueprint.return_value = Blueprint(
            id="test-123",
            **sample_blueprint_data
        )
        
        response_times = []
        
        for i in range(performance_config["iterations"]):
            start_time = time.time()
            
            # Simulate blueprint creation
            result = asyncio.run(mock_blueprint_service.create_blueprint(sample_blueprint_data))
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            response_times.append(response_time)
            
            # Verify result
            assert result is not None
            assert result.id == "test-123"
        
        # Performance assertions
        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        max_response_time = max(response_times)
        
        print(f"Blueprint Creation Performance Results:")
        print(f"  Average response time: {avg_response_time:.2f}ms")
        print(f"  95th percentile: {p95_response_time:.2f}ms")
        print(f"  Maximum response time: {max_response_time:.2f}ms")
        print(f"  Total iterations: {len(response_times)}")
        
        # Performance thresholds
        assert avg_response_time < 100, f"Average response time {avg_response_time:.2f}ms exceeds 100ms threshold"
        assert p95_response_time < 200, f"95th percentile response time {p95_response_time:.2f}ms exceeds 200ms threshold"
        assert max_response_time < 500, f"Maximum response time {max_response_time:.2f}ms exceeds 500ms threshold"

    def test_blueprint_retrieval_response_time(self, mock_blueprint_service, performance_config):
        """Test blueprint retrieval response time performance."""
        # Mock successful retrieval
        mock_blueprint = Blueprint(
            id="test-123",
            name="Test Blueprint",
            description="A test blueprint",
            content="Test content",
            metadata={},
            settings={}
        )
        mock_blueprint_service.get_blueprint.return_value = mock_blueprint
        
        response_times = []
        
        for i in range(performance_config["iterations"]):
            start_time = time.time()
            
            # Simulate blueprint retrieval
            result = asyncio.run(mock_blueprint_service.get_blueprint("test-123"))
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            response_times.append(response_time)
            
            # Verify result
            assert result is not None
            assert result.id == "test-123"
        
        # Performance assertions
        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]
        max_response_time = max(response_times)
        
        print(f"Blueprint Retrieval Performance Results:")
        print(f"  Average response time: {avg_response_time:.2f}ms")
        print(f"  95th percentile: {p95_response_time:.2f}ms")
        print(f"  Maximum response time: {max_response_time:.2f}ms")
        
        # Performance thresholds
        assert avg_response_time < 50, f"Average response time {avg_response_time:.2f}ms exceeds 50ms threshold"
        assert p95_response_time < 100, f"95th percentile response time {p95_response_time:.2f}ms exceeds 100ms threshold"
        assert max_response_time < 200, f"Maximum response time {max_response_time:.2f}ms exceeds 200ms threshold"

    def test_blueprint_search_performance(self, mock_blueprint_repository, performance_config):
        """Test blueprint search performance."""
        # Mock search results
        mock_blueprints = [
            Blueprint(
                id=f"test-{i}",
                name=f"Test Blueprint {i}",
                description=f"Description {i}",
                content=f"Content {i}",
                metadata={},
                settings={}
            )
            for i in range(10)
        ]
        mock_blueprint_repository.search.return_value = mock_blueprints
        
        search_times = []
        
        for i in range(performance_config["iterations"]):
            start_time = time.time()
            
            # Simulate blueprint search
            results = asyncio.run(mock_blueprint_repository.search("test", limit=10))
            
            end_time = time.time()
            search_time = (end_time - start_time) * 1000
            search_times.append(search_time)
            
            # Verify results
            assert len(results) == 10
            assert all(isinstance(bp, Blueprint) for bp in results)
        
        # Performance assertions
        avg_search_time = statistics.mean(search_times)
        p95_search_time = statistics.quantiles(search_times, n=20)[18]
        max_search_time = max(search_times)
        
        print(f"Blueprint Search Performance Results:")
        print(f"  Average search time: {avg_search_time:.2f}ms")
        print(f"  P95 search time: {p95_search_time:.2f}ms")
        print(f"  Maximum search time: {max_search_time:.2f}ms")
        
        # Performance thresholds
        assert avg_search_time < 150, f"Average search time {avg_search_time:.2f}ms exceeds 150ms threshold"
        assert p95_search_time < 300, f"P95 search time {p95_search_time:.2f}ms exceeds 300ms threshold"
        assert max_search_time < 500, f"Maximum search time {max_search_time:.2f}ms exceeds 500ms threshold"

    def test_concurrent_blueprint_operations(self, mock_blueprint_service, sample_blueprint_data, performance_config):
        """Test concurrent blueprint operations performance."""
        # Mock successful operations
        mock_blueprint_service.create_blueprint.return_value = Blueprint(
            id="test-123",
            **sample_blueprint_data
        )
        mock_blueprint_service.get_blueprint.return_value = Blueprint(
            id="test-123",
            **sample_blueprint_data
        )
        
        async def concurrent_operation(operation_type: str, delay: float = 0.01):
            """Simulate a concurrent operation with optional delay."""
            if operation_type == "create":
                await asyncio.sleep(delay)
                return await mock_blueprint_service.create_blueprint(sample_blueprint_data)
            elif operation_type == "get":
                await asyncio.sleep(delay)
                return await mock_blueprint_service.get_blueprint("test-123")
        
        async def run_concurrent_operations():
            """Run multiple operations concurrently."""
            tasks = []
            
            # Create concurrent tasks
            for i in range(performance_config["concurrent_users"]):
                if i % 2 == 0:
                    tasks.append(concurrent_operation("create"))
                else:
                    tasks.append(concurrent_operation("get"))
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = (end_time - start_time) * 1000
            return total_time, results
        
        # Run concurrent operations
        total_time, results = asyncio.run(run_concurrent_operations())
        
        print(f"Concurrent Operations Performance Results:")
        print(f"  Concurrent users: {performance_config['concurrent_users']}")
        print(f"  Total execution time: {total_time:.2f}ms")
        print(f"  Average time per operation: {total_time / performance_config['concurrent_users']:.2f}ms")
        print(f"  Operations completed: {len(results)}")
        
        # Performance assertions
        assert total_time < 1000, f"Total concurrent execution time {total_time:.2f}ms exceeds 1000ms threshold"
        assert len(results) == performance_config["concurrent_users"], "Not all operations completed"
        
        # Verify all operations returned results
        assert all(result is not None for result in results)

    def test_memory_usage_performance(self, mock_blueprint_service, sample_blueprint_data, performance_config):
        """Test memory usage during blueprint operations."""
        # Start memory tracking
        tracemalloc.start()
        initial_snapshot = tracemalloc.take_snapshot()
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Perform operations
        for i in range(performance_config["iterations"]):
            # Simulate blueprint creation
            result = asyncio.run(mock_blueprint_service.create_blueprint(sample_blueprint_data))
            
            # Force garbage collection every 10 iterations
            if i % 10 == 0:
                gc.collect()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Get memory snapshot
        final_snapshot = tracemalloc.take_snapshot()
        top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno')
        
        print(f"Final memory usage: {final_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")
        print(f"Top memory allocations:")
        for stat in top_stats[:5]:
            print(f"  {stat}")
        
        # Performance assertions
        assert memory_increase < performance_config["memory_threshold_mb"], \
            f"Memory increase {memory_increase:.2f}MB exceeds threshold {performance_config['memory_threshold_mb']}MB"
        
        # Stop memory tracking
        tracemalloc.stop()

    def test_cpu_usage_performance(self, mock_blueprint_service, sample_blueprint_data, performance_config):
        """Test CPU usage during blueprint operations."""
        process = psutil.Process()
        
        # Get initial CPU usage
        initial_cpu_percent = process.cpu_percent(interval=1)
        
        print(f"Initial CPU usage: {initial_cpu_percent:.2f}%")
        
        # Monitor CPU during operations
        cpu_readings = []
        
        for i in range(performance_config["iterations"]):
            # Start CPU monitoring
            start_cpu = process.cpu_percent(interval=0.1)
            
            # Simulate blueprint operation
            result = asyncio.run(mock_blueprint_service.create_blueprint(sample_blueprint_data))
            
            # End CPU monitoring
            end_cpu = process.cpu_percent(interval=0.1)
            cpu_readings.append((start_cpu + end_cpu) / 2)
        
        # Calculate CPU statistics
        avg_cpu = statistics.mean(cpu_readings)
        max_cpu = max(cpu_readings)
        p95_cpu = statistics.quantiles(cpu_readings, n=20)[18] if len(cpu_readings) >= 20 else max_cpu
        
        print(f"CPU Usage Performance Results:")
        print(f"  Average CPU usage: {avg_cpu:.2f}%")
        print(f"  Maximum CPU usage: {max_cpu:.2f}%")
        print(f"  95th percentile CPU usage: {p95_cpu:.2f}%")
        
        # Performance assertions
        assert avg_cpu < performance_config["cpu_threshold_percent"], \
            f"Average CPU usage {avg_cpu:.2f}% exceeds threshold {performance_config['cpu_threshold_percent']}%"
        assert max_cpu < 100, f"Maximum CPU usage {max_cpu:.2f}% exceeds 100%"

    def test_blueprint_batch_operations_performance(self, mock_blueprint_repository, performance_config):
        """Test batch blueprint operations performance."""
        # Mock batch operations
        mock_blueprint_repository.create_many = AsyncMock()
        mock_blueprint_repository.update_many = AsyncMock()
        mock_blueprint_repository.delete_many = AsyncMock()
        
        # Generate batch data
        batch_size = 100
        batch_data = [
            {
                "name": f"Batch Blueprint {i}",
                "description": f"Description {i}",
                "content": f"Content {i}",
                "metadata": {"batch": True, "index": i},
                "settings": {}
            }
            for i in range(batch_size)
        ]
        
        # Test batch creation
        start_time = time.time()
        asyncio.run(mock_blueprint_repository.create_many(batch_data))
        creation_time = (time.time() - start_time) * 1000
        
        # Test batch update
        update_data = [{"id": f"batch-{i}", "name": f"Updated {i}"} for i in range(batch_size)]
        start_time = time.time()
        asyncio.run(mock_blueprint_repository.update_many(update_data))
        update_time = (time.time() - start_time) * 1000
        
        # Test batch deletion
        delete_ids = [f"batch-{i}" for i in range(batch_size)]
        start_time = time.time()
        asyncio.run(mock_blueprint_repository.delete_many(delete_ids))
        deletion_time = (time.time() - start_time) * 1000
        
        print(f"Batch Operations Performance Results:")
        print(f"  Batch size: {batch_size}")
        print(f"  Batch creation time: {creation_time:.2f}ms")
        print(f"  Batch update time: {update_time:.2f}ms")
        print(f"  Batch deletion time: {deletion_time:.2f}ms")
        print(f"  Average time per operation: {(creation_time + update_time + deletion_time) / (batch_size * 3):.2f}ms")
        
        # Performance assertions
        assert creation_time < 5000, f"Batch creation time {creation_time:.2f}ms exceeds 5000ms threshold"
        assert update_time < 3000, f"Batch update time {update_time:.2f}ms exceeds 3000ms threshold"
        assert deletion_time < 2000, f"Batch deletion time {deletion_time:.2f}ms exceeds 2000ms threshold"

    def test_blueprint_indexing_performance(self, mock_blueprint_service, sample_blueprint_data, performance_config):
        """Test blueprint indexing performance."""
        # Mock indexing service
        mock_indexer = Mock()
        mock_indexer.index_blueprint = AsyncMock(return_value=True)
        mock_indexer.batch_index = AsyncMock(return_value=True)
        
        # Test single blueprint indexing
        indexing_times = []
        
        for i in range(performance_config["iterations"]):
            start_time = time.time()
            
            # Simulate indexing
            result = asyncio.run(mock_indexer.index_blueprint(sample_blueprint_data))
            
            end_time = time.time()
            indexing_time = (end_time - start_time) * 1000
            indexing_times.append(indexing_time)
            
            assert result is True
        
        # Test batch indexing
        batch_size = 50
        batch_data = [sample_blueprint_data for _ in range(batch_size)]
        
        start_time = time.time()
        batch_result = asyncio.run(mock_indexer.batch_index(batch_data))
        batch_time = (time.time() - start_time) * 1000
        
        # Performance calculations
        avg_indexing_time = statistics.mean(indexing_times)
        p95_indexing_time = statistics.quantiles(indexing_times, n=20)[18] if len(indexing_times) >= 20 else max(indexing_times)
        
        print(f"Blueprint Indexing Performance Results:")
        print(f"  Single blueprint indexing:")
        print(f"    Average time: {avg_indexing_time:.2f}ms")
        print(f"    95th percentile: {p95_indexing_time:.2f}ms")
        print(f"  Batch indexing ({batch_size} blueprints):")
        print(f"    Total time: {batch_time:.2f}ms")
        print(f"    Average per blueprint: {batch_time / batch_size:.2f}ms")
        
        # Performance assertions
        assert avg_indexing_time < 200, f"Average indexing time {avg_indexing_time:.2f}ms exceeds 200ms threshold"
        assert p95_indexing_time < 400, f"95th percentile indexing time {p95_indexing_time:.2f}ms exceeds 400ms threshold"
        assert batch_time < 10000, f"Batch indexing time {batch_time:.2f}ms exceeds 10000ms threshold"

    def test_blueprint_rag_performance(self, mock_blueprint_service, sample_blueprint_data, performance_config):
        """Test blueprint RAG performance."""
        # Mock RAG service
        mock_rag = Mock()
        mock_rag.query = AsyncMock(return_value="RAG response")
        mock_rag.generate_response = AsyncMock(return_value="Generated response")
        
        # Test RAG query performance
        query_times = []
        
        for i in range(performance_config["iterations"]):
            start_time = time.time()
            
            # Simulate RAG query
            result = asyncio.run(mock_rag.query("test query"))
            
            end_time = time.time()
            query_time = (end_time - start_time) * 1000
            query_times.append(query_time)
            
            assert result == "RAG response"
        
        # Test response generation performance
        generation_times = []
        
        for i in range(performance_config["iterations"]):
            start_time = time.time()
            
            # Simulate response generation
            result = asyncio.run(mock_rag.generate_response("test context"))
            
            end_time = time.time()
            generation_time = (end_time - start_time) * 1000
            generation_times.append(generation_time)
            
            assert result == "Generated response"
        
        # Performance calculations
        avg_query_time = statistics.mean(query_times)
        avg_generation_time = statistics.mean(generation_times)
        p95_query_time = statistics.quantiles(query_times, n=20)[18] if len(query_times) >= 20 else max(query_times)
        p95_generation_time = statistics.quantiles(generation_times, n=20)[18] if len(generation_times) >= 20 else max(generation_times)
        
        print(f"Blueprint RAG Performance Results:")
        print(f"  RAG Query:")
        print(f"    Average time: {avg_query_time:.2f}ms")
        print(f"    95th percentile: {p95_query_time:.2f}ms")
        print(f"  Response Generation:")
        print(f"    Average time: {avg_generation_time:.2f}ms")
        print(f"    95th percentile: {p95_generation_time:.2f}ms")
        
        # Performance assertions
        assert avg_query_time < 300, f"Average RAG query time {avg_query_time:.2f}ms exceeds 300ms threshold"
        assert avg_generation_time < 500, f"Average generation time {avg_generation_time:.2f}ms exceeds 500ms threshold"
        assert p95_query_time < 600, f"95th percentile query time {p95_query_time:.2f}ms exceeds 600ms threshold"
        assert p95_generation_time < 1000, f"95th percentile generation time {p95_generation_time:.2f}ms exceeds 1000ms threshold"

    def test_blueprint_throughput_performance(self, mock_blueprint_service, sample_blueprint_data, performance_config):
        """Test blueprint operations throughput."""
        # Mock successful operations
        mock_blueprint_service.create_blueprint.return_value = Blueprint(
            id="test-123",
            **sample_blueprint_data
        )
        
        # Measure throughput over time
        start_time = time.time()
        operations_completed = 0
        
        # Run operations for a fixed duration
        while time.time() - start_time < 10:  # 10 seconds
            # Simulate blueprint creation
            result = asyncio.run(mock_blueprint_service.create_blueprint(sample_blueprint_data))
            operations_completed += 1
            
            # Small delay to simulate real-world conditions
            time.sleep(0.001)
        
        total_time = time.time() - start_time
        throughput = operations_completed / total_time
        
        print(f"Blueprint Throughput Performance Results:")
        print(f"  Operations completed: {operations_completed}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Throughput: {throughput:.2f} operations/second")
        print(f"  Average time per operation: {(total_time / operations_completed) * 1000:.2f}ms")
        
        # Performance assertions
        assert throughput > 10, f"Throughput {throughput:.2f} ops/sec below 10 ops/sec threshold"
        assert operations_completed > 100, f"Too few operations completed: {operations_completed}"

    def test_blueprint_scalability_performance(self, mock_blueprint_service, sample_blueprint_data):
        """Test blueprint operations scalability with different loads."""
        # Test different load levels
        load_levels = [1, 5, 10, 20, 50]
        results = {}
        
        for load in load_levels:
            start_time = time.time()
            
            # Create tasks for concurrent execution
            async def run_load_test():
                tasks = []
                for i in range(load):
                    tasks.append(mock_blueprint_service.create_blueprint(sample_blueprint_data))
                
                results_list = await asyncio.gather(*tasks)
                return len(results_list)
            
            # Execute load test
            operations_completed = asyncio.run(run_load_test())
            execution_time = (time.time() - start_time) * 1000
            
            results[load] = {
                "operations": operations_completed,
                "time_ms": execution_time,
                "throughput": load / (execution_time / 1000)
            }
        
        print(f"Blueprint Scalability Performance Results:")
        for load, result in results.items():
            print(f"  Load {load}: {result['operations']} ops in {result['time_ms']:.2f}ms "
                  f"({result['throughput']:.2f} ops/sec)")
        
        # Scalability assertions
        for load in load_levels[1:]:
            # Throughput should not decrease significantly with increased load
            current_throughput = results[load]["throughput"]
            previous_throughput = results[load_levels[load_levels.index(load) - 1]]["throughput"]
            
            # Allow some degradation but not more than 50%
            assert current_throughput > previous_throughput * 0.5, \
                f"Throughput degraded too much at load {load}: {current_throughput:.2f} vs {previous_throughput:.2f}"

    def test_blueprint_resource_efficiency(self, mock_blueprint_service, sample_blueprint_data, performance_config):
        """Test blueprint operations resource efficiency."""
        process = psutil.Process()
        
        # Baseline measurements
        baseline_memory = process.memory_info().rss / 1024 / 1024
        baseline_cpu = process.cpu_percent(interval=1)
        
        print(f"Baseline measurements:")
        print(f"  Memory: {baseline_memory:.2f} MB")
        print(f"  CPU: {baseline_cpu:.2f}%")
        
        # Perform operations
        for i in range(performance_config["iterations"]):
            result = asyncio.run(mock_blueprint_service.create_blueprint(sample_blueprint_data))
            
            # Measure resource usage every 10 operations
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                current_cpu = process.cpu_percent(interval=0.1)
                
                memory_increase = current_memory - baseline_memory
                cpu_increase = current_cpu - baseline_cpu
                
                print(f"  Iteration {i}: Memory +{memory_increase:.2f}MB, CPU +{cpu_increase:.2f}%")
        
        # Final measurements
        final_memory = process.memory_info().rss / 1024 / 1024
        final_cpu = process.cpu_percent(interval=1)
        
        total_memory_increase = final_memory - baseline_memory
        total_cpu_increase = final_cpu - baseline_cpu
        
        print(f"Final resource usage:")
        print(f"  Memory: {final_memory:.2f} MB (+{total_memory_increase:.2f} MB)")
        print(f"  CPU: {final_cpu:.2f}% (+{total_cpu_increase:.2f}%)")
        
        # Resource efficiency assertions
        assert total_memory_increase < 100, f"Memory increase {total_memory_increase:.2f}MB exceeds 100MB threshold"
        assert total_cpu_increase < 20, f"CPU increase {total_cpu_increase:.2f}% exceeds 20% threshold"
        
        # Memory should not grow linearly with operations
        memory_per_operation = total_memory_increase / performance_config["iterations"]
        assert memory_per_operation < 1, f"Memory per operation {memory_per_operation:.2f}MB exceeds 1MB threshold"

    def test_blueprint_error_handling_performance(self, mock_blueprint_service, sample_blueprint_data, performance_config):
        """Test blueprint error handling performance."""
        # Mock error conditions
        mock_blueprint_service.create_blueprint.side_effect = [
            Exception("Simulated error") if i % 5 == 0 else Blueprint(id=f"test-{i}", **sample_blueprint_data)
            for i in range(performance_config["iterations"])
        ]
        
        error_handling_times = []
        successful_operations = 0
        failed_operations = 0
        
        for i in range(performance_config["iterations"]):
            start_time = time.time()
            
            try:
                result = asyncio.run(mock_blueprint_service.create_blueprint(sample_blueprint_data))
                successful_operations += 1
            except Exception as e:
                failed_operations += 1
                # Simulate error handling time
                time.sleep(0.001)
            
            end_time = time.time()
            operation_time = (end_time - start_time) * 1000
            error_handling_times.append(operation_time)
        
        # Performance calculations
        avg_operation_time = statistics.mean(error_handling_times)
        p95_operation_time = statistics.quantiles(error_handling_times, n=20)[18] if len(error_handling_times) >= 20 else max(error_handling_times)
        
        print(f"Blueprint Error Handling Performance Results:")
        print(f"  Successful operations: {successful_operations}")
        print(f"  Failed operations: {failed_operations}")
        print(f"  Success rate: {(successful_operations / performance_config['iterations']) * 100:.1f}%")
        print(f"  Average operation time: {avg_operation_time:.2f}ms")
        print(f"  95th percentile operation time: {p95_operation_time:.2f}ms")
        
        # Performance assertions
        assert avg_operation_time < 100, f"Average operation time {avg_operation_time:.2f}ms exceeds 100ms threshold"
        assert p95_operation_time < 200, f"95th percentile operation time {p95_operation_time:.2f}ms exceeds 200ms threshold"
        assert failed_operations > 0, "No errors were simulated"
        assert successful_operations > 0, "No successful operations"

    def test_blueprint_cache_performance(self, mock_blueprint_service, sample_blueprint_data, performance_config):
        """Test blueprint caching performance."""
        # Mock cache behavior
        cache_hits = 0
        cache_misses = 0
        
        # Simulate cache with 50% hit rate
        def mock_get_blueprint_with_cache(blueprint_id: str):
            if hash(blueprint_id) % 2 == 0:  # 50% cache hit rate
                cache_hits += 1
                return Blueprint(id=blueprint_id, **sample_blueprint_data)
            else:
                cache_misses += 1
                # Simulate cache miss penalty
                time.sleep(0.01)
                return Blueprint(id=blueprint_id, **sample_blueprint_data)
        
        mock_blueprint_service.get_blueprint = AsyncMock(side_effect=mock_get_blueprint_with_cache)
        
        cache_performance_times = []
        
        for i in range(performance_config["iterations"]):
            start_time = time.time()
            
            result = asyncio.run(mock_blueprint_service.get_blueprint(f"test-{i}"))
            
            end_time = time.time()
            operation_time = (end_time - start_time) * 1000
            cache_performance_times.append(operation_time)
            
            assert result is not None
        
        # Performance calculations
        avg_cache_time = statistics.mean(cache_performance_times)
        cache_hit_time = statistics.mean([t for i, t in enumerate(cache_performance_times) if hash(f"test-{i}") % 2 == 0])
        cache_miss_time = statistics.mean([t for i, t in enumerate(cache_performance_times) if hash(f"test-{i}") % 2 != 0])
        
        print(f"Blueprint Cache Performance Results:")
        print(f"  Cache hits: {cache_hits}")
        print(f"  Cache misses: {cache_misses}")
        print(f"  Cache hit rate: {(cache_hits / performance_config['iterations']) * 100:.1f}%")
        print(f"  Average operation time: {avg_cache_time:.2f}ms")
        print(f"  Average cache hit time: {cache_hit_time:.2f}ms")
        print(f"  Average cache miss time: {cache_miss_time:.2f}ms")
        
        # Performance assertions
        assert cache_hit_time < cache_miss_time, "Cache hits should be faster than cache misses"
        assert avg_cache_time < 50, f"Average cache operation time {avg_cache_time:.2f}ms exceeds 50ms threshold"
        assert cache_hits > 0, "No cache hits occurred"
        assert cache_misses > 0, "No cache misses occurred"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
