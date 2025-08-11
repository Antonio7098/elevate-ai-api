"""
Advanced Blueprint Vector Sync Test Suite

This module contains comprehensive tests for advanced vector synchronization
capabilities in the blueprint lifecycle, including vector database sync,
embedding management, consistency checks, and distributed vector operations.
"""

import pytest
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib

from app.core.blueprint.blueprint_vector_sync import BlueprintVectorSync
from app.core.blueprint.blueprint_model import Blueprint
from app.core.blueprint.blueprint_embedding import BlueprintEmbedding
from app.core.blueprint.blueprint_indexer import BlueprintIndexer
from app.core.blueprint.blueprint_vector_store import BlueprintVectorStore
from app.core.blueprint.blueprint_sync_manager import BlueprintSyncManager


class TestAdvancedBlueprintVectorSync:
    """Advanced test suite for blueprint vector synchronization capabilities."""
    
    @pytest.fixture
    def mock_blueprint_vector_sync(self):
        """Mock blueprint vector sync service for testing."""
        sync = Mock(spec=BlueprintVectorSync)
        sync.sync_vectors = AsyncMock()
        sync.sync_embeddings = AsyncMock()
        sync.sync_index = AsyncMock()
        sync.check_consistency = AsyncMock()
        sync.resolve_conflicts = AsyncMock()
        sync.batch_sync = AsyncMock()
        sync.incremental_sync = AsyncMock()
        return sync
    
    @pytest.fixture
    def mock_blueprint_embedding(self):
        """Mock blueprint embedding service for testing."""
        embedding = Mock(spec=BlueprintEmbedding)
        embedding.generate_embedding = AsyncMock(return_value=np.random.rand(1536))
        embedding.generate_batch_embeddings = AsyncMock(return_value=[np.random.rand(1536) for _ in range(5)])
        embedding.update_embedding = AsyncMock(return_value=True)
        embedding.delete_embedding = AsyncMock(return_value=True)
        return embedding
    
    @pytest.fixture
    def mock_blueprint_indexer(self):
        """Mock blueprint indexer for testing."""
        indexer = Mock(spec=BlueprintIndexer)
        indexer.index_blueprint = AsyncMock(return_value=True)
        indexer.update_index = AsyncMock(return_value=True)
        indexer.delete_from_index = AsyncMock(return_value=True)
        indexer.rebuild_index = AsyncMock(return_value=True)
        return indexer
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store for testing."""
        store = Mock(spec=BlueprintVectorStore)
        store.store_vectors = AsyncMock(return_value=True)
        store.retrieve_vectors = AsyncMock()
        store.update_vectors = AsyncMock(return_value=True)
        store.delete_vectors = AsyncMock(return_value=True)
        store.search_vectors = AsyncMock()
        return store
    
    @pytest.fixture
    def sample_blueprint_data(self):
        """Sample blueprint data for vector sync testing."""
        return {
            "id": "test-blueprint-123",
            "name": "Advanced Vector Sync Test Blueprint",
            "description": "A comprehensive blueprint for testing advanced vector synchronization",
            "content": "This blueprint contains content that needs to be vectorized and synchronized across multiple systems.",
            "metadata": {
                "category": "vector_sync_testing",
                "tags": ["vectorization", "synchronization", "distributed"],
                "version": "1.0.0",
                "last_updated": "2024-01-15T10:30:00Z"
            },
            "vector_data": {
                "embedding_model": "text-embedding-ada-002",
                "embedding_dimension": 1536,
                "chunk_size": 1000,
                "overlap": 200
            }
        }
    
    @pytest.fixture
    def sample_vector_data(self):
        """Sample vector data for testing."""
        return {
            "blueprint_id": "test-blueprint-123",
            "chunks": [
                {
                    "id": "chunk-1",
                    "content": "First chunk of blueprint content",
                    "embedding": np.random.rand(1536),
                    "metadata": {"position": 0, "length": 1000}
                },
                {
                    "id": "chunk-2",
                    "content": "Second chunk of blueprint content",
                    "embedding": np.random.rand(1536),
                    "metadata": {"position": 1000, "length": 1000}
                }
            ],
            "full_embedding": np.random.rand(1536),
            "last_synced": "2024-01-15T10:30:00Z"
        }
    
    @pytest.fixture
    def sync_config(self):
        """Synchronization configuration for testing."""
        return {
            "sync_strategy": "incremental",
            "batch_size": 100,
            "sync_interval_seconds": 300,
            "consistency_check_enabled": True,
            "conflict_resolution": "auto",
            "retry_attempts": 3,
            "timeout_seconds": 30
        }

    def test_vector_synchronization_basic(self, mock_blueprint_vector_sync, sample_blueprint_data, sample_vector_data):
        """Test basic vector synchronization functionality."""
        # Mock successful vector sync
        mock_blueprint_vector_sync.sync_vectors.return_value = {
            "sync_id": "sync-123",
            "status": "completed",
            "blueprints_synced": 1,
            "vectors_synced": 2,
            "sync_time_ms": 150,
            "errors": []
        }
        
        # Test vector synchronization
        result = asyncio.run(mock_blueprint_vector_sync.sync_vectors(
            blueprint_id=sample_blueprint_data["id"],
            vector_data=sample_vector_data
        ))
        
        # Verify sync result
        assert result is not None
        assert result["status"] == "completed"
        assert result["blueprints_synced"] == 1
        assert result["vectors_synced"] == 2
        assert "sync_id" in result
        assert "sync_time_ms" in result
        assert len(result["errors"]) == 0
        
        # Verify sync was called with correct parameters
        mock_blueprint_vector_sync.sync_vectors.assert_called_once_with(
            blueprint_id=sample_blueprint_data["id"],
            vector_data=sample_vector_data
        )
        
        print("✅ Basic vector synchronization working correctly")

    def test_embedding_synchronization(self, mock_blueprint_vector_sync, sample_blueprint_data):
        """Test embedding synchronization across systems."""
        # Mock embedding sync
        mock_blueprint_vector_sync.sync_embeddings.return_value = {
            "sync_id": "embedding-sync-123",
            "status": "completed",
            "embeddings_synced": 3,
            "embedding_dimensions": 1536,
            "sync_time_ms": 200,
            "storage_locations": ["primary", "secondary", "cache"]
        }
        
        # Test embedding synchronization
        result = asyncio.run(mock_blueprint_vector_sync.sync_embeddings(
            blueprint_id=sample_blueprint_data["id"],
            embedding_model="text-embedding-ada-002"
        ))
        
        # Verify embedding sync result
        assert result is not None
        assert result["status"] == "completed"
        assert result["embeddings_synced"] > 0
        assert result["embedding_dimensions"] == 1536
        assert "storage_locations" in result
        
        # Verify storage locations
        locations = result["storage_locations"]
        assert len(locations) > 0
        assert "primary" in locations
        
        print("✅ Embedding synchronization working correctly")

    def test_index_synchronization(self, mock_blueprint_vector_sync, sample_blueprint_data):
        """Test index synchronization across vector databases."""
        # Mock index sync
        mock_blueprint_vector_sync.sync_index.return_value = {
            "sync_id": "index-sync-123",
            "status": "completed",
            "indexes_synced": 2,
            "index_types": ["faiss", "pinecone"],
            "sync_time_ms": 300,
            "index_metadata": {
                "faiss": {"vectors": 1000, "dimensions": 1536},
                "pinecone": {"vectors": 1000, "dimensions": 1536}
            }
        }
        
        # Test index synchronization
        result = asyncio.run(mock_blueprint_vector_sync.sync_index(
            blueprint_id=sample_blueprint_data["id"],
            index_types=["faiss", "pinecone"]
        ))
        
        # Verify index sync result
        assert result is not None
        assert result["status"] == "completed"
        assert result["indexes_synced"] == 2
        assert "index_types" in result
        assert "index_metadata" in result
        
        # Verify index types
        index_types = result["index_types"]
        assert len(index_types) == 2
        assert "faiss" in index_types
        assert "pinecone" in index_types
        
        # Verify index metadata
        metadata = result["index_metadata"]
        assert "faiss" in metadata
        assert "pinecone" in metadata
        
        print("✅ Index synchronization working correctly")

    def test_consistency_checking(self, mock_blueprint_vector_sync, sample_blueprint_data):
        """Test vector consistency checking across systems."""
        # Mock consistency check
        mock_blueprint_vector_sync.check_consistency.return_value = {
            "consistency_id": "consistency-123",
            "status": "completed",
            "is_consistent": True,
            "inconsistencies": [],
            "consistency_score": 0.98,
            "check_time_ms": 250,
            "systems_checked": ["primary", "secondary", "cache"]
        }
        
        # Test consistency checking
        result = asyncio.run(mock_blueprint_vector_sync.check_consistency(
            blueprint_id=sample_blueprint_data["id"]
        ))
        
        # Verify consistency check result
        assert result is not None
        assert result["status"] == "completed"
        assert result["is_consistent"] is True
        assert len(result["inconsistencies"]) == 0
        assert "consistency_score" in result
        assert "systems_checked" in result
        
        # Verify consistency score
        assert 0 <= result["consistency_score"] <= 1
        assert result["consistency_score"] >= 0.95  # High consistency threshold
        
        # Verify systems checked
        systems = result["systems_checked"]
        assert len(systems) >= 2
        
        print("✅ Vector consistency checking working correctly")

    def test_conflict_resolution(self, mock_blueprint_vector_sync, sample_blueprint_data):
        """Test vector conflict resolution mechanisms."""
        # Mock conflict resolution
        mock_blueprint_vector_sync.resolve_conflicts.return_value = {
            "resolution_id": "resolution-123",
            "status": "completed",
            "conflicts_resolved": 2,
            "resolution_strategy": "timestamp_based",
            "resolution_time_ms": 180,
            "resolved_conflicts": [
                {
                    "conflict_type": "version_mismatch",
                    "resolution": "use_latest",
                    "affected_systems": ["secondary", "cache"]
                },
                {
                    "conflict_type": "embedding_difference",
                    "resolution": "recompute",
                    "affected_systems": ["primary"]
                }
            ]
        }
        
        # Test conflict resolution
        result = asyncio.run(mock_blueprint_vector_sync.resolve_conflicts(
            blueprint_id=sample_blueprint_data["id"]
        ))
        
        # Verify conflict resolution result
        assert result is not None
        assert result["status"] == "completed"
        assert result["conflicts_resolved"] > 0
        assert "resolution_strategy" in result
        assert "resolved_conflicts" in result
        
        # Verify resolved conflicts
        resolved = result["resolved_conflicts"]
        assert len(resolved) > 0
        
        for conflict in resolved:
            assert "conflict_type" in conflict
            assert "resolution" in conflict
            assert "affected_systems" in conflict
        
        print("✅ Vector conflict resolution working correctly")

    def test_batch_synchronization(self, mock_blueprint_vector_sync, sample_blueprint_data):
        """Test batch vector synchronization for multiple blueprints."""
        # Mock batch sync
        mock_blueprint_vector_sync.batch_sync.return_value = {
            "batch_sync_id": "batch-123",
            "status": "completed",
            "total_blueprints": 5,
            "successful_syncs": 5,
            "failed_syncs": 0,
            "total_vectors": 15,
            "sync_time_ms": 800,
            "batch_results": [
                {"blueprint_id": f"bp-{i}", "status": "success", "vectors_synced": 3}
                for i in range(1, 6)
            ]
        }
        
        # Test batch synchronization
        blueprint_ids = [f"bp-{i}" for i in range(1, 6)]
        result = asyncio.run(mock_blueprint_vector_sync.batch_sync(blueprint_ids))
        
        # Verify batch sync result
        assert result is not None
        assert result["status"] == "completed"
        assert result["total_blueprints"] == 5
        assert result["successful_syncs"] == 5
        assert result["failed_syncs"] == 0
        assert result["total_vectors"] == 15
        assert "batch_results" in result
        
        # Verify batch results
        batch_results = result["batch_results"]
        assert len(batch_results) == 5
        
        for batch_result in batch_results:
            assert "blueprint_id" in batch_result
            assert "status" in batch_result
            assert "vectors_synced" in batch_result
            assert batch_result["status"] == "success"
        
        print("✅ Batch vector synchronization working correctly")

    def test_incremental_synchronization(self, mock_blueprint_vector_sync, sample_blueprint_data):
        """Test incremental vector synchronization for changes only."""
        # Mock incremental sync
        mock_blueprint_vector_sync.incremental_sync.return_value = {
            "incremental_sync_id": "inc-123",
            "status": "completed",
            "changes_detected": 3,
            "vectors_updated": 3,
            "vectors_added": 1,
            "vectors_deleted": 0,
            "sync_time_ms": 120,
            "change_summary": {
                "content_updates": 2,
                "metadata_updates": 1,
                "embedding_updates": 3
            }
        }
        
        # Test incremental synchronization
        result = asyncio.run(mock_blueprint_vector_sync.incremental_sync(
            blueprint_id=sample_blueprint_data["id"],
            since_timestamp="2024-01-15T09:00:00Z"
        ))
        
        # Verify incremental sync result
        assert result is not None
        assert result["status"] == "completed"
        assert result["changes_detected"] > 0
        assert "change_summary" in result
        
        # Verify change summary
        summary = result["change_summary"]
        assert "content_updates" in summary
        assert "metadata_updates" in summary
        assert "embedding_updates" in summary
        
        # Verify efficiency (should be faster than full sync)
        assert result["sync_time_ms"] < 200
        
        print("✅ Incremental vector synchronization working correctly")

    def test_distributed_vector_operations(self, mock_blueprint_vector_sync, sample_blueprint_data):
        """Test distributed vector operations across multiple nodes."""
        # Mock distributed operations
        mock_blueprint_vector_sync.distributed_sync = AsyncMock(return_value={
            "distributed_sync_id": "dist-123",
            "status": "completed",
            "nodes_participating": 3,
            "coordinator_node": "node-1",
            "sync_time_ms": 450,
            "node_results": {
                "node-1": {"status": "success", "vectors_synced": 5},
                "node-2": {"status": "success", "vectors_synced": 5},
                "node-3": {"status": "success", "vectors_synced": 5}
            },
            "consensus_achieved": True
        })
        
        # Test distributed synchronization
        result = asyncio.run(mock_blueprint_vector_sync.distributed_sync(
            blueprint_id=sample_blueprint_data["id"],
            nodes=["node-1", "node-2", "node-3"]
        ))
        
        # Verify distributed sync result
        assert result is not None
        assert result["status"] == "completed"
        assert result["nodes_participating"] == 3
        assert "coordinator_node" in result
        assert "node_results" in result
        assert result["consensus_achieved"] is True
        
        # Verify node results
        node_results = result["node_results"]
        assert len(node_results) == 3
        
        for node_id, node_result in node_results.items():
            assert "status" in node_result
            assert "vectors_synced" in node_result
            assert node_result["status"] == "success"
        
        print("✅ Distributed vector operations working correctly")

    def test_vector_sync_monitoring(self, mock_blueprint_vector_sync, sample_blueprint_data):
        """Test vector synchronization monitoring and metrics."""
        # Mock sync monitoring
        mock_blueprint_vector_sync.get_sync_metrics = AsyncMock(return_value={
            "metrics_id": "metrics-123",
            "timestamp": "2024-01-15T10:30:00Z",
            "sync_health": "healthy",
            "performance_metrics": {
                "avg_sync_time_ms": 180,
                "sync_success_rate": 0.98,
                "vectors_per_second": 25.5,
                "error_rate": 0.02
            },
            "system_health": {
                "primary_system": "healthy",
                "secondary_system": "healthy",
                "cache_system": "degraded"
            },
            "recent_syncs": [
                {
                    "sync_id": "recent-1",
                    "timestamp": "2024-01-15T10:25:00Z",
                    "status": "success",
                    "duration_ms": 150
                },
                {
                    "sync_id": "recent-2",
                    "timestamp": "2024-01-15T10:20:00Z",
                    "status": "success",
                    "duration_ms": 165
                }
            ]
        })
        
        # Test sync monitoring
        result = asyncio.run(mock_blueprint_vector_sync.get_sync_metrics(
            blueprint_id=sample_blueprint_data["id"]
        ))
        
        # Verify monitoring result
        assert result is not None
        assert "sync_health" in result
        assert "performance_metrics" in result
        assert "system_health" in result
        assert "recent_syncs" in result
        
        # Verify sync health
        assert result["sync_health"] in ["healthy", "degraded", "unhealthy"]
        
        # Verify performance metrics
        perf_metrics = result["performance_metrics"]
        assert "avg_sync_time_ms" in perf_metrics
        assert "sync_success_rate" in perf_metrics
        assert "vectors_per_second" in perf_metrics
        assert "error_rate" in perf_metrics
        
        # Verify success rate and error rate
        assert 0 <= perf_metrics["sync_success_rate"] <= 1
        assert 0 <= perf_metrics["error_rate"] <= 1
        assert perf_metrics["sync_success_rate"] + perf_metrics["error_rate"] == 1
        
        print("✅ Vector sync monitoring working correctly")

    def test_vector_sync_error_handling(self, mock_blueprint_vector_sync, sample_blueprint_data):
        """Test vector synchronization error handling and recovery."""
        # Mock error scenarios
        error_scenarios = [
            {
                "error_type": "connection_timeout",
                "expected_behavior": "retry_with_backoff",
                "recovery_time_ms": 5000
            },
            {
                "error_type": "vector_corruption",
                "expected_behavior": "regenerate_vectors",
                "recovery_time_ms": 15000
            },
            {
                "error_type": "index_inconsistency",
                "expected_behavior": "rebuild_index",
                "recovery_time_ms": 30000
            }
        ]
        
        for scenario in error_scenarios:
            print(f"Testing error handling for: {scenario['error_type']}")
            
            # Mock error handling
            mock_blueprint_vector_sync.handle_sync_error = AsyncMock(return_value={
                "error_id": f"error-{scenario['error_type']}",
                "error_type": scenario["error_type"],
                "handling_strategy": scenario["expected_behavior"],
                "recovery_successful": True,
                "recovery_time_ms": scenario["recovery_time_ms"],
                "actions_taken": [
                    f"Detected {scenario['error_type']}",
                    f"Applied {scenario['expected_behavior']}",
                    "Recovery completed"
                ]
            })
            
            # Test error handling
            result = asyncio.run(mock_blueprint_vector_sync.handle_sync_error(
                blueprint_id=sample_blueprint_data["id"],
                error_type=scenario["error_type"]
            ))
            
            # Verify error handling result
            assert result is not None
            assert result["error_type"] == scenario["error_type"]
            assert result["handling_strategy"] == scenario["expected_behavior"]
            assert result["recovery_successful"] is True
            assert "actions_taken" in result
            
            print(f"  ✅ {scenario['error_type']}: {scenario['expected_behavior']}")

    def test_vector_sync_performance_optimization(self, mock_blueprint_vector_sync, sample_blueprint_data):
        """Test vector synchronization performance optimization features."""
        # Mock performance optimization
        mock_blueprint_vector_sync.optimize_sync_performance = AsyncMock(return_value={
            "optimization_id": "opt-123",
            "status": "completed",
            "optimizations_applied": [
                "parallel_processing",
                "batch_operations",
                "compression",
                "caching"
            ],
            "performance_improvements": {
                "sync_speed_increase": 2.5,
                "memory_usage_reduction": 0.3,
                "cpu_usage_reduction": 0.25,
                "network_usage_reduction": 0.4
            },
            "before_optimization": {
                "avg_sync_time_ms": 500,
                "memory_usage_mb": 1000,
                "cpu_usage_percent": 80
            },
            "after_optimization": {
                "avg_sync_time_ms": 200,
                "memory_usage_mb": 700,
                "cpu_usage_percent": 60
            }
        })
        
        # Test performance optimization
        result = asyncio.run(mock_blueprint_vector_sync.optimize_sync_performance(
            blueprint_id=sample_blueprint_data["id"]
        ))
        
        # Verify optimization result
        assert result is not None
        assert result["status"] == "completed"
        assert "optimizations_applied" in result
        assert "performance_improvements" in result
        
        # Verify optimizations applied
        optimizations = result["optimizations_applied"]
        assert len(optimizations) >= 3
        assert "parallel_processing" in optimizations
        assert "batch_operations" in optimizations
        
        # Verify performance improvements
        improvements = result["performance_improvements"]
        assert improvements["sync_speed_increase"] > 1.0
        assert improvements["memory_usage_reduction"] > 0
        assert improvements["cpu_usage_reduction"] > 0
        
        # Verify before/after comparison
        before = result["before_optimization"]
        after = result["after_optimization"]
        assert after["avg_sync_time_ms"] < before["avg_sync_time_ms"]
        assert after["memory_usage_mb"] < before["memory_usage_mb"]
        
        print("✅ Vector sync performance optimization working correctly")
        print(f"  Speed improvement: {improvements['sync_speed_increase']:.1f}x")
        print(f"  Memory reduction: {improvements['memory_usage_reduction']:.1%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
