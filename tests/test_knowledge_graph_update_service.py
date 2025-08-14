"""
Tests for Knowledge Graph Update Service.

This module tests the KnowledgeGraphUpdateService which handles automatic
knowledge graph updates when blueprints change, consistency checks, and
performance monitoring.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from app.services.knowledge_graph_update_service import (
    KnowledgeGraphUpdateService,
    UpdateType,
    UpdateStatus,
    GraphUpdateOperation,
    UpdateBatch
)
from app.models.blueprint_centric import BlueprintSection, DifficultyLevel


class TestKnowledgeGraphUpdateService:
    """Test class for KnowledgeGraphUpdateService."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock VectorStore for testing."""
        return MagicMock()
    
    @pytest.fixture
    def mock_traversal_service(self):
        """Mock KnowledgeGraphTraversal for testing."""
        return MagicMock()
    
    @pytest.fixture
    def update_service(self, mock_vector_store, mock_traversal_service):
        """KnowledgeGraphUpdateService instance for testing."""
        return KnowledgeGraphUpdateService(mock_vector_store, mock_traversal_service)
    
    @pytest.fixture
    def sample_section(self):
        """Sample BlueprintSection for testing."""
        return BlueprintSection(
            id=1,
            title="Test Section",
            description="A test section",
            blueprint_id=123,
            parent_section_id=None,
            depth=0,
            order_index=1,
            difficulty=DifficultyLevel.INTERMEDIATE,
            estimated_time_minutes=30,
            user_id=456,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            children=[]
        )
    
    def test_service_initialization(self, update_service):
        """Test service initialization with proper defaults."""
        assert update_service.max_batch_size == 50
        assert update_service.max_concurrent_batches == 3
        assert update_service.consistency_check_interval == 300
        assert len(update_service.update_queue) == 0
        assert len(update_service.processing_batches) == 0
        assert len(update_service.completed_batches) == 0
        assert len(update_service.failed_batches) == 0
    
    @pytest.mark.asyncio
    async def test_schedule_section_update_success(self, update_service, sample_section):
        """Test successful scheduling of a section update."""
        # Schedule update
        batch_id = await update_service.schedule_section_update(
            blueprint_id="test-blueprint-123",
            section_id="1",
            update_type=UpdateType.SECTION_ADDED,
            section=sample_section,
            metadata={"test": "data"}
        )
        
        # Verify batch was created and queued
        assert batch_id.startswith("batch_section_1_")
        assert len(update_service.update_queue) == 1
        
        batch = update_service.update_queue[0]
        assert batch.batch_id == batch_id
        assert batch.blueprint_id == "test-blueprint-123"
        assert batch.status == UpdateStatus.PENDING
        assert len(batch.operations) == 1
        
        operation = batch.operations[0]
        assert operation.update_type == UpdateType.SECTION_ADDED
        assert operation.target_id == "1"
        assert operation.target_type == "section"
        assert operation.blueprint_id == "test-blueprint-123"
        assert operation.section_id == "1"
        assert operation.metadata["test"] == "data"
        assert operation.metadata["section_title"] == "Test Section"
        assert operation.metadata["section_depth"] == 0
    
    @pytest.mark.asyncio
    async def test_schedule_section_update_without_section(self, update_service):
        """Test scheduling section update without section data."""
        batch_id = await update_service.schedule_section_update(
            blueprint_id="test-blueprint-123",
            section_id="1",
            update_type=UpdateType.SECTION_DELETED,
            metadata={"deletion_reason": "obsolete"}
        )
        
        assert batch_id.startswith("batch_section_1_")
        assert len(update_service.update_queue) == 1
        
        batch = update_service.update_queue[0]
        operation = batch.operations[0]
        assert operation.update_type == UpdateType.SECTION_DELETED
        assert operation.metadata["deletion_reason"] == "obsolete"
    
    @pytest.mark.asyncio
    async def test_schedule_section_update_error_handling(self, update_service):
        """Test error handling in section update scheduling."""
        # Test with invalid section ID that would cause an error
        with pytest.raises(Exception):
            await update_service.schedule_section_update(
                blueprint_id="test-blueprint",
                section_id="",  # Empty section ID should cause an error
                update_type=UpdateType.SECTION_ADDED
            )
    
    @pytest.mark.asyncio
    async def test_check_graph_consistency_success(self, update_service):
        """Test successful graph consistency check."""
        # Mock the private methods to return test data
        update_service._find_orphaned_nodes = AsyncMock(return_value=["node1", "node2"])
        update_service._find_broken_relationships = AsyncMock(return_value=["rel1"])
        update_service._find_circular_dependencies = AsyncMock(return_value=[])
        update_service._find_missing_metadata = AsyncMock(return_value=["meta1", "meta2", "meta3"])
        
        result = await update_service.check_graph_consistency("test-blueprint-123")
        
        # Verify result structure
        assert isinstance(result, dict)
        assert result["consistent"] is False  # Has issues
        assert len(result["issues"]) == 2
        assert len(result["warnings"]) == 1
        assert result["orphaned_nodes_count"] == 2
        assert result["broken_relationships_count"] == 1
        assert result["circular_dependencies_count"] == 0
        assert result["missing_metadata_count"] == 3
        assert "check_duration" in result
        assert "timestamp" in result
        
        # Verify performance metrics were recorded
        assert len(update_service.performance_metrics["consistency_check_duration"]) == 1
    
    @pytest.mark.asyncio
    async def test_check_graph_consistency_no_issues(self, update_service):
        """Test consistency check when no issues are found."""
        # Mock all checks to return empty lists
        update_service._find_orphaned_nodes = AsyncMock(return_value=[])
        update_service._find_broken_relationships = AsyncMock(return_value=[])
        update_service._find_circular_dependencies = AsyncMock(return_value=[])
        update_service._find_missing_metadata = AsyncMock(return_value=[])
        
        result = await update_service.check_graph_consistency()
        
        assert result["consistent"] is True
        assert len(result["issues"]) == 0
        assert len(result["warnings"]) == 0
        assert result["orphaned_nodes_count"] == 0
        assert result["broken_relationships_count"] == 0
        assert result["circular_dependencies_count"] == 0
        assert result["missing_metadata_count"] == 0
    
    @pytest.mark.asyncio
    async def test_check_graph_consistency_error_handling(self, update_service):
        """Test error handling in consistency check."""
        # Mock one method to raise an exception
        update_service._find_orphaned_nodes = AsyncMock(side_effect=Exception("Database error"))
        
        with pytest.raises(Exception, match="Database error"):
            await update_service.check_graph_consistency()
    
    def test_get_performance_metrics_empty(self, update_service):
        """Test performance metrics when no operations have been performed."""
        metrics = update_service.get_performance_metrics()
        
        assert "update_duration" in metrics
        assert "consistency_check_duration" in metrics
        assert "queue_status" in metrics
        
        # All averages should be 0 when no data
        assert metrics["update_duration"]["average"] == 0
        assert metrics["update_duration"]["count"] == 0
        assert metrics["consistency_check_duration"]["average"] == 0
        assert metrics["consistency_check_duration"]["count"] == 0
        
        # Queue status should reflect current state
        assert metrics["queue_status"]["pending_batches"] == 0
        assert metrics["queue_status"]["processing_batches"] == 0
        assert metrics["queue_status"]["completed_batches"] == 0
        assert metrics["queue_status"]["failed_batches"] == 0
    
    def test_get_performance_metrics_with_data(self, update_service):
        """Test performance metrics with recorded data."""
        # Add some test data to performance metrics
        update_service.performance_metrics["update_duration"] = [1.5, 2.0, 1.0]
        update_service.performance_metrics["consistency_check_duration"] = [0.5, 0.8]
        
        # Add some test batches
        update_service.update_queue = [MagicMock()]  # 1 pending
        update_service.processing_batches = {"batch1", "batch2"}  # 2 processing
        update_service.completed_batches = {"batch3": MagicMock()}  # 1 completed
        update_service.failed_batches = {"batch4": MagicMock()}  # 1 failed
        
        metrics = update_service.get_performance_metrics()
        
        # Verify calculations
        assert metrics["update_duration"]["average"] == 1.5  # (1.5 + 2.0 + 1.0) / 3
        assert metrics["update_duration"]["min"] == 1.0
        assert metrics["update_duration"]["max"] == 2.0
        assert metrics["update_duration"]["count"] == 3
        
        assert metrics["consistency_check_duration"]["average"] == 0.65  # (0.5 + 0.8) / 2
        assert metrics["consistency_check_duration"]["min"] == 0.5
        assert metrics["consistency_check_duration"]["max"] == 0.8
        assert metrics["consistency_check_duration"]["count"] == 2
        
        # Verify queue status
        assert metrics["queue_status"]["pending_batches"] == 1
        assert metrics["queue_status"]["processing_batches"] == 2
        assert metrics["queue_status"]["completed_batches"] == 1
        assert metrics["queue_status"]["failed_batches"] == 1
    
    def test_graph_update_operation_creation(self):
        """Test GraphUpdateOperation creation and defaults."""
        operation = GraphUpdateOperation(
            operation_id="test_op",
            update_type=UpdateType.SECTION_ADDED,
            target_id="section1",
            target_type="section",
            blueprint_id="blueprint1"
        )
        
        assert operation.operation_id == "test_op"
        assert operation.update_type == UpdateType.SECTION_ADDED
        assert operation.target_id == "section1"
        assert operation.target_type == "section"
        assert operation.blueprint_id == "blueprint1"
        assert operation.status == UpdateStatus.PENDING
        assert operation.retry_count == 0
        assert operation.max_retries == 3
        assert operation.created_at is not None
    
    def test_update_batch_creation(self):
        """Test UpdateBatch creation and defaults."""
        operations = [
            GraphUpdateOperation(
                operation_id="op1",
                update_type=UpdateType.SECTION_ADDED,
                target_id="section1",
                target_type="section",
                blueprint_id="blueprint1"
            ),
            GraphUpdateOperation(
                operation_id="op2",
                update_type=UpdateType.SECTION_UPDATED,
                target_id="section2",
                target_type="section",
                blueprint_id="blueprint1"
            )
        ]
        
        batch = UpdateBatch(
            batch_id="test_batch",
            blueprint_id="blueprint1",
            operations=operations
        )
        
        assert batch.batch_id == "test_batch"
        assert batch.blueprint_id == "blueprint1"
        assert len(batch.operations) == 2
        assert batch.status == UpdateStatus.PENDING
        assert batch.total_operations == 2
        assert batch.completed_operations == 0
        assert batch.failed_operations == 0
        assert batch.created_at is not None
    
    def test_update_type_enum_values(self):
        """Test UpdateType enum has expected values."""
        expected_types = [
            "blueprint_created", "blueprint_updated", "blueprint_deleted",
            "section_added", "section_updated", "section_deleted", "section_moved",
            "primitive_added", "primitive_updated", "primitive_deleted",
            "relationship_added", "relationship_updated", "relationship_deleted"
        ]
        
        for expected_type in expected_types:
            assert expected_type in [t.value for t in UpdateType]
    
    def test_update_status_enum_values(self):
        """Test UpdateStatus enum has expected values."""
        expected_statuses = [
            "pending", "in_progress", "completed", "failed", "rolled_back"
        ]
        
        for expected_status in expected_statuses:
            assert expected_status in [s.value for s in UpdateStatus]
    
    @pytest.mark.asyncio
    async def test_private_methods_return_empty_lists(self, update_service):
        """Test that private consistency check methods return empty lists by default."""
        orphaned = await update_service._find_orphaned_nodes()
        broken = await update_service._find_broken_relationships()
        circular = await update_service._find_circular_dependencies()
        missing = await update_service._find_missing_metadata()
        
        assert orphaned == []
        assert broken == []
        assert circular == []
        assert missing == []
    
    @pytest.mark.asyncio
    async def test_schedule_section_update_creates_unique_batch_ids(self, update_service):
        """Test that multiple section updates create unique batch IDs."""
        batch_id1 = await update_service.schedule_section_update(
            blueprint_id="blueprint1",
            section_id="section1",
            update_type=UpdateType.SECTION_ADDED
        )
        
        batch_id2 = await update_service.schedule_section_update(
            blueprint_id="blueprint1",
            section_id="section2",
            update_type=UpdateType.SECTION_ADDED
        )
        
        assert batch_id1 != batch_id2
        assert len(update_service.update_queue) == 2
    
    def test_performance_metrics_structure(self, update_service):
        """Test that performance metrics have the expected structure."""
        metrics = update_service.get_performance_metrics()
        
        # Check update_duration structure
        assert "average" in metrics["update_duration"]
        assert "min" in metrics["update_duration"]
        assert "max" in metrics["update_duration"]
        assert "count" in metrics["update_duration"]
        
        # Check consistency_check_duration structure
        assert "average" in metrics["consistency_check_duration"]
        assert "min" in metrics["consistency_check_duration"]
        assert "max" in metrics["consistency_check_duration"]
        assert "count" in metrics["consistency_check_duration"]
        
        # Check queue_status structure
        assert "pending_batches" in metrics["queue_status"]
        assert "processing_batches" in metrics["queue_status"]
        assert "completed_batches" in metrics["queue_status"]
        assert "failed_batches" in metrics["queue_status"]
