"""
Tests for Knowledge Graph Update Endpoints.

This module tests the knowledge graph update endpoints that integrate
the KnowledgeGraphUpdateService with the FastAPI router.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from datetime import datetime, timezone
from app.main import app
from app.services.knowledge_graph_update_service import UpdateType
from app.models.blueprint_centric import BlueprintSection, DifficultyLevel

client = TestClient(app)


class TestKnowledgeGraphUpdateEndpoints:
    """Test class for knowledge graph update endpoints."""
    
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
    
    @pytest.fixture
    def mock_services(self):
        """Mock the required services."""
        with patch('app.api.blueprint_lifecycle_endpoints.PineconeVectorStore') as mock_vector_store, \
             patch('app.api.blueprint_lifecycle_endpoints.KnowledgeGraphTraversal') as mock_traversal, \
             patch('app.api.blueprint_lifecycle_endpoints.KnowledgeGraphUpdateService') as mock_update_service:
            
            # Mock VectorStore
            mock_vector_store.return_value = MagicMock()
            
            # Mock KnowledgeGraphTraversal
            mock_traversal.return_value = MagicMock()
            
            # Mock KnowledgeGraphUpdateService
            mock_service_instance = MagicMock()
            mock_service_instance.schedule_section_update = AsyncMock(return_value="batch_123")
            mock_service_instance.check_graph_consistency = AsyncMock(return_value={
                "consistent": True,
                "issues": [],
                "warnings": [],
                "orphaned_nodes_count": 0,
                "broken_relationships_count": 0,
                "circular_dependencies_count": 0,
                "missing_metadata_count": 0,
                "check_duration": 0.5,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            mock_service_instance.get_performance_metrics.return_value = {
                "update_duration": {
                    "average": 1.5,
                    "min": 1.0,
                    "max": 2.0,
                    "count": 3
                },
                "consistency_check_duration": {
                    "average": 0.5,
                    "min": 0.3,
                    "max": 0.8,
                    "count": 2
                },
                "queue_status": {
                    "pending_batches": 1,
                    "processing_batches": 0,
                    "completed_batches": 2,
                    "failed_batches": 0
                }
            }
            mock_update_service.return_value = mock_service_instance
            
            yield {
                'vector_store': mock_vector_store,
                'traversal': mock_traversal,
                'update_service': mock_update_service,
                'service_instance': mock_service_instance
            }
    
    def test_trigger_knowledge_graph_update_section_added(self, mock_services):
        """Test triggering knowledge graph update for section addition."""
        response = client.post(
            "/api/v1/blueprints/test-blueprint-123/knowledge-graph/update",
            params={
                "update_type": UpdateType.SECTION_ADDED.value,
                "section_id": "section-456",
                "metadata": '{"reason": "new_content"}'
            },
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "Knowledge graph update scheduled successfully"
        assert data["batch_id"] == "batch_123"
        assert data["blueprint_id"] == "test-blueprint-123"
        assert data["update_type"] == UpdateType.SECTION_ADDED
        assert data["status"] == "scheduled"
        assert "timestamp" in data
        
        # Verify service was called correctly
        mock_services['service_instance'].schedule_section_update.assert_called_once()
        call_args = mock_services['service_instance'].schedule_section_update.call_args
        assert call_args[1]["blueprint_id"] == "test-blueprint-123"
        assert call_args[1]["section_id"] == "section-456"
        assert call_args[1]["update_type"] == UpdateType.SECTION_ADDED
    
    def test_trigger_knowledge_graph_update_section_updated(self, mock_services):
        """Test triggering knowledge graph update for section update."""
        response = client.post(
            "/api/v1/blueprints/test-blueprint-123/knowledge-graph/update",
            params={
                "update_type": UpdateType.SECTION_UPDATED.value,
                "section_id": "section-456"
            },
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["update_type"] == UpdateType.SECTION_UPDATED
        assert data["status"] == "scheduled"
    
    def test_trigger_knowledge_graph_update_section_deleted(self, mock_services):
        """Test triggering knowledge graph update for section deletion."""
        response = client.post(
            "/api/v1/blueprints/test-blueprint-123/knowledge-graph/update",
            params={
                "update_type": UpdateType.SECTION_DELETED.value,
                "section_id": "section-456"
            },
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["update_type"] == UpdateType.SECTION_DELETED
        assert data["status"] == "scheduled"
    
    def test_trigger_knowledge_graph_update_missing_section_id(self, mock_services):
        """Test that section ID is required for section-related updates."""
        response = client.post(
            "/api/v1/blueprints/test-blueprint-123/knowledge-graph/update",
            params={
                "update_type": UpdateType.SECTION_ADDED.value
                # Missing section_id
            },
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Section ID is required" in data["detail"]
    
    def test_trigger_knowledge_graph_update_blueprint_level_not_implemented(self, mock_services):
        """Test that blueprint-level updates are not yet implemented."""
        response = client.post(
            "/api/v1/blueprints/test-blueprint-123/knowledge-graph/update",
            params={
                "update_type": UpdateType.BLUEPRINT_CREATED.value
                # No section_id needed for blueprint-level updates
            },
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 501
        data = response.json()
        assert "not yet implemented" in data["detail"]
    
    def test_check_knowledge_graph_consistency_success(self, mock_services):
        """Test successful knowledge graph consistency check."""
        response = client.get(
            "/api/v1/blueprints/test-blueprint-123/knowledge-graph/consistency",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["blueprint_id"] == "test-blueprint-123"
        assert data["consistent"] is True
        assert data["issues"] == []
        assert data["warnings"] == []
        assert data["orphaned_nodes_count"] == 0
        assert data["broken_relationships_count"] == 0
        assert data["circular_dependencies_count"] == 0
        assert data["missing_metadata_count"] == 0
        assert "check_duration" in data
        assert "timestamp" in data
        
        # Verify service was called correctly
        mock_services['service_instance'].check_graph_consistency.assert_called_once_with("test-blueprint-123")
    
    def test_check_knowledge_graph_consistency_with_issues(self, mock_services):
        """Test knowledge graph consistency check that finds issues."""
        # Mock service to return issues
        mock_services['service_instance'].check_graph_consistency.return_value = {
            "consistent": False,
            "issues": ["Found 2 orphaned nodes", "Found 1 broken relationship"],
            "warnings": ["Found 1 potential circular dependency"],
            "orphaned_nodes_count": 2,
            "broken_relationships_count": 1,
            "circular_dependencies_count": 1,
            "missing_metadata_count": 0,
            "check_duration": 1.2,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        response = client.get(
            "/api/v1/blueprints/test-blueprint-123/knowledge-graph/consistency",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["consistent"] is False
        assert len(data["issues"]) == 2
        assert len(data["warnings"]) == 1
        assert data["orphaned_nodes_count"] == 2
        assert data["broken_relationships_count"] == 1
        assert data["circular_dependencies_count"] == 1
    
    def test_get_knowledge_graph_performance_success(self, mock_services):
        """Test successful retrieval of knowledge graph performance metrics."""
        response = client.get(
            "/api/v1/blueprints/test-blueprint-123/knowledge-graph/performance",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["blueprint_id"] == "test-blueprint-123"
        assert "update_duration" in data
        assert "consistency_check_duration" in data
        assert "queue_status" in data
        assert "timestamp" in data
        
        # Verify specific metrics structure
        update_duration = data["update_duration"]
        assert "average" in update_duration
        assert "min" in update_duration
        assert "max" in update_duration
        assert "count" in update_duration
        assert update_duration["average"] == 1.5
        assert update_duration["count"] == 3
        
        queue_status = data["queue_status"]
        assert queue_status["pending_batches"] == 1
        assert queue_status["pending_batches"] == 1
        assert queue_status["processing_batches"] == 0
        assert queue_status["completed_batches"] == 2
        assert queue_status["failed_batches"] == 0
        
        # Verify service was called correctly
        mock_services['service_instance'].get_performance_metrics.assert_called_once()
    
    def test_trigger_knowledge_graph_update_with_metadata(self, mock_services):
        """Test triggering update with additional metadata."""
        metadata = '{"reason": "content_update", "priority": "high"}'
        
        response = client.post(
            "/api/v1/blueprints/test-blueprint-123/knowledge-graph/update",
            params={
                "update_type": UpdateType.SECTION_UPDATED.value,
                "section_id": "section-456",
                "metadata": metadata
            },
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        
        # Verify metadata was passed to service
        call_args = mock_services['service_instance'].schedule_section_update.call_args
        # The endpoint parses JSON metadata into a dict, so we need to compare with the parsed version
        import json
        expected_metadata = json.loads(metadata)
        assert call_args[1]["metadata"] == expected_metadata
    
    def test_trigger_knowledge_graph_update_invalid_update_type(self, mock_services):
        """Test that invalid update types are handled gracefully."""
        response = client.post(
            "/api/v1/blueprints/test-blueprint-123/knowledge-graph/update",
            params={
                "update_type": "invalid_type",
                "section_id": "section-456"
            },
            headers={"Authorization": "Bearer test-token"}
        )
        
        # Should fail validation due to invalid enum value
        assert response.status_code == 422
    
    def test_endpoint_error_handling(self, mock_services):
        """Test that service errors are properly handled."""
        # Mock service to raise an exception
        mock_services['service_instance'].schedule_section_update.side_effect = Exception("Service error")
        
        response = client.post(
            "/api/v1/blueprints/test-blueprint-123/knowledge-graph/update",
            params={
                "update_type": UpdateType.SECTION_ADDED.value,
                "section_id": "section-456"
            },
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "Knowledge graph update failed" in data["detail"]
        assert "Service error" in data["detail"]
    
    def test_consistency_check_error_handling(self, mock_services):
        """Test that consistency check errors are properly handled."""
        # Mock service to raise an exception
        mock_services['service_instance'].check_graph_consistency.side_effect = Exception("Consistency check failed")
        
        response = client.get(
            "/api/v1/blueprints/test-blueprint-123/knowledge-graph/consistency",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "Consistency check failed" in data["detail"]
    
    def test_performance_metrics_error_handling(self, mock_services):
        """Test that performance metrics errors are properly handled."""
        # Mock service to raise an exception
        mock_services['service_instance'].get_performance_metrics.side_effect = Exception("Metrics retrieval failed")
        
        response = client.get(
            "/api/v1/blueprints/test-blueprint-123/knowledge-graph/performance",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "Performance metrics retrieval failed" in data["detail"]
    
    def test_endpoint_integration_with_real_services(self):
        """Test that endpoints can be called without mocking (integration test)."""
        # This test verifies the endpoints are properly registered and accessible
        # Note: This will fail if the required services aren't properly configured
        # but it's useful for catching basic integration issues
        
        try:
            # Test that the endpoint exists (should return 422 for missing required params)
            response = client.post(
                "/api/v1/blueprints/test-blueprint-123/knowledge-graph/update",
                headers={"Authorization": "Bearer test-token"}
            )
            # Should fail validation, not 404 (endpoint not found)
            assert response.status_code in [422, 500]  # 422 for validation, 500 for service errors
            
        except Exception as e:
            # If we get a service error, that's fine - it means the endpoint exists
            # but the services aren't configured for testing
            assert "Service error" in str(e) or "Import error" in str(e)
    
    def test_update_type_enum_values(self):
        """Test that all UpdateType enum values are valid."""
        valid_types = [
            "blueprint_created", "blueprint_updated", "blueprint_deleted",
            "section_added", "section_updated", "section_deleted", "section_moved",
            "primitive_added", "primitive_updated", "primitive_deleted",
            "relationship_added", "relationship_updated", "relationship_deleted"
        ]
        
        for valid_type in valid_types:
            response = client.post(
                "/api/v1/blueprints/test-blueprint-123/knowledge-graph/update",
                params={
                    "update_type": valid_type,
                    "section_id": "section-456"
                },
                headers={"Authorization": "Bearer test-token"}
            )
            # Should either succeed (200) or fail with 501 (not implemented) but not 422 (validation error)
            assert response.status_code in [200, 501]
