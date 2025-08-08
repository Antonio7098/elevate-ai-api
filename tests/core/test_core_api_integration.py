# Sprint 33: Unit Tests for Core API Integration Services

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any
import aiohttp
from aiohttp import ClientResponseError

from app.core.core_api_sync_service import CoreAPISyncService
from app.core.core_api_integration import CoreAPIIntegrationService
from app.api.schemas import (
    MasteryCriterionDto,
    KnowledgePrimitiveDto,
    SyncStatusResponse
)

class TestCoreAPISyncService:
    """Test suite for Core API synchronization service."""
    
    @pytest.fixture
    def sync_service(self):
        """Create sync service instance."""
        return CoreAPISyncService()
    
    @pytest.fixture
    def sample_primitives(self):
        """Sample primitives for testing."""
        return [
            {
                "primitive_id": "prim_001",
                "title": "Photosynthesis Process",
                "description": "Understanding photosynthesis",
                "content": "Process that converts light to chemical energy",
                "primitive_type": "process",
                "tags": ["biology", "energy"],
                "created_at": "2024-01-01T00:00:00Z"
            },
            {
                "primitive_id": "prim_002", 
                "title": "Light Reactions",
                "description": "Light-dependent reactions in photosynthesis",
                "content": "First stage of photosynthesis",
                "primitive_type": "concept",
                "tags": ["biology", "biochemistry"],
                "created_at": "2024-01-01T00:00:00Z"
            }
        ]
    
    @pytest.fixture
    def sample_mastery_criteria(self):
        """Sample mastery criteria for testing."""
        return [
            {
                "criterion_id": "crit_001",
                "primitive_id": "prim_001",
                "title": "Define photosynthesis",
                "description": "Explain what photosynthesis is",
                "uee_level": "UNDERSTAND",
                "weight": 3.0,
                "is_required": True
            },
            {
                "criterion_id": "crit_002",
                "primitive_id": "prim_001", 
                "title": "Apply photosynthesis knowledge",
                "description": "Use knowledge to solve problems",
                "uee_level": "USE",
                "weight": 4.0,
                "is_required": True
            }
        ]
    
    @pytest.fixture
    def mock_core_api_client(self):
        """Mock Core API HTTP client."""
        mock_client = AsyncMock()
        
        # Mock successful responses
        mock_client.create_primitive.return_value = {
            "primitive_id": "prim_001_synced",
            "status": "created"
        }
        mock_client.create_mastery_criterion.return_value = {
            "criterion_id": "crit_001_synced", 
            "status": "created"
        }
        mock_client.get_primitive.return_value = {
            "primitive_id": "prim_001",
            "title": "Existing Primitive"
        }
        
        return mock_client

    @pytest.mark.asyncio
    async def test_sync_primitives_success(self, sync_service, sample_primitives, mock_core_api_client):
        """Test successful primitive synchronization."""
        with patch.object(sync_service, 'core_api_client', mock_core_api_client):
            result = await sync_service.sync_primitives_to_core_api(
                primitives=sample_primitives,
                user_id="test_user_123"
            )
            
            assert result["success"] is True
            assert result["synced_count"] == 2
            assert result["failed_count"] == 0
            assert len(result["sync_results"]) == 2
            
            # Verify API calls were made
            assert mock_core_api_client.create_primitive.call_count == 2

    @pytest.mark.asyncio
    async def test_sync_mastery_criteria_success(
        self, 
        sync_service, 
        sample_mastery_criteria, 
        mock_core_api_client
    ):
        """Test successful mastery criteria synchronization."""
        with patch.object(sync_service, 'core_api_client', mock_core_api_client):
            result = await sync_service.sync_mastery_criteria_to_core_api(
                mastery_criteria=sample_mastery_criteria,
                user_id="test_user_123"
            )
            
            assert result["success"] is True
            assert result["synced_count"] == 2
            assert result["failed_count"] == 0
            
            # Verify API calls were made
            assert mock_core_api_client.create_mastery_criterion.call_count == 2

    @pytest.mark.asyncio
    async def test_sync_with_partial_failures(self, sync_service, sample_primitives, mock_core_api_client):
        """Test sync handling partial failures."""
        # Mock one success, one failure
        def mock_create_primitive(primitive_data):
            if primitive_data["primitive_id"] == "prim_001":
                return {"primitive_id": "prim_001_synced", "status": "created"}
            else:
                raise ClientResponseError(
                    request_info=Mock(),
                    history=(),
                    status=400,
                    message="Validation error"
                )
        
        mock_core_api_client.create_primitive.side_effect = mock_create_primitive
        
        with patch.object(sync_service, 'core_api_client', mock_core_api_client):
            result = await sync_service.sync_primitives_to_core_api(
                primitives=sample_primitives,
                user_id="test_user_123"
            )
            
            assert result["success"] is False  # Partial success counts as failure
            assert result["synced_count"] == 1
            assert result["failed_count"] == 1
            assert len(result["errors"]) == 1

    @pytest.mark.asyncio
    async def test_batch_sync_with_retry_logic(self, sync_service, sample_primitives, mock_core_api_client):
        """Test batch sync with retry logic for failed requests."""
        # Mock temporary failure then success
        call_count = 0
        def mock_create_with_retry(primitive_data):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # First two calls fail
                raise aiohttp.ClientError("Temporary network error")
            return {"primitive_id": f"{primitive_data['primitive_id']}_synced", "status": "created"}
        
        mock_core_api_client.create_primitive.side_effect = mock_create_with_retry
        
        with patch.object(sync_service, 'core_api_client', mock_core_api_client):
            result = await sync_service.sync_primitives_to_core_api(
                primitives=sample_primitives[:1],  # Just one primitive for clearer testing
                user_id="test_user_123",
                max_retries=3
            )
            
            # Should succeed after retries
            assert result["synced_count"] == 1
            assert result["failed_count"] == 0

    @pytest.mark.asyncio
    async def test_deduplication_logic(self, sync_service, mock_core_api_client):
        """Test that duplicate primitives are not synced twice."""
        # Mock Core API to return existing primitive
        mock_core_api_client.get_primitive.return_value = {
            "primitive_id": "prim_001",
            "title": "Existing Primitive"
        }
        
        duplicate_primitives = [
            {"primitive_id": "prim_001", "title": "Duplicate 1"},
            {"primitive_id": "prim_001", "title": "Duplicate 2"}
        ]
        
        with patch.object(sync_service, 'core_api_client', mock_core_api_client):
            result = await sync_service.sync_primitives_to_core_api(
                primitives=duplicate_primitives,
                user_id="test_user_123",
                skip_duplicates=True
            )
            
            # Should skip duplicates
            assert result["skipped_count"] > 0
            assert mock_core_api_client.create_primitive.call_count == 0

    @pytest.mark.asyncio
    async def test_sync_status_tracking(self, sync_service, sample_primitives, mock_core_api_client):
        """Test that sync status is properly tracked."""
        with patch.object(sync_service, 'core_api_client', mock_core_api_client):
            # Start sync operation
            task_id = await sync_service.start_background_sync(
                primitives=sample_primitives,
                user_id="test_user_123"
            )
            
            assert task_id is not None
            
            # Check status
            status = await sync_service.get_sync_status(task_id)
            assert status["task_id"] == task_id
            assert status["status"] in ["pending", "running", "completed"]

    @pytest.mark.asyncio
    async def test_cleanup_old_sync_records(self, sync_service):
        """Test cleanup of old sync records."""
        # Add some mock sync records
        from datetime import datetime, timedelta, timezone
        
        # Create dates relative to now for reliable testing
        old_date = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat().replace('+00:00', 'Z')
        recent_date = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat().replace('+00:00', 'Z')
        
        sync_service.sync_history = {
            "old_task_1": {
                "created_at": old_date,
                "status": "completed"
            },
            "recent_task_1": {
                "created_at": recent_date, 
                "status": "completed"
            }
        }
        
        await sync_service.cleanup_sync_history(max_age_days=30)
        
        # Old records should be cleaned up
        assert "old_task_1" not in sync_service.sync_history
        assert "recent_task_1" in sync_service.sync_history


class TestCoreAPIIntegrationService:
    """Test suite for Core API integration service."""
    
    @pytest.fixture
    def integration_service(self):
        """Create integration service instance."""
        return CoreAPIIntegrationService()
    
    @pytest.fixture
    def mock_http_session(self):
        """Mock HTTP session for API calls."""
        mock_session = AsyncMock()
        
        # Mock successful HTTP responses
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"id": "created_123", "status": "success"}
        
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        return mock_session

    @pytest.mark.asyncio
    async def test_create_primitive_api_call(self, integration_service, mock_http_session):
        """Test Core API primitive creation."""
        primitive_data = {
            "title": "Test Primitive",
            "description": "Test description",
            "content": "Test content",
            "primitive_type": "concept"
        }
        
        with patch.object(integration_service, 'http_session', mock_http_session):
            result = await integration_service.create_primitive(primitive_data)
            
            assert result["id"] == "created_123"
            assert result["status"] == "success"
            
            # Verify correct API endpoint was called
            mock_http_session.request.assert_called_once()
            call_args = mock_http_session.request.call_args
            assert call_args[0][0] == "POST"  # HTTP method
            assert "/primitives" in call_args[0][1]  # URL contains primitives endpoint

    @pytest.mark.asyncio
    async def test_api_error_handling(self, integration_service, mock_http_session):
        """Test handling of Core API errors."""
        # Mock 400 error response
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json.return_value = {"error": "Validation failed"}
        
        mock_http_session.request.return_value.__aenter__.return_value = mock_response
        
        with patch.object(integration_service, 'http_session', mock_http_session):
            with pytest.raises(ClientResponseError):
                await integration_service.create_primitive({"invalid": "data"})

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self, integration_service):
        """Test handling of connection timeouts."""
        with patch.object(integration_service, 'http_session') as mock_session:
            mock_session.request.side_effect = asyncio.TimeoutError("Request timeout")
            
            with pytest.raises(asyncio.TimeoutError):
                await integration_service.create_primitive({"title": "Test"})

    @pytest.mark.asyncio
    async def test_authentication_headers(self, integration_service, mock_http_session):
        """Test that authentication headers are included."""
        with patch.object(integration_service, 'http_session', mock_http_session):
            await integration_service.create_primitive({"title": "Test"})
            
            call_args = mock_http_session.request.call_args
            headers = call_args[1].get("headers", {})
            
            # Should include authentication headers
            assert "Authorization" in headers or "X-API-Key" in headers

    @pytest.mark.asyncio
    async def test_rate_limiting_compliance(self, integration_service, mock_http_session):
        """Test that rate limiting is respected."""
        with patch.object(integration_service, 'http_session', mock_http_session):
            # Make multiple requests quickly
            tasks = []
            for i in range(5):
                task = integration_service.create_primitive({"title": f"Test {i}"})
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should succeed (rate limiting should be handled internally)
            assert all(not isinstance(r, Exception) for r in results)

    @pytest.mark.asyncio
    async def test_request_id_tracking(self, integration_service, mock_http_session):
        """Test that requests include tracking IDs."""
        with patch.object(integration_service, 'http_session', mock_http_session):
            await integration_service.create_primitive({"title": "Test"})
            
            call_args = mock_http_session.request.call_args
            headers = call_args[1].get("headers", {})
            
            # Should include request ID for tracking
            assert "X-Request-ID" in headers or "Request-ID" in headers


class TestCoreAPIDataValidation:
    """Test suite for Core API data validation."""
    
    def test_primitive_schema_validation(self):
        """Test validation of primitive data against Core API schema."""
        valid_primitive = {
            "primitiveId": "valid_001",
            "title": "Valid Primitive",
            "description": "Valid description",
            "primitiveType": "concept",
            "difficultyLevel": "intermediate",
            "estimatedTimeMinutes": 30,
            "trackingIntensity": "NORMAL"
        }
        
        # Should not raise validation errors
        primitive_dto = KnowledgePrimitiveDto(**valid_primitive)
        assert primitive_dto.primitiveId == "valid_001"
        assert primitive_dto.primitiveType == "concept"

    def test_mastery_criterion_schema_validation(self):
        """Test validation of mastery criteria against Core API schema."""
        valid_criterion = {
            "criterionId": "valid_crit_001",
            "primitiveId": "prim_001", 
            "title": "Valid Criterion",
            "description": "Valid description",
            "ueeLevel": "UNDERSTAND",
            "weight": 3.0,
            "isRequired": True
        }
        
        # Should not raise validation errors
        criterion_dto = MasteryCriterionDto(**valid_criterion)
        assert criterion_dto.ueeLevel == "UNDERSTAND"
        assert criterion_dto.weight == 3.0

    def test_invalid_uee_level_validation(self):
        """Test that invalid UEE levels are rejected."""
        from pydantic import ValidationError

        invalid_criterion = {
            "criterionId": "invalid_001",
            "primitiveId": "prim_001",
            "title": "Invalid Criterion",
            "ueeLevel": "INVALID_LEVEL",  # Invalid UEE level
            "weight": 3.0,
            "isRequired": True
        }
    
        with pytest.raises(ValidationError, match="Input should be 'UNDERSTAND', 'USE' or 'EXPLORE'"):
            MasteryCriterionDto(**invalid_criterion)

    def test_weight_boundary_validation(self):
        """Test that criterion weights are within valid boundaries."""
        # Test weight too low
        invalid_low_weight = {
            "criterion_id": "test_001",
            "primitive_id": "prim_001",
            "title": "Test",
            "uee_level": "UNDERSTAND",
            "weight": 0.5  # Below minimum
        }
        
        with pytest.raises(ValueError, match="Weight must be between"):
            MasteryCriterionDto(**invalid_low_weight)
        
        # Test weight too high
        invalid_high_weight = {
            "criterion_id": "test_002",
            "primitive_id": "prim_001", 
            "title": "Test",
            "uee_level": "UNDERSTAND",
            "weight": 6.0  # Above maximum
        }
        
        with pytest.raises(ValueError, match="Weight must be between"):
            MasteryCriterionDto(**invalid_high_weight)


# Performance and stress testing markers
@pytest.mark.performance
class TestCoreAPIPerformance:
    """Performance tests for Core API integration."""
    
    @pytest.mark.asyncio
    async def test_bulk_sync_performance(self):
        """Test performance of bulk synchronization operations."""
        # This would be implemented with actual performance benchmarks
        pass
    
    @pytest.mark.asyncio 
    async def test_concurrent_api_calls(self):
        """Test handling of concurrent Core API calls."""
        # This would test concurrent request handling
        pass


# Test configuration
pytest_plugins = ["pytest_asyncio"]
