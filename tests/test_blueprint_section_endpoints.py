"""
Tests for Blueprint Section Endpoints.

This module tests the new blueprint section endpoints that provide
CRUD operations and hierarchy management for blueprint sections.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from app.api.schemas import (
    BlueprintSectionRequest,
    BlueprintSectionResponse,
    BlueprintSectionTreeResponse,
    SectionMoveRequest,
    SectionReorderRequest,
    SectionContentRequest,
    SectionContentResponse,
    SectionStatsResponse,
    BlueprintSectionSyncRequest,
    BlueprintSectionSyncResponse
)
from app.models.blueprint_centric import BlueprintSection, DifficultyLevel
from app.services.blueprint_section_service import BlueprintSectionService


class TestBlueprintSectionEndpoints:
    """Test class for blueprint section endpoints."""
    
    @pytest.fixture
    def mock_section_service(self):
        """Mock BlueprintSectionService for testing."""
        with patch('app.api.blueprint_lifecycle_endpoints.BlueprintSectionService') as mock:
            service_instance = mock.return_value
            # Setup async methods
            service_instance.create_section = AsyncMock()
            service_instance.get_section_tree = AsyncMock()
            service_instance.move_section = AsyncMock()
            service_instance.reorder_sections = AsyncMock()
            service_instance.get_section_content = AsyncMock()
            service_instance.get_section_stats = AsyncMock()
            yield service_instance
    
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
    def sample_section_request(self):
        """Sample BlueprintSectionRequest for testing."""
        return BlueprintSectionRequest(
            title="New Section",
            description="A new section",
            content="New content",
            order_index=2,
            parent_section_id=None,
            difficulty_level="intermediate",
            estimated_time_minutes=45
        )
    
    @pytest.mark.asyncio
    async def test_create_blueprint_section_success(self, mock_section_service, sample_section, sample_section_request):
        """Test successful creation of a blueprint section."""
        # Setup mock
        mock_section_service.create_section.return_value = sample_section
        
        # Mock the endpoint function
        from app.api.blueprint_lifecycle_endpoints import create_blueprint_section
        
        # Test the endpoint
        result = await create_blueprint_section(
            blueprint_id="test-blueprint-123",
            section_data=sample_section_request,
            user_id="test-user-123"
        )
        
        # Verify service was called correctly
        mock_section_service.create_section.assert_called_once_with(
            blueprint_id="test-blueprint-123",
            title="New Section",
            description="A new section",
            content="New content",
            order_index=2,
            parent_section_id=None,
            difficulty_level="intermediate",
            estimated_time_minutes=45
        )
        
        # Verify response format
        assert isinstance(result, BlueprintSectionResponse)
        assert result.id == 1
        assert result.title == "Test Section"
        assert result.blueprint_id == "test-blueprint-123"
    
    @pytest.mark.asyncio
    async def test_get_blueprint_sections_success(self, mock_section_service, sample_section):
        """Test successful retrieval of blueprint sections."""
        # Setup mock
        mock_section_service.get_section_tree.return_value = [sample_section]
        
        # Mock the endpoint function
        from app.api.blueprint_lifecycle_endpoints import get_blueprint_sections
        
        # Test the endpoint
        result = await get_blueprint_sections(blueprint_id="test-blueprint-123")
        
        # Verify service was called correctly
        mock_section_service.get_section_tree.assert_called_once_with("test-blueprint-123")
        
        # Verify response format
        assert isinstance(result, BlueprintSectionTreeResponse)
        assert result.blueprint_id == "test-blueprint-123"
        assert len(result.sections) == 1
        assert result.total_sections == 1
        assert result.max_depth == 0
    
    @pytest.mark.asyncio
    async def test_move_blueprint_section_success(self, mock_section_service):
        """Test successful moving of a blueprint section."""
        # Setup mock
        mock_section_service.move_section.return_value = {"success": True}
        
        # Mock the endpoint function
        from app.api.blueprint_lifecycle_endpoints import move_blueprint_section
        
        # Test data
        move_data = SectionMoveRequest(
            section_id=1,
            new_parent_id=2,
            new_order_index=3
        )
        
        # Test the endpoint
        result = await move_blueprint_section(
            blueprint_id="test-blueprint-123",
            section_id=1,
            move_data=move_data
        )
        
        # Verify service was called correctly
        mock_section_service.move_section.assert_called_once_with(
            section_id=1,
            new_parent_id=2,
            new_order_index=3
        )
        
        # Verify response format
        assert result["success"] is True
        assert result["section_id"] == 1
        assert result["blueprint_id"] == "test-blueprint-123"
    
    @pytest.mark.asyncio
    async def test_reorder_blueprint_sections_success(self, mock_section_service):
        """Test successful reordering of blueprint sections."""
        # Setup mock
        mock_section_service.reorder_sections.return_value = {"success": True}
        
        # Mock the endpoint function
        from app.api.blueprint_lifecycle_endpoints import reorder_blueprint_sections
        
        # Test data
        reorder_data = SectionReorderRequest(
            section_orders=[
                {"section_id": 1, "order_index": 1},
                {"section_id": 2, "order_index": 2}
            ]
        )
        
        # Test the endpoint
        result = await reorder_blueprint_sections(
            blueprint_id="test-blueprint-123",
            reorder_data=reorder_data
        )
        
        # Verify service was called correctly
        mock_section_service.reorder_sections.assert_called_once_with([
            {"section_id": 1, "order_index": 1},
            {"section_id": 2, "order_index": 2}
        ])
        
        # Verify response format
        assert result["success"] is True
        assert result["blueprint_id"] == "test-blueprint-123"
        assert result["sections_reordered"] == 2
    
    @pytest.mark.asyncio
    async def test_get_section_content_success(self, mock_section_service, sample_section):
        """Test successful retrieval of section content."""
        # Setup mock
        mock_section_service.get_section_content.return_value = {
            "section": sample_section,
            "primitives": [],
            "mastery_criteria": [],
            "content_summary": "Test summary",
            "learning_progress": {},
            "related_sections": []
        }
        
        # Mock the endpoint function
        from app.api.blueprint_lifecycle_endpoints import get_section_content
        
        # Test the endpoint
        result = await get_section_content(
            blueprint_id="test-blueprint-123",
            section_id=1,
            include_metadata=True,
            include_primitives=True,
            include_criteria=True
        )
        
        # Verify service was called correctly
        mock_section_service.get_section_content.assert_called_once_with(
            section_id=1,
            include_metadata=True,
            include_primitives=True,
            include_criteria=True
        )
        
        # Verify response format
        assert isinstance(result, SectionContentResponse)
        assert result.section.id == 1
        assert result.content_summary == "Test summary"
    
    @pytest.mark.asyncio
    async def test_get_section_stats_success(self, mock_section_service):
        """Test successful retrieval of section statistics."""
        # Setup mock
        mock_section_service.get_section_stats.return_value = {
            "total_primitives": 5,
            "total_criteria": 3,
            "difficulty_distribution": {"intermediate": 3, "advanced": 2},
            "uue_stage_distribution": {"understand": 2, "use": 2, "explore": 1},
            "estimated_completion_time": 120
        }
        
        # Mock the endpoint function
        from app.api.blueprint_lifecycle_endpoints import get_section_stats
        
        # Test the endpoint
        result = await get_section_stats(
            blueprint_id="test-blueprint-123",
            section_id=1
        )
        
        # Verify service was called correctly
        mock_section_service.get_section_stats.assert_called_once_with(1)
        
        # Verify response format
        assert isinstance(result, SectionStatsResponse)
        assert result.section_id == 1
        assert result.total_primitives == 5
        assert result.total_criteria == 3
        assert result.estimated_completion_time == 120
    
    def test_blueprint_section_request_validation(self):
        """Test validation of BlueprintSectionRequest."""
        # Test valid request
        valid_request = BlueprintSectionRequest(
            title="Valid Section",
            description="Valid description",
            difficulty_level="intermediate"
        )
        assert valid_request.title == "Valid Section"
        assert valid_request.difficulty_level == "intermediate"
        
        # Test invalid title
        with pytest.raises(ValueError, match="Section title cannot be empty"):
            BlueprintSectionRequest(title="", description="Description")
        
        # Test invalid difficulty level
        with pytest.raises(ValueError, match="Difficulty level must be one of"):
            BlueprintSectionRequest(title="Title", difficulty_level="invalid")
    
    def test_section_move_request_validation(self):
        """Test validation of SectionMoveRequest."""
        # Test valid request
        valid_request = SectionMoveRequest(
            section_id=1,
            new_parent_id=2,
            new_order_index=3
        )
        assert valid_request.section_id == 1
        assert valid_request.new_parent_id == 2
        assert valid_request.new_order_index == 3
        
        # Test invalid section ID
        with pytest.raises(ValueError, match="Section ID must be a positive integer"):
            SectionMoveRequest(section_id=0)
    
    def test_section_reorder_request_validation(self):
        """Test validation of SectionReorderRequest."""
        # Test valid request
        valid_request = SectionReorderRequest(
            section_orders=[
                {"section_id": 1, "order_index": 1},
                {"section_id": 2, "order_index": 2}
            ]
        )
        assert len(valid_request.section_orders) == 2
        
        # Test empty orders
        with pytest.raises(ValueError, match="Section orders cannot be empty"):
            SectionReorderRequest(section_orders=[])
        
        # Test invalid order format
        with pytest.raises(ValueError, match="Each item must contain section_id and order_index"):
            SectionReorderRequest(section_orders=[{"section_id": 1}])
    
    def test_section_content_request_validation(self):
        """Test validation of SectionContentRequest."""
        # Test valid request
        valid_request = SectionContentRequest(
            section_id=1,
            include_metadata=True,
            include_primitives=False,
            include_criteria=True
        )
        assert valid_request.section_id == 1
        assert valid_request.include_metadata is True
        assert valid_request.include_primitives is False
        assert valid_request.include_criteria is True
        
        # Test invalid section ID
        with pytest.raises(ValueError, match="Section ID must be a positive integer"):
            SectionContentRequest(section_id=0)
