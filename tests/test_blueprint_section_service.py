"""
Comprehensive tests for BlueprintSectionService

This test suite covers all service methods including section CRUD operations,
hierarchy management, content aggregation, and statistics.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.services.blueprint_section_service import BlueprintSectionService
from app.models.blueprint_centric import (
    BlueprintSection, DifficultyLevel
)


class TestBlueprintSectionService:
    """Test BlueprintSectionService class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = BlueprintSectionService()
        
        # Create test data
        self.test_section_data = {
            "title": "Test Section",
            "description": "Test description",
            "blueprint_id": 1,
            "user_id": 1,
            "difficulty": DifficultyLevel.INTERMEDIATE,
            "estimated_time_minutes": 60
        }
    
    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service is not None
        assert hasattr(self.service, 'logger')
        assert self.service.logger is not None
        assert hasattr(self.service, 'sections')
        assert isinstance(self.service.sections, dict)
    
    @pytest.mark.asyncio
    async def test_create_section_success(self):
        """Test successful section creation."""
        result = await self.service.create_section(self.test_section_data)
        
        assert isinstance(result, BlueprintSection)
        assert result.title == "Test Section"
        assert result.description == "Test description"
        assert result.blueprint_id == 1
        assert result.user_id == 1
        assert result.difficulty == DifficultyLevel.INTERMEDIATE
        assert result.estimated_time_minutes == 60
        assert result.depth == 0
        assert result.order_index == 0
        assert result.id is not None
        assert result.created_at is not None
        assert result.updated_at is not None
        
        # Check if section was stored
        assert result.id in self.service.sections
        assert len(self.service.sections) == 1
    
    @pytest.mark.asyncio
    async def test_create_section_with_parent(self):
        """Test section creation with parent section."""
        # Create parent section first
        parent_data = self.test_section_data.copy()
        parent_data["title"] = "Parent Section"
        parent = await self.service.create_section(parent_data)
        
        # Create child section
        child_data = self.test_section_data.copy()
        child_data["title"] = "Child Section"
        child_data["parent_section_id"] = parent.id
        
        child = await self.service.create_section(child_data)
        
        assert child.parent_section_id == parent.id
        assert child.depth == 1
        assert child.order_index == 0
        
        # Check parent's children list
        assert len(parent.children) == 1
        assert parent.children[0].id == child.id
    
    @pytest.mark.asyncio
    async def test_get_section_tree_empty(self):
        """Test getting section tree for empty blueprint."""
        tree = await self.service.get_section_tree("1")
        
        assert tree["blueprint_id"] == "1"
        assert tree["sections"] == []
        assert tree["total_sections"] == 0
        assert tree["max_depth"] == 0
    
    @pytest.mark.asyncio
    async def test_get_section_tree_with_sections(self):
        """Test getting section tree with sections."""
        # Create parent section first
        parent = await self.service.create_section(self.test_section_data)
        
        # Create child section with correct parent reference
        child_data = self.test_section_data.copy()
        child_data["title"] = "Child Section"
        child_data["parent_section_id"] = parent.id
        await self.service.create_section(child_data)
        
        tree = await self.service.get_section_tree("1")
        
        assert tree["blueprint_id"] == "1"
        assert tree["total_sections"] == 2
        assert tree["max_depth"] == 1
        assert len(tree["sections"]) == 1  # Root sections
        assert len(tree["sections"][0].children) == 1  # Child sections
    
    @pytest.mark.asyncio
    async def test_move_section_success(self):
        """Test successful section movement."""
        # Create sections
        section1 = await self.service.create_section(self.test_section_data)
        
        section2_data = self.test_section_data.copy()
        section2_data["title"] = "Section 2"
        section2 = await self.service.create_section(section2_data)
        
        # Move section2 to be child of section1
        result = await self.service.move_section(section2.id, section1.id)
        
        assert result.parent_section_id == section1.id
        assert result.depth == 1
        assert len(section1.children) == 1
        assert section1.children[0].id == section2.id
    
    @pytest.mark.asyncio
    async def test_get_section_content(self):
        """Test getting section content."""
        section = await self.service.create_section(self.test_section_data)
        
        content = await self.service.get_section_content(section.id)
        
        assert content["section_id"] == section.id
        assert content["title"] == "Test Section"
        assert content["description"] == "Test description"
        assert "content_count" in content
        assert "mastery_criteria_count" in content
        assert "questions_count" in content
        assert "primitives_count" in content
        assert "children_content" in content
    
    @pytest.mark.asyncio
    async def test_get_section_stats(self):
        """Test getting section statistics."""
        section = await self.service.create_section(self.test_section_data)
        
        stats = await self.service.get_section_stats(section.id)
        
        assert stats["section_id"] == section.id
        assert stats["title"] == "Test Section"
        assert "total_criteria" in stats
        assert "mastered_criteria" in stats
        assert "mastery_progress" in stats
        assert "average_difficulty" in stats
        assert "estimated_completion_time" in stats
        assert "children_count" in stats
        assert "depth" in stats
        assert "last_activity" in stats
