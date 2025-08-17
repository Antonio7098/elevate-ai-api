"""
Tests for the Blueprint Editing Service

Tests the comprehensive blueprint editing capabilities including
blueprints, primitives, mastery criteria, and questions.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from app.core.blueprint_editing_service import BlueprintEditingService
from app.models.blueprint_editing_models import (
    BlueprintEditingRequest, PrimitiveEditingRequest, 
    MasteryCriterionEditingRequest, QuestionEditingRequest
)


class TestBlueprintEditingService:
    """Test cases for the BlueprintEditingService."""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        mock_service = Mock()
        mock_service.call_llm = AsyncMock(return_value='{"test": "response"}')
        return mock_service
    
    @pytest.fixture
    def blueprint_service(self, mock_llm_service):
        """Create a BlueprintEditingService instance with mocked dependencies."""
        return BlueprintEditingService(mock_llm_service)
    
    @pytest.fixture
    def sample_blueprint_request(self):
        """Create a sample blueprint editing request."""
        return BlueprintEditingRequest(
            blueprint_id=1,
            edit_type="improve_clarity",
            edit_instruction="Make the content clearer and more understandable",
            preserve_original_structure=True,
            include_reasoning=True
        )
    
    @pytest.fixture
    def sample_primitive_request(self):
        """Create a sample primitive editing request."""
        return PrimitiveEditingRequest(
            primitive_id=1,
            edit_type="improve_clarity",
            edit_instruction="Simplify the concept definition",
            preserve_original_structure=True,
            include_reasoning=True
        )
    
    @pytest.fixture
    def sample_criterion_request(self):
        """Create a sample mastery criterion editing request."""
        return MasteryCriterionEditingRequest(
            criterion_id=1,
            edit_type="improve_clarity",
            edit_instruction="Make the assessment criteria clearer",
            preserve_original_structure=True,
            include_reasoning=True
        )
    
    @pytest.fixture
    def sample_question_request(self):
        """Create a sample question editing request."""
        return QuestionEditingRequest(
            question_id=1,
            edit_type="improve_clarity",
            edit_instruction="Make the question clearer and more focused",
            preserve_original_structure=True,
            include_reasoning=True
        )
    
    def test_service_initialization(self, mock_llm_service):
        """Test that the service initializes correctly."""
        service = BlueprintEditingService(mock_llm_service)
        assert service.llm_service == mock_llm_service
        assert service.granular_editing_service is not None
    
    @pytest.mark.asyncio
    async def test_edit_blueprint_agentically(self, blueprint_service, sample_blueprint_request):
        """Test blueprint editing functionality."""
        response = await blueprint_service.edit_blueprint_agentically(sample_blueprint_request)
        
        assert response is not None
        # The response should be a BlueprintEditingResponse object
        assert hasattr(response, 'success')
        assert hasattr(response, 'message')
    
    @pytest.mark.asyncio
    async def test_edit_primitive_agentically(self, blueprint_service, sample_primitive_request):
        """Test primitive editing functionality."""
        response = await blueprint_service.edit_primitive_agentically(sample_primitive_request)
        
        assert response is not None
        # The response should be a PrimitiveEditingResponse object
        assert hasattr(response, 'success')
        assert hasattr(response, 'message')
    
    @pytest.mark.asyncio
    async def test_edit_mastery_criterion_agentically(self, blueprint_service, sample_criterion_request):
        """Test mastery criterion editing functionality."""
        response = await blueprint_service.edit_mastery_criterion_agentically(sample_criterion_request)
        
        assert response is not None
        # The response should be a MasteryCriterionEditingResponse object
        assert hasattr(response, 'success')
        assert hasattr(response, 'message')
    
    @pytest.mark.asyncio
    async def test_edit_question_agentically(self, blueprint_service, sample_question_request):
        """Test question editing functionality."""
        response = await blueprint_service.edit_question_agentically(sample_question_request)
        
        assert response is not None
        # The response should be a QuestionEditingResponse object
        assert hasattr(response, 'success')
        assert hasattr(response, 'message')
    
    @pytest.mark.asyncio
    async def test_get_blueprint_editing_suggestions(self, blueprint_service):
        """Test getting blueprint editing suggestions."""
        response = await blueprint_service.get_blueprint_editing_suggestions(
            blueprint_id=1,
            include_structure=True,
            include_content=True,
            include_relationships=True
        )
        
        assert response is not None
        assert hasattr(response, 'success')
        assert hasattr(response, 'suggestions')
    
    @pytest.mark.asyncio
    async def test_get_primitive_editing_suggestions(self, blueprint_service):
        """Test getting primitive editing suggestions."""
        response = await blueprint_service.get_primitive_editing_suggestions(
            primitive_id=1,
            include_clarity=True,
            include_complexity=True,
            include_relationships=True
        )
        
        assert response is not None
        assert hasattr(response, 'success')
        assert hasattr(response, 'suggestions')
    
    @pytest.mark.asyncio
    async def test_get_mastery_criterion_editing_suggestions(self, blueprint_service):
        """Test getting mastery criterion editing suggestions."""
        response = await blueprint_service.get_mastery_criterion_editing_suggestions(
            criterion_id=1,
            include_clarity=True,
            include_difficulty=True,
            include_assessment=True
        )
        
        assert response is not None
        assert hasattr(response, 'success')
        assert hasattr(response, 'suggestions')
    
    @pytest.mark.asyncio
    async def test_get_question_editing_suggestions(self, blueprint_service):
        """Test getting question editing suggestions."""
        response = await blueprint_service.get_question_editing_suggestions(
            question_id=1,
            include_clarity=True,
            include_difficulty=True,
            include_quality=True
        )
        
        assert response is not None
        assert hasattr(response, 'success')
        assert hasattr(response, 'suggestions')
    
    def test_is_granular_blueprint_edit_request(self, blueprint_service):
        """Test granular edit request detection."""
        # Test granular edit types
        assert blueprint_service._is_granular_blueprint_edit_request(
            BlueprintEditingRequest(
                blueprint_id=1,
                edit_type="edit_section",
                edit_instruction="test"
            )
        ) is True
        
        # Test non-granular edit types
        assert blueprint_service._is_granular_blueprint_edit_request(
            BlueprintEditingRequest(
                blueprint_id=1,
                edit_type="improve_clarity",
                edit_instruction="test"
            )
        ) is False


if __name__ == "__main__":
    pytest.main([__file__])
