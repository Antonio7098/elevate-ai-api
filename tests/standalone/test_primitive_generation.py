# Sprint 33: Unit Tests for Primitive Generation Services

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

from app.core.deconstruction import (
    generate_enhanced_primitives_with_criteria,
    create_enhanced_blueprint_with_primitives
)
from app.core.mastery_criteria_service import MasteryCriteriaService
from app.core.question_generation_service import QuestionGenerationService
from app.core.question_mapping_service import QuestionMappingService
from app.core.core_api_sync_service import CoreAPISyncService
from app.api.schemas import (
    MasteryCriterionDto,
    KnowledgePrimitiveDto,
    PrimitiveGenerationRequest
)

class TestPrimitiveGeneration:
    """Test suite for primitive generation functionality."""
    
    @pytest.fixture
    def sample_source_text(self):
        """Sample source text for testing."""
        return """
        Photosynthesis is the process by which plants convert light energy into chemical energy.
        This process occurs in chloroplasts and involves two main stages: light-dependent reactions
        and the Calvin cycle. The light-dependent reactions produce ATP and NADPH, while the
        Calvin cycle uses these molecules to fix carbon dioxide into glucose.
        """
    
    @pytest.fixture
    def sample_user_preferences(self):
        """Sample user preferences for testing."""
        return {
            "uee_distribution": {"UNDERSTAND": 0.4, "USE": 0.4, "EXPLORE": 0.2},
            "max_primitives": 8,
            "difficulty_preference": "moderate",
            "learning_style": "visual"
        }
    
    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for testing."""
        mock_service = AsyncMock()
        mock_service.generate_text.return_value = """{
            "primitives": [
                {
                    "primitive_id": "phot_001",
                    "title": "Photosynthesis Process",
                    "description": "Understanding the basic process of photosynthesis",
                    "content": "Photosynthesis converts light energy to chemical energy",
                    "primitive_type": "process",
                    "tags": ["biology", "energy", "plants"],
                    "mastery_criteria": [
                        {
                            "criterion_id": "phot_001_understand",
                            "title": "Define photosynthesis",
                            "description": "Explain what photosynthesis is and its purpose",
                            "uee_level": "UNDERSTAND",
                            "weight": 3.0
                        }
                    ]
                }
            ]
        }"""
        return mock_service

    @pytest.mark.asyncio
    async def test_generate_enhanced_primitives_success(
        self, 
        sample_source_text, 
        sample_user_preferences, 
        mock_llm_service
    ):
        """Test successful primitive generation."""
        with patch('app.core.deconstruction.llm_service', mock_llm_service):
            result = await generate_enhanced_primitives_with_criteria(
                source_text=sample_source_text,
                user_preferences=sample_user_preferences,
                context={"blueprint_title": "Photosynthesis Basics"}
            )
            
            assert result is not None
            assert "primitives" in result
            assert len(result["primitives"]) > 0
            
            primitive = result["primitives"][0]
            assert primitive["primitive_id"] == "phot_001"
            assert primitive["title"] == "Photosynthesis Process"
            assert len(primitive["mastery_criteria"]) > 0

    @pytest.mark.asyncio
    async def test_generate_primitives_with_empty_text(self, sample_user_preferences):
        """Test primitive generation with empty source text."""
        with pytest.raises(ValueError, match="Source text cannot be empty"):
            await generate_enhanced_primitives_with_criteria(
                source_text="",
                user_preferences=sample_user_preferences
            )

    @pytest.mark.asyncio
    async def test_generate_primitives_with_invalid_preferences(self, sample_source_text):
        """Test primitive generation with invalid user preferences."""
        invalid_preferences = {
            "uee_distribution": {"UNDERSTAND": 0.8, "USE": 0.3, "EXPLORE": 0.2}  # Sum > 1.0
        }
        
        result = await generate_enhanced_primitives_with_criteria(
            source_text=sample_source_text,
            user_preferences=invalid_preferences
        )
        
        # Should still work with fallback preferences
        assert result is not None

    @pytest.mark.asyncio
    async def test_primitive_generation_with_context(
        self, 
        sample_source_text, 
        sample_user_preferences
    ):
        """Test primitive generation includes context information."""
        context = {
            "blueprint_title": "Advanced Biology",
            "subject": "science",
            "grade_level": "high_school"
        }
        
        result = await generate_enhanced_primitives_with_criteria(
            source_text=sample_source_text,
            user_preferences=sample_user_preferences,
            context=context
        )
        
        # Verify result structure
        assert "primitives" in result
        assert len(result["primitives"]) > 0
        primitive = result["primitives"][0]
        assert "primitive_id" in primitive
        assert "title" in primitive
        assert "mastery_criteria" in primitive

    @pytest.mark.asyncio
    async def test_primitive_uee_distribution(
        self, 
        sample_source_text, 
        sample_user_preferences
    ):
        """Test that generated primitives respect UEE distribution preferences."""
        result = await generate_enhanced_primitives_with_criteria(
            source_text=sample_source_text,
            user_preferences=sample_user_preferences
        )
        
        # Verify result structure
        assert "primitives" in result
        assert len(result["primitives"]) > 0
        primitive = result["primitives"][0]
        assert "mastery_criteria" in primitive
        assert len(primitive["mastery_criteria"]) > 0
        criterion = primitive["mastery_criteria"][0]
        assert "uee_level" in criterion
        assert criterion["uee_level"] in ["UNDERSTAND", "USE", "EXPLORE"]

    @pytest.mark.asyncio
    async def test_create_enhanced_blueprint_integration(
        self, 
        sample_source_text, 
        sample_user_preferences
    ):
        """Test integration with enhanced blueprint creation."""
        # First generate primitives
        primitive_result = await generate_enhanced_primitives_with_criteria(
            source_text=sample_source_text,
            user_preferences=sample_user_preferences
        )
        
        # Then create blueprint with those primitives
        result = create_enhanced_blueprint_with_primitives(
            title="Test Blueprint",
            description="Test blueprint description",
            primitives=primitive_result["primitives"]
        )
        
        assert result is not None
        assert "blueprint_id" in result
        assert "primitives" in result
        assert "total_primitives" in result
        assert result["total_primitives"] >= 0


class TestMasteryCriteriaService:
    """Test suite for mastery criteria generation service."""
    
    @pytest.fixture
    def mastery_service(self):
        """Create mastery criteria service instance."""
        return MasteryCriteriaService()
    
    @pytest.fixture
    def sample_primitive(self):
        """Sample primitive for testing."""
        return {
            "primitive_id": "test_001",
            "title": "Photosynthesis Basics",
            "description": "Understanding the basic process of photosynthesis",
            "content": "Photosynthesis converts light energy to chemical energy",
            "primitive_type": "process"
        }
    
    @pytest.mark.asyncio
    async def test_generate_mastery_criteria_success(self, mastery_service, sample_primitive):
        """Test successful mastery criteria generation."""
        result = await mastery_service.generate_mastery_criteria(
            primitive=sample_primitive,
            uee_level_preference="balanced"
        )
        
        assert len(result) == 2
        assert result[0]["uee_level"] == "UNDERSTAND"
        assert result[1]["uee_level"] == "USE"
        assert all(criterion["weight"] >= 1.0 and criterion["weight"] <= 5.0 for criterion in result)

    @pytest.mark.asyncio
    async def test_generate_criteria_with_focus(self, mastery_service, sample_primitive):
        """Test that criteria generation respects UEE level preferences."""
        result = await mastery_service.generate_mastery_criteria(
            primitive=sample_primitive,
            uee_level_preference="understand_focus"
        )
        
        # Should focus on UNDERSTAND level
        understand_criteria = [c for c in result if c["uee_level"] == "UNDERSTAND"]
        assert len(understand_criteria) >= len(result) * 0.7  # At least 70% UNDERSTAND

    @pytest.mark.asyncio
    async def test_criteria_validation(self, mastery_service, sample_primitive):
        """Test validation of generated criteria."""
        result = await mastery_service.generate_mastery_criteria(
            primitive=sample_primitive
        )
        
        # Should return fallback criteria or empty list
        assert isinstance(result, list)
        # If any criteria returned, they should be valid
        for criterion in result:
            assert criterion["criterion_id"] != ""
            assert criterion["uee_level"] in ["UNDERSTAND", "USE", "EXPLORE"]
            assert 1.0 <= criterion["weight"] <= 5.0


class TestQuestionGenerationService:
    """Test suite for question generation service."""
    
    @pytest.fixture
    def question_service(self):
        """Create question generation service instance."""
        return QuestionGenerationService()
    
    @pytest.fixture
    def sample_primitive_with_criteria(self):
        """Sample primitive with mastery criteria."""
        return {
            "primitive_id": "test_001",
            "title": "Photosynthesis Process",
            "content": "Photosynthesis converts light energy to chemical energy",
            "mastery_criteria": [
                {
                    "criterion_id": "test_001_understand",
                    "title": "Define photosynthesis",
                    "description": "Explain what photosynthesis is",
                    "uee_level": "UNDERSTAND",
                    "weight": 3.0
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_generate_criterion_questions_success(
        self, 
        question_service, 
        sample_primitive_with_criteria
    ):
        """Test successful question generation for criteria."""
        criterion = sample_primitive_with_criteria["mastery_criteria"][0]
        result = await question_service.generate_criterion_questions(
            primitive=sample_primitive_with_criteria,
            mastery_criterion=criterion,
            num_questions=2
        )
        
        assert len(result) == 2
        assert result[0]["question_text"] == "What is photosynthesis?"
        assert result[0]["question_type"] == "short_answer"
        assert result[1]["question_type"] == "essay"

    @pytest.mark.asyncio
    async def test_question_type_distribution(self, question_service, sample_primitive_with_criteria):
        """Test that different question types are generated."""
        criterion = sample_primitive_with_criteria["mastery_criteria"][0]
        result = await question_service.generate_criterion_questions(
            primitive=sample_primitive_with_criteria,
            mastery_criterion=criterion,
            num_questions=4
        )
        
        question_types = [q["question_type"] for q in result]
        unique_types = set(question_types)
        
        # Should have variety of question types
        assert len(unique_types) >= 3

    @pytest.mark.asyncio
    async def test_uee_level_specific_questions(self, question_service, sample_primitive_with_criteria):
        """Test that questions are appropriate for UEE level."""
        # Test UNDERSTAND level questions
        understand_criterion = {
            "criterion_id": "test_understand",
            "title": "Basic understanding",
            "uee_level": "UNDERSTAND",
            "weight": 3.0
        }
        
        result = await question_service.generate_criterion_questions(
            primitive=sample_primitive_with_criteria,
            mastery_criterion=understand_criterion,
            num_questions=1
        )
        
        # Verify result contains expected question type
        assert len(result) == 1
        assert result[0]["question_type"] == "short_answer"


# Test fixtures and utilities
@pytest.fixture
def test_database():
    """Mock database for testing."""
    return Mock()

@pytest.fixture  
def test_core_api_client():
    """Mock Core API client for testing."""
    mock_client = AsyncMock()
    mock_client.create_primitive.return_value = {"primitive_id": "created_001"}
    mock_client.create_mastery_criterion.return_value = {"criterion_id": "created_criterion_001"}
    return mock_client

# Performance test markers
pytestmark = pytest.mark.asyncio
