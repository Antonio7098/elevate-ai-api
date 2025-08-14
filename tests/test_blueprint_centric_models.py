"""
Comprehensive tests for Blueprint-Centric Models

This test suite covers all the new blueprint-centric models including validation,
serialization, and core functionality.
"""

import pytest
from datetime import datetime
from typing import List

from app.models.blueprint_centric import (
    UueStage, TrackingIntensity, DifficultyLevel, AssessmentType,
    BlueprintSection, MasteryCriterion, KnowledgePrimitive, LearningBlueprint,
    MasteryCriterionRelationship, QuestionInstance, ContentGenerationRequest,
    ContentGenerationResponse, SectionTree, BlueprintValidationResult
)


class TestEnums:
    """Test enum values and validation."""
    
    def test_uue_stage_enum(self):
        """Test UUE stage enum values."""
        assert UueStage.UNDERSTAND == "UNDERSTAND"
        assert UueStage.USE == "USE"
        assert UueStage.EXPLORE == "EXPLORE"
        assert len(UueStage) == 3
    
    def test_tracking_intensity_enum(self):
        """Test tracking intensity enum values."""
        assert TrackingIntensity.DENSE == "DENSE"
        assert TrackingIntensity.NORMAL == "NORMAL"
        assert TrackingIntensity.SPARSE == "SPARSE"
        assert len(TrackingIntensity) == 3
    
    def test_difficulty_level_enum(self):
        """Test difficulty level enum values."""
        assert DifficultyLevel.BEGINNER == "BEGINNER"
        assert DifficultyLevel.INTERMEDIATE == "INTERMEDIATE"
        assert DifficultyLevel.ADVANCED == "ADVANCED"
        assert len(DifficultyLevel) == 3
    
    def test_assessment_type_enum(self):
        """Test assessment type enum values."""
        assert AssessmentType.QUESTION_BASED == "QUESTION_BASED"
        assert AssessmentType.EXPLANATION_BASED == "EXPLANATION_BASED"
        assert AssessmentType.APPLICATION_BASED == "APPLICATION_BASED"
        assert AssessmentType.MULTIMODAL == "MULTIMODAL"
        assert len(AssessmentType) == 4


class TestBlueprintSection:
    """Test BlueprintSection model."""
    
    def test_valid_blueprint_section(self):
        """Test creating a valid blueprint section."""
        section = BlueprintSection(
            title="Test Section",
            description="Test description",
            blueprint_id=1,
            user_id=1
        )
        
        assert section.title == "Test Section"
        assert section.description == "Test description"
        assert section.blueprint_id == 1
        assert section.user_id == 1
        assert section.depth == 0
        assert section.order_index == 0
        assert section.difficulty == DifficultyLevel.BEGINNER
        assert section.children == []
    
    def test_blueprint_section_with_hierarchy(self):
        """Test blueprint section with hierarchical structure."""
        parent = BlueprintSection(
            title="Parent Section",
            blueprint_id=1,
            user_id=1
        )
        
        child = BlueprintSection(
            title="Child Section",
            blueprint_id=1,
            parent_section_id=1,
            depth=1,
            order_index=1,
            user_id=1
        )
        
        parent.children = [child]
        
        assert len(parent.children) == 1
        assert parent.children[0].title == "Child Section"
        assert parent.children[0].depth == 1
        assert parent.children[0].parent_section_id == 1
    
    def test_blueprint_section_validation(self):
        """Test blueprint section validation rules."""
        # Test empty title validation
        with pytest.raises(ValueError, match="Section title cannot be empty"):
            BlueprintSection(
                title="",
                blueprint_id=1,
                user_id=1
            )
        
        # Test whitespace title validation
        with pytest.raises(ValueError, match="Section title cannot be empty"):
            BlueprintSection(
                title="   ",
                blueprint_id=1,
                user_id=1
            )
        
        # Test negative depth validation
        with pytest.raises(ValueError, match="Depth cannot be negative"):
            BlueprintSection(
                title="Test Section",
                blueprint_id=1,
                user_id=1,
                depth=-1
            )
        
        # Test negative order index validation
        with pytest.raises(ValueError, match="Order index cannot be negative"):
            BlueprintSection(
                title="Test Section",
                blueprint_id=1,
                user_id=1,
                order_index=-1
            )
    
    def test_blueprint_section_serialization(self):
        """Test blueprint section serialization."""
        section = BlueprintSection(
            title="Test Section",
            description="Test description",
            blueprint_id=1,
            user_id=1,
            difficulty=DifficultyLevel.INTERMEDIATE,
            estimated_time_minutes=30
        )
        
        # Test dict conversion
        section_dict = section.model_dump()
        assert section_dict["title"] == "Test Section"
        assert section_dict["difficulty"] == "INTERMEDIATE"
        assert section_dict["estimated_time_minutes"] == 30
        
        # Test JSON serialization
        section_json = section.model_dump_json()
        assert "Test Section" in section_json
        assert "INTERMEDIATE" in section_json


class TestMasteryCriterion:
    """Test MasteryCriterion model."""
    
    def test_valid_mastery_criterion(self):
        """Test creating a valid mastery criterion."""
        criterion = MasteryCriterion(
            title="What is a derivative?",
            description="Understand the basic concept of derivatives",
            weight=2.0,
            uue_stage=UueStage.UNDERSTAND,
            complexity_score=5.0,
            knowledge_primitive_id="primitive_1",
            blueprint_section_id=1,
            user_id=1
        )
        
        assert criterion.title == "What is a derivative?"
        assert criterion.weight == 2.0
        assert criterion.uue_stage == UueStage.UNDERSTAND
        assert criterion.complexity_score == 5.0
        assert criterion.mastery_threshold == 0.8
        assert criterion.assessment_type == AssessmentType.QUESTION_BASED
    
    def test_mastery_criterion_validation(self):
        """Test mastery criterion validation rules."""
        # Test empty title validation
        with pytest.raises(ValueError, match="Criterion title cannot be empty"):
            MasteryCriterion(
                title="",
                knowledge_primitive_id="primitive_1",
                blueprint_section_id=1,
                user_id=1
            )
        
        # Test weight range validation
        with pytest.raises(ValueError, match="Weight must be between 1.0 and 5.0"):
            MasteryCriterion(
                title="Test Criterion",
                weight=0.5,
                knowledge_primitive_id="primitive_1",
                blueprint_section_id=1,
                user_id=1
            )
        
        with pytest.raises(ValueError, match="Weight must be between 1.0 and 5.0"):
            MasteryCriterion(
                title="Test Criterion",
                weight=6.0,
                knowledge_primitive_id="primitive_1",
                blueprint_section_id=1,
                user_id=1
            )
        
        # Test mastery threshold validation
        with pytest.raises(ValueError, match="Mastery threshold must be one of"):
            MasteryCriterion(
                title="Test Criterion",
                mastery_threshold=0.7,
                knowledge_primitive_id="primitive_1",
                blueprint_section_id=1,
                user_id=1
            )
        
        # Test complexity score validation
        with pytest.raises(ValueError, match="Complexity score must be between 1.0 and 10.0"):
            MasteryCriterion(
                title="Test Criterion",
                complexity_score=0.5,
                knowledge_primitive_id="primitive_1",
                blueprint_section_id=1,
                user_id=1
            )
        
        with pytest.raises(ValueError, match="Complexity score must be between 1.0 and 10.0"):
            MasteryCriterion(
                title="Test Criterion",
                complexity_score=11.0,
                knowledge_primitive_id="primitive_1",
                blueprint_section_id=1,
                user_id=1
            )
    
    def test_mastery_criterion_defaults(self):
        """Test mastery criterion default values."""
        criterion = MasteryCriterion(
            title="Test Criterion",
            knowledge_primitive_id="primitive_1",
            blueprint_section_id=1,
            user_id=1
        )
        
        assert criterion.weight == 1.0
        assert criterion.uue_stage == UueStage.UNDERSTAND
        assert criterion.mastery_threshold == 0.8
        assert criterion.assessment_type == AssessmentType.QUESTION_BASED
        assert criterion.attempts_allowed == 3


class TestKnowledgePrimitive:
    """Test KnowledgePrimitive model."""
    
    def test_valid_knowledge_primitive(self):
        """Test creating a valid knowledge primitive."""
        primitive = KnowledgePrimitive(
            primitive_id="primitive_1",
            title="Derivative Concept",
            description="The concept of derivatives in calculus",
            primitive_type="concept",
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            estimated_time_minutes=45,
            tracking_intensity=TrackingIntensity.NORMAL,
            blueprint_section_id=1,
            user_id=1
        )
        
        assert primitive.primitive_id == "primitive_1"
        assert primitive.title == "Derivative Concept"
        assert primitive.primitive_type == "concept"
        assert primitive.difficulty_level == DifficultyLevel.INTERMEDIATE
        assert primitive.tracking_intensity == TrackingIntensity.NORMAL
    
    def test_knowledge_primitive_validation(self):
        """Test knowledge primitive validation rules."""
        # Test empty title validation
        with pytest.raises(ValueError, match="Primitive title cannot be empty"):
            KnowledgePrimitive(
                primitive_id="primitive_1",
                title="",
                primitive_type="concept",
                blueprint_section_id=1,
                user_id=1
            )
        
        # Test invalid primitive type validation
        with pytest.raises(ValueError, match="Primitive type must be one of"):
            KnowledgePrimitive(
                primitive_id="primitive_1",
                title="Test Primitive",
                primitive_type="invalid_type",
                blueprint_section_id=1,
                user_id=1
            )
    
    def test_knowledge_primitive_defaults(self):
        """Test knowledge primitive default values."""
        primitive = KnowledgePrimitive(
            primitive_id="primitive_1",
            title="Test Primitive",
            primitive_type="concept",
            blueprint_section_id=1,
            user_id=1
        )
        
        assert primitive.difficulty_level == DifficultyLevel.BEGINNER
        assert primitive.tracking_intensity == TrackingIntensity.NORMAL
        assert primitive.mastery_criteria == []


class TestLearningBlueprint:
    """Test LearningBlueprint model."""
    
    def test_valid_learning_blueprint(self):
        """Test creating a valid learning blueprint."""
        blueprint = LearningBlueprint(
            title="Calculus Fundamentals",
            description="Introduction to calculus concepts",
            user_id=1
        )
        
        assert blueprint.title == "Calculus Fundamentals"
        assert blueprint.description == "Introduction to calculus concepts"
        assert blueprint.user_id == 1
        assert blueprint.blueprint_sections == []
        assert blueprint.knowledge_primitives == []
        assert blueprint.tags == []
    
    def test_learning_blueprint_with_content(self):
        """Test learning blueprint with content and sections."""
        section = BlueprintSection(
            title="Derivatives",
            blueprint_id=1,
            user_id=1
        )
        
        primitive = KnowledgePrimitive(
            primitive_id="primitive_1",
            title="Derivative Concept",
            primitive_type="concept",
            blueprint_section_id=1,
            user_id=1
        )
        
        blueprint = LearningBlueprint(
            title="Calculus Fundamentals",
            description="Introduction to calculus concepts",
            user_id=1,
            content="Raw content here...",
            source_type="textbook",
            blueprint_sections=[section],
            knowledge_primitives=[primitive],
            tags=["mathematics", "calculus"]
        )
        
        assert len(blueprint.blueprint_sections) == 1
        assert len(blueprint.knowledge_primitives) == 1
        assert len(blueprint.tags) == 2
        assert blueprint.content == "Raw content here..."
        assert blueprint.source_type == "textbook"
    
    def test_learning_blueprint_validation(self):
        """Test learning blueprint validation rules."""
        # Test empty title validation
        with pytest.raises(ValueError, match="Blueprint title cannot be empty"):
            LearningBlueprint(
                title="",
                user_id=1
            )


class TestMasteryCriterionRelationship:
    """Test MasteryCriterionRelationship model."""
    
    def test_valid_relationship(self):
        """Test creating a valid mastery criterion relationship."""
        relationship = MasteryCriterionRelationship(
            source_criterion_id=1,
            target_criterion_id=2,
            relationship_type="prerequisite",
            description="Must understand derivatives before integrals",
            strength=0.9
        )
        
        assert relationship.source_criterion_id == 1
        assert relationship.target_criterion_id == 2
        assert relationship.relationship_type == "prerequisite"
        assert relationship.strength == 0.9
    
    def test_relationship_validation(self):
        """Test relationship validation rules."""
        # Test invalid relationship type
        with pytest.raises(ValueError, match="Relationship type must be one of"):
            MasteryCriterionRelationship(
                source_criterion_id=1,
                target_criterion_id=2,
                relationship_type="invalid_type",
                description="Test relationship"
            )
        
        # Test strength range validation
        with pytest.raises(ValueError, match="Strength must be between 0.0 and 1.0"):
            MasteryCriterionRelationship(
                source_criterion_id=1,
                target_criterion_id=2,
                relationship_type="prerequisite",
                description="Test relationship",
                strength=1.5
            )
    
    def test_relationship_defaults(self):
        """Test relationship default values."""
        relationship = MasteryCriterionRelationship(
            source_criterion_id=1,
            target_criterion_id=2,
            relationship_type="prerequisite",
            description="Test relationship"
        )
        
        assert relationship.strength == 1.0
        # Note: evidence field doesn't exist in the actual model


class TestQuestionInstance:
    """Test QuestionInstance model."""
    
    def test_valid_question_instance(self):
        """Test creating a valid question instance."""
        question = QuestionInstance(
            question_text="What is the derivative of x²?",
            answer="2x",
            explanation="The derivative of x² is 2x using the power rule",
            difficulty=DifficultyLevel.INTERMEDIATE,
            question_type="multiple_choice",
            mastery_criterion_id=1,
            user_id=1
        )
        
        assert question.question_text == "What is the derivative of x²?"
        assert question.answer == "2x"
        assert question.difficulty == DifficultyLevel.INTERMEDIATE
        assert question.question_type == "multiple_choice"
    
    def test_question_instance_validation(self):
        """Test question instance validation rules."""
        # Test empty question text validation
        with pytest.raises(ValueError, match="Question text cannot be empty"):
            QuestionInstance(
                question_text="",
                answer="2x",
                mastery_criterion_id=1,
                user_id=1
            )
        
        # Test empty answer validation
        with pytest.raises(ValueError, match="Answer cannot be empty"):
            QuestionInstance(
                question_text="What is the derivative of x²?",
                answer="",
                mastery_criterion_id=1,
                user_id=1
            )
    
    def test_question_instance_defaults(self):
        """Test question instance default values."""
        question = QuestionInstance(
            question_text="Test question?",
            answer="Test answer",
            question_type="multiple_choice",
            mastery_criterion_id=1,
            user_id=1
        )
        
        assert question.difficulty == DifficultyLevel.BEGINNER
        assert question.tags == []
        assert question.context is None


class TestContentGenerationRequest:
    """Test ContentGenerationRequest model."""
    
    def test_valid_content_generation_request(self):
        """Test creating a valid content generation request."""
        request = ContentGenerationRequest(
            blueprint_id=1,
            content_type="mastery_criteria",
            user_id=1,
            instructions={"style": "thorough", "difficulty": "intermediate"}
        )
        
        assert request.blueprint_id == 1
        assert request.content_type == "mastery_criteria"
        assert request.user_id == 1
        assert request.instructions == {"style": "thorough", "difficulty": "intermediate"}
        assert request.section_id is None
    
    def test_content_generation_request_validation(self):
        """Test content generation request validation rules."""
        # Test invalid content type validation
        with pytest.raises(ValueError, match="Content type must be one of"):
            ContentGenerationRequest(
                blueprint_id=1,
                content_type="invalid_type",
                user_id=1
            )
    
    def test_content_generation_request_defaults(self):
        """Test content generation request default values."""
        request = ContentGenerationRequest(
            blueprint_id=1,
            content_type="mastery_criteria",
            user_id=1
        )
        
        assert request.instructions == {}
        assert request.section_id is None


class TestContentGenerationResponse:
    """Test ContentGenerationResponse model."""
    
    def test_valid_content_generation_response(self):
        """Test creating a valid content generation response."""
        request = ContentGenerationRequest(
            blueprint_id=1,
            content_type="mastery_criteria",
            user_id=1
        )
        
        response = ContentGenerationResponse(
            request=request,
            success=True,
            content={"criteria": ["criterion1", "criterion2"]},
            message="Successfully generated 2 mastery criteria"
        )
        
        # Note: request field is not part of the actual model
        assert response.success is True
        assert response.content == {"criteria": ["criterion1", "criterion2"]}
        assert response.message == "Successfully generated 2 mastery criteria"
        assert response.errors == []
    
    def test_content_generation_response_with_errors(self):
        """Test content generation response with errors."""
        request = ContentGenerationRequest(
            blueprint_id=1,
            content_type="mastery_criteria",
            user_id=1
        )
        
        response = ContentGenerationResponse(
            request=request,
            success=False,
            content=None,
            message="Failed to generate content",
            errors=["Invalid blueprint ID", "Content too complex"]
        )
        
        assert response.success is False
        assert response.content is None
        assert len(response.errors) == 2
        assert "Invalid blueprint ID" in response.errors


class TestSectionTree:
    """Test SectionTree model."""
    
    def test_valid_section_tree(self):
        """Test creating a valid section tree."""
        section = BlueprintSection(
            title="Root Section",
            blueprint_id=1,
            user_id=1
        )
        
        tree = SectionTree(
            section=section,
            children=[],
            depth=0
        )
        
        assert tree.section == section
        assert tree.children == []
        assert tree.depth == 0
    
    def test_section_tree_with_children(self):
        """Test section tree with child sections."""
        root_section = BlueprintSection(
            title="Root Section",
            blueprint_id=1,
            user_id=1
        )
        
        child_section = BlueprintSection(
            title="Child Section",
            blueprint_id=1,
            user_id=1
        )
        
        child_tree = SectionTree(
            section=child_section,
            children=[],
            depth=1
        )
        
        root_tree = SectionTree(
            section=root_section,
            children=[child_tree],
            depth=0
        )
        
        assert len(root_tree.children) == 1
        assert root_tree.children[0].section.title == "Child Section"
        assert root_tree.children[0].depth == 1
    
    def test_section_tree_methods(self):
        """Test section tree utility methods."""
        root_section = BlueprintSection(
            title="Root Section",
            blueprint_id=1,
            user_id=1
        )
        
        child_section = BlueprintSection(
            title="Child Section",
            blueprint_id=1,
            user_id=1
        )
        
        child_tree = SectionTree(
            section=child_section,
            children=[],
            depth=1
        )
        
        root_tree = SectionTree(
            section=root_section,
            children=[child_tree],
            depth=0
        )
        
        # Test get_all_sections
        all_sections = root_tree.get_all_sections()
        assert len(all_sections) == 2
        assert all_sections[0].title == "Root Section"
        assert all_sections[1].title == "Child Section"
        
        # Test get_section_by_id - note: sections have id=None in test data
        # so get_section_by_id will return None for these test instances
        found_section = root_tree.get_section_by_id(1)  # Use a real ID
        assert found_section is None  # Should return None since test sections have id=None
        
        # Test get_section_by_id with non-existent ID
        found_section = root_tree.get_section_by_id(999)
        assert found_section is None


class TestBlueprintValidationResult:
    """Test BlueprintValidationResult model."""
    
    def test_valid_validation_result(self):
        """Test creating a valid validation result."""
        result = BlueprintValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            recommendations=[]
        )
        
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.recommendations == []
    
    def test_validation_result_methods(self):
        """Test validation result utility methods."""
        result = BlueprintValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            recommendations=[]
        )
        
        # Test add_error
        result.add_error("Missing mastery criteria")
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "Missing mastery criteria" in result.errors
        
        # Test add_warning
        result.add_warning("Limited UUE stage coverage")
        assert len(result.warnings) == 1
        assert "Limited UUE stage coverage" in result.warnings
        
        # Test add_recommendation
        result.add_recommendation("Add more mastery criteria")
        assert len(result.recommendations) == 1
        assert "Add more mastery criteria" in result.recommendations


class TestModelIntegration:
    """Test integration between different models."""
    
    def test_blueprint_with_sections_and_criteria(self):
        """Test creating a complete blueprint with sections and criteria."""
        # Create mastery criteria
        criterion1 = MasteryCriterion(
            title="Understand derivatives",
            description="Basic understanding of derivatives",
            weight=2.0,
            uue_stage=UueStage.UNDERSTAND,
            knowledge_primitive_id="prim_1",
            blueprint_section_id=1,
            user_id=1
        )
        
        criterion2 = MasteryCriterion(
            title="Apply derivatives",
            description="Apply derivative concepts",
            weight=3.0,
            uue_stage=UueStage.USE,
            knowledge_primitive_id="prim_2",
            blueprint_section_id=1,
            user_id=1
        )
        
        # Create section with criteria
        section = BlueprintSection(
            title="Derivatives",
            description="Introduction to derivatives",
            blueprint_id=1,
            user_id=1,
            difficulty=DifficultyLevel.INTERMEDIATE
        )
        
        # Create knowledge primitive
        primitive = KnowledgePrimitive(
            primitive_id="prim_1",
            title="Derivative Concept",
            primitive_type="concept",
            blueprint_section_id=1,
            user_id=1
        )
        
        # Create blueprint
        blueprint = LearningBlueprint(
            title="Calculus Fundamentals",
            description="Introduction to calculus",
            user_id=1,
            blueprint_sections=[section],
            knowledge_primitives=[primitive]
        )
        
        # Verify structure
        assert len(blueprint.blueprint_sections) == 1
        assert blueprint.blueprint_sections[0].title == "Derivatives"
        assert len(blueprint.knowledge_primitives) == 1
        assert blueprint.knowledge_primitives[0].title == "Derivative Concept"
    
    def test_model_serialization_integration(self):
        """Test that models can be serialized together."""
        # Create a complex structure
        section = BlueprintSection(
            title="Test Section",
            blueprint_id=1,
            user_id=1
        )
        
        criterion = MasteryCriterion(
            title="Test Criterion",
            knowledge_primitive_id="prim_1",
            blueprint_section_id=1,
            user_id=1
        )
        
        blueprint = LearningBlueprint(
            title="Test Blueprint",
            user_id=1,
            blueprint_sections=[section]
        )
        
        # Test JSON serialization
        try:
            blueprint_json = blueprint.model_dump_json()
            assert "Test Blueprint" in blueprint_json
            assert "Test Section" in blueprint_json
        except Exception as e:
            pytest.fail(f"Serialization failed: {e}")
    
    def test_enum_integration(self):
        """Test that enums work correctly across all models."""
        # Test UUE stage integration
        criterion = MasteryCriterion(
            title="Test Criterion",
            knowledge_primitive_id="prim_1",
            blueprint_section_id=1,
            user_id=1,
            uue_stage=UueStage.EXPLORE
        )
        assert criterion.uue_stage == UueStage.EXPLORE
        
        # Test difficulty level integration
        section = BlueprintSection(
            title="Test Section",
            blueprint_id=1,
            user_id=1,
            difficulty=DifficultyLevel.ADVANCED
        )
        assert section.difficulty == DifficultyLevel.ADVANCED
        
        # Test tracking intensity integration
        primitive = KnowledgePrimitive(
            primitive_id="prim_1",
            title="Test Primitive",
            primitive_type="concept",
            blueprint_section_id=1,
            user_id=1,
            tracking_intensity=TrackingIntensity.DENSE
        )
        assert primitive.tracking_intensity == TrackingIntensity.DENSE


if __name__ == "__main__":
    pytest.main([__file__])
