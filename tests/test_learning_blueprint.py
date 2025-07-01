"""
Tests for LearningBlueprint models and validation.
"""

import pytest
from app.models.learning_blueprint import (
    LearningBlueprint,
    Section,
    Proposition,
    Entity,
    Process,
    Relationship,
    Question,
    KnowledgePrimitives
)
from pydantic import ValidationError


class TestLearningBlueprintModels:
    """Test cases for LearningBlueprint Pydantic models."""
    
    def test_section_model(self):
        """Test Section model creation and validation."""
        section = Section(
            section_id="test_section",
            section_name="Test Section",
            description="A test section",
            parent_section_id=None
        )
        
        assert section.section_id == "test_section"
        assert section.section_name == "Test Section"
        assert section.description == "A test section"
        assert section.parent_section_id is None
    
    def test_proposition_model(self):
        """Test Proposition model creation and validation."""
        proposition = Proposition(
            id="prop_1",
            statement="Test proposition",
            supporting_evidence=["Evidence 1", "Evidence 2"],
            sections=["sec_1", "sec_2"]
        )
        
        assert proposition.id == "prop_1"
        assert proposition.statement == "Test proposition"
        assert len(proposition.supporting_evidence) == 2
        assert len(proposition.sections) == 2
    
    def test_entity_model(self):
        """Test Entity model creation and validation."""
        entity = Entity(
            id="entity_1",
            entity="Test Entity",
            definition="A test entity definition",
            category="Concept",
            sections=["sec_1"]
        )
        
        assert entity.id == "entity_1"
        assert entity.entity == "Test Entity"
        assert entity.definition == "A test entity definition"
        assert entity.category == "Concept"
    
    def test_process_model(self):
        """Test Process model creation and validation."""
        process = Process(
            id="process_1",
            process_name="Test Process",
            steps=["Step 1", "Step 2", "Step 3"],
            sections=["sec_1"]
        )
        
        assert process.id == "process_1"
        assert process.process_name == "Test Process"
        assert len(process.steps) == 3
    
    def test_relationship_model(self):
        """Test Relationship model creation and validation."""
        relationship = Relationship(
            id="rel_1",
            relationship_type="causal",
            source_primitive_id="prop_1",
            target_primitive_id="prop_2",
            description="A causes B",
            sections=["sec_1"]
        )
        
        assert relationship.id == "rel_1"
        assert relationship.relationship_type == "causal"
        assert relationship.source_primitive_id == "prop_1"
        assert relationship.target_primitive_id == "prop_2"
    
    def test_question_model(self):
        """Test Question model creation and validation."""
        question = Question(
            id="q_1",
            question="What is the main idea?",
            sections=["sec_1"]
        )
        
        assert question.id == "q_1"
        assert question.question == "What is the main idea?"
    
    def test_knowledge_primitives_model(self):
        """Test KnowledgePrimitives model creation and validation."""
        primitives = KnowledgePrimitives(
            key_propositions_and_facts=[
                Proposition(id="p1", statement="Test prop", supporting_evidence=[], sections=[])
            ],
            key_entities_and_definitions=[
                Entity(id="e1", entity="Test", definition="Test def", category="Concept", sections=[])
            ],
            described_processes_and_steps=[],
            identified_relationships=[],
            implicit_and_open_questions=[]
        )
        
        assert len(primitives.key_propositions_and_facts) == 1
        assert len(primitives.key_entities_and_definitions) == 1
    
    def test_learning_blueprint_model(self):
        """Test LearningBlueprint model creation and validation."""
        blueprint = LearningBlueprint(
            source_id="test_source",
            source_title="Test Source",
            source_type="chapter",
            source_summary={
                "core_thesis_or_main_argument": "Test thesis",
                "inferred_purpose": "Test purpose"
            },
            sections=[],
            knowledge_primitives=KnowledgePrimitives(
                key_propositions_and_facts=[],
                key_entities_and_definitions=[],
                described_processes_and_steps=[],
                identified_relationships=[],
                implicit_and_open_questions=[]
            )
        )
        
        assert blueprint.source_id == "test_source"
        assert blueprint.source_title == "Test Source"
        assert blueprint.source_type == "chapter"
        assert "core_thesis_or_main_argument" in blueprint.source_summary


class TestLearningBlueprintValidation:
    """Test cases for LearningBlueprint validation scenarios."""
    
    def test_invalid_entity_category(self):
        """Test that invalid entity categories raise validation errors."""
        with pytest.raises(ValueError):
            Entity(
                id="e1",
                entity="Test",
                definition="Test def",
                category="InvalidCategory",  # Invalid category
                sections=[]
            )
    
    def test_empty_required_fields(self):
        """Test that empty required fields raise validation errors."""
        with pytest.raises(ValidationError):
            Section(
                section_id="",  # Empty required field
                section_name="Test",
                description="Test"
            ) 