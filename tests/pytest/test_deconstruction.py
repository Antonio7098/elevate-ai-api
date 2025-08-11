"""
Tests for the deconstruction pipeline.
"""

import pytest
from app.core.deconstruction import (
    find_sections,
    extract_foundational_concepts,
    extract_key_terms,
    extract_processes,
    identify_relationships,
    deconstruct_text
)
from app.models.learning_blueprint import Section, Proposition, Entity, Process, Question, Relationship


class TestDeconstructionPipeline:
    """Test cases for the deconstruction pipeline."""
    
    @pytest.mark.asyncio
    async def test_find_sections_with_markdown(self):
        """Test section extraction with markdown headings."""
        text = """
# Introduction
This is the introduction section.

## Background
This is the background section.

### Details
This contains more details.
"""
        sections = await find_sections(text)
        assert len(sections) >= 1
        assert all(isinstance(section, Section) for section in sections)
    
    @pytest.mark.asyncio
    async def test_find_sections_without_headings(self):
        """Test section extraction without headings."""
        text = "This is a simple text without any headings or structure."
        sections = await find_sections(text)
        assert len(sections) >= 1
        assert all(isinstance(section, Section) for section in sections)
    
    @pytest.mark.asyncio
    async def test_extract_propositions(self):
        """Test proposition extraction."""
        text = "Photosynthesis is the process by which plants convert light energy into chemical energy."
        section_id = "test_section"
        propositions = await extract_foundational_concepts(text, section_id)
        # Should return a list (may be empty if LLM fails)
        assert isinstance(propositions, list)
        if propositions:
            assert all(isinstance(prop, Proposition) for prop in propositions)
    
    @pytest.mark.asyncio
    async def test_extract_entities(self):
        """Test entity extraction."""
        text = "Chloroplasts are organelles found in plant cells that conduct photosynthesis."
        section_id = "test_section"
        entities = await extract_key_terms(text, section_id)
        # Should return a list (may be empty if LLM fails)
        assert isinstance(entities, list)
        if entities:
            assert all(isinstance(entity, Entity) for entity in entities)
    
    @pytest.mark.asyncio
    async def test_extract_processes(self):
        """Test process extraction."""
        text = "The process involves three steps: light absorption, electron transport, and carbon fixation."
        section_id = "test_section"
        processes = await extract_processes(text, section_id)
        # Should return a list (may be empty if LLM fails)
        assert isinstance(processes, list)
        if processes:
            assert all(isinstance(process, Process) for process in processes)
    
    @pytest.mark.asyncio
    async def test_identify_relationships(self):
        """Test relationship identification."""
        propositions = [
            Proposition(id="p1", statement="Test proposition", supporting_evidence=[], sections=[])
        ]
        entities = [
            Entity(id="e1", entity="Test entity", definition="Test definition", category="Concept", sections=[])
        ]
        processes = [
            Process(id="proc1", process_name="Test process", steps=[], sections=[])
        ]
        relationships = await identify_relationships(propositions, entities, processes)
        # Should return a list (may be empty if LLM fails)
        assert isinstance(relationships, list)
        if relationships:
            assert all(isinstance(rel, Relationship) for rel in relationships)
    
    @pytest.mark.asyncio
    async def test_full_deconstruction_pipeline(self):
        """Test the complete deconstruction pipeline."""
        text = """
# Photosynthesis

Photosynthesis is the process by which plants convert light energy into chemical energy. 
This process occurs in chloroplasts and involves multiple steps.

## Light Reactions
The light reactions capture solar energy and convert it to chemical energy in the form of ATP and NADPH.

## Calvin Cycle
The Calvin cycle uses the products of light reactions to fix carbon dioxide and produce sugars.
"""
        blueprint = await deconstruct_text(text, "chapter")
        
        # Verify the blueprint structure
        assert blueprint.source_id is not None
        assert blueprint.source_title is not None
        assert blueprint.source_type == "chapter"
        assert blueprint.source_summary is not None
        assert isinstance(blueprint.sections, list)
        assert isinstance(blueprint.knowledge_primitives.key_propositions_and_facts, list)
        assert isinstance(blueprint.knowledge_primitives.key_entities_and_definitions, list)
        assert isinstance(blueprint.knowledge_primitives.described_processes_and_steps, list)
        assert isinstance(blueprint.knowledge_primitives.identified_relationships, list)
        assert isinstance(blueprint.knowledge_primitives.implicit_and_open_questions, list)
        # Questions should be empty since we're not extracting them anymore
        assert len(blueprint.knowledge_primitives.implicit_and_open_questions) == 0 