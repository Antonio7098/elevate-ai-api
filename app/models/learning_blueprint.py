"""
Pydantic models for the LearningBlueprint structure.

This module defines the complete schema for LearningBlueprints, including
all knowledge primitives and their relationships.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal


class Section(BaseModel):
    """Represents a hierarchical section of the source."""
    section_id: str = Field(..., description="Unique identifier for the section")
    section_name: str = Field(..., description="Name/title of the section")
    description: str = Field(..., description="Description of what the section covers")
    parent_section_id: Optional[str] = Field(None, description="ID of the parent section, if any")
    
    @field_validator('section_id', 'section_name')
    @classmethod
    def validate_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Field cannot be empty')
        return v


class Proposition(BaseModel):
    """Represents a key proposition or fact from the source."""
    id: str = Field(..., description="Unique identifier for the proposition")
    statement: str = Field(..., description="The proposition statement")
    supporting_evidence: List[str] = Field(default_factory=list, description="Supporting evidence for the proposition")
    sections: List[str] = Field(default_factory=list, description="IDs of sections where this proposition appears")


class Entity(BaseModel):
    """Represents a key entity, term, or definition from the source."""
    id: str = Field(..., description="Unique identifier for the entity")
    entity: str = Field(..., description="The entity/term itself")
    definition: str = Field(..., description="Definition or explanation of the entity")
    category: Literal["Person", "Organization", "Concept", "Place", "Object"] = Field(..., description="Category of the entity")
    sections: List[str] = Field(default_factory=list, description="IDs of sections where this entity appears")


class Process(BaseModel):
    """Represents a described process or set of steps from the source."""
    id: str = Field(..., description="Unique identifier for the process")
    process_name: str = Field(..., description="Name of the process")
    steps: List[str] = Field(default_factory=list, description="List of steps in the process")
    sections: List[str] = Field(default_factory=list, description="IDs of sections where this process is described")


class Relationship(BaseModel):
    """Represents a relationship between knowledge primitives."""
    id: str = Field(..., description="Unique identifier for the relationship")
    relationship_type: str = Field(..., description="Type of relationship (e.g., causal, part-of, component-of)")
    source_primitive_id: str = Field(..., description="ID of the source primitive")
    target_primitive_id: str = Field(..., description="ID of the target primitive")
    description: str = Field(..., description="Description of the relationship")
    sections: List[str] = Field(default_factory=list, description="IDs of sections where this relationship is described")


class Question(BaseModel):
    """Represents an implicit or open question from the source."""
    id: str = Field(..., description="Unique identifier for the question")
    question: str = Field(..., description="The question text")
    sections: List[str] = Field(default_factory=list, description="IDs of sections where this question is relevant")


class KnowledgePrimitives(BaseModel):
    """Container for all knowledge primitives extracted from the source."""
    key_propositions_and_facts: List[Proposition] = Field(default_factory=list, description="Key propositions and facts")
    key_entities_and_definitions: List[Entity] = Field(default_factory=list, description="Key entities and their definitions")
    described_processes_and_steps: List[Process] = Field(default_factory=list, description="Described processes and their steps")
    identified_relationships: List[Relationship] = Field(default_factory=list, description="Identified relationships between primitives")
    implicit_and_open_questions: List[Question] = Field(default_factory=list, description="Implicit and open questions")


class LearningBlueprint(BaseModel):
    """Root model representing a complete LearningBlueprint."""
    source_id: str = Field(..., description="Unique identifier for the source")
    source_title: str = Field(..., description="Title of the learning source")
    source_type: str = Field(..., description="Type of source (e.g., chapter, article, video)")
    source_summary: dict = Field(..., description="Summary information about the source")
    sections: List[Section] = Field(default_factory=list, description="Hierarchical sections of the source")
    knowledge_primitives: KnowledgePrimitives = Field(..., description="All knowledge primitives extracted from the source")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "source_id": "chapter1_photosynthesis",
                "source_title": "Introduction to Photosynthesis",
                "source_type": "chapter",
                "source_summary": {
                    "core_thesis_or_main_argument": "Photosynthesis is the fundamental process by which plants convert light energy into chemical energy.",
                    "inferred_purpose": "To explain the biochemical mechanisms of photosynthesis and its ecological significance."
                },
                "sections": [
                    {
                        "section_id": "sec_intro",
                        "section_name": "Introduction",
                        "description": "Overview of photosynthesis",
                        "parent_section_id": None
                    }
                ],
                "knowledge_primitives": {
                    "key_propositions_and_facts": [
                        {
                            "id": "prop_1",
                            "statement": "Photosynthesis converts light energy into chemical energy.",
                            "supporting_evidence": ["Experimental observations", "Chemical analysis"],
                            "sections": ["sec_intro"]
                        }
                    ],
                    "key_entities_and_definitions": [
                        {
                            "id": "entity_1",
                            "entity": "Chloroplast",
                            "definition": "Organelles found in plant cells that conduct photosynthesis.",
                            "category": "Object",
                            "sections": ["sec_intro"]
                        }
                    ],
                    "described_processes_and_steps": [],
                    "identified_relationships": [],
                    "implicit_and_open_questions": []
                }
            }
        } 