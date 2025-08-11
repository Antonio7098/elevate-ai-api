"""
Pydantic models for the LearningBlueprint structure.

This module defines the complete schema for LearningBlueprints, including
all knowledge primitives and their relationships.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, Dict, Any


class MasteryCriterion(BaseModel):
    """Represents a mastery criterion for a knowledge primitive (Core API Prisma compatible)."""
    criterionId: str = Field(..., description="Unique criterion ID (matches Prisma criterionId)")
    title: str = Field(..., description="Criterion title")
    description: Optional[str] = Field(None, description="Criterion description")
    ueeLevel: Literal["UNDERSTAND", "USE", "EXPLORE"] = Field(..., description="UEE level (matches Prisma enum)")
    weight: float = Field(..., description="Criterion importance weight (matches Prisma Float)")
    isRequired: bool = Field(default=True, description="Whether criterion is required (matches Prisma)")
    
    @field_validator('title')
    @classmethod
    def validate_title_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Criterion title cannot be empty')
        return v.strip()
    
    @field_validator('ueeLevel')
    @classmethod
    def validate_uee_level(cls, v):
        valid_levels = ["UNDERSTAND", "USE", "EXPLORE"]
        if v not in valid_levels:
            raise ValueError(f'UEE level must be one of: {", ".join(valid_levels)}')
        return v


class KnowledgePrimitive(BaseModel):
    """Represents a knowledge primitive (Core API Prisma compatible)."""
    primitiveId: str = Field(..., description="Unique primitive ID (matches Prisma primitiveId)")
    title: str = Field(..., description="Primitive title")
    description: Optional[str] = Field(None, description="Primitive description")
    primitiveType: str = Field(..., description="Primitive type: fact, concept, process (matches Prisma)")
    difficultyLevel: str = Field(..., description="Difficulty level: beginner, intermediate, advanced")
    estimatedTimeMinutes: Optional[int] = Field(None, description="Estimated time in minutes")
    trackingIntensity: Literal["DENSE", "NORMAL", "SPARSE"] = Field(default="NORMAL", description="Tracking intensity")
    masteryCriteria: List[MasteryCriterion] = Field(default_factory=list, description="Associated mastery criteria")
    
    @field_validator('title')
    @classmethod
    def validate_title_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Primitive title cannot be empty')
        return v.strip()
    
    @field_validator('primitiveType')
    @classmethod
    def validate_primitive_type(cls, v):
        valid_types = ["fact", "concept", "process"]
        if v not in valid_types:
            raise ValueError(f'Primitive type must be one of: {", ".join(valid_types)}')
        return v
    
    @field_validator('difficultyLevel')
    @classmethod
    def validate_difficulty_level(cls, v):
        valid_levels = ["beginner", "intermediate", "advanced"]
        if v not in valid_levels:
            raise ValueError(f'Difficulty level must be one of: {", ".join(valid_levels)}')
        return v
    
    @field_validator('trackingIntensity')
    @classmethod
    def validate_tracking_intensity(cls, v):
        valid_intensities = ["DENSE", "NORMAL", "SPARSE"]
        if v not in valid_intensities:
            raise ValueError(f'Tracking intensity must be one of: {", ".join(valid_intensities)}')
        return v


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
    mastery_criteria: List[MasteryCriterion] = Field(default_factory=list, description="Mastery criteria for this proposition")


class Entity(BaseModel):
    """Represents a key entity, term, or definition from the source."""
    id: str = Field(..., description="Unique identifier for the entity")
    entity: str = Field(..., description="The entity/term itself")
    definition: str = Field(..., description="Definition or explanation of the entity")
    category: Literal["Person", "Organization", "Concept", "Place", "Object"] = Field(..., description="Category of the entity")
    sections: List[str] = Field(default_factory=list, description="IDs of sections where this entity appears")
    mastery_criteria: List[MasteryCriterion] = Field(default_factory=list, description="Mastery criteria for this entity")


class Process(BaseModel):
    """Represents a described process or set of steps from the source."""
    id: str = Field(..., description="Unique identifier for the process")
    process_name: str = Field(..., description="Name of the process")
    steps: List[str] = Field(default_factory=list, description="List of steps in the process")
    sections: List[str] = Field(default_factory=list, description="IDs of sections where this process is described")
    mastery_criteria: List[MasteryCriterion] = Field(default_factory=list, description="Mastery criteria for this process")


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
    
    def validate_mastery_criteria_coverage(self) -> Dict[str, Any]:
        """Validate mastery criteria coverage across all primitives."""
        all_primitives = (
            self.key_propositions_and_facts + 
            self.key_entities_and_definitions + 
            self.described_processes_and_steps
        )
        
        total_primitives = len(all_primitives)
        primitives_with_criteria = sum(1 for p in all_primitives if p.mastery_criteria)
        
        # Count UEE level distribution
        uee_counts = {"Understand": 0, "Use": 0, "Explore": 0}
        weight_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for primitive in all_primitives:
            for criterion in primitive.mastery_criteria:
                uee_counts[criterion.uee_level] += 1
                weight_distribution[criterion.weight] += 1
        
        return {
            "total_primitives": total_primitives,
            "primitives_with_criteria": primitives_with_criteria,
            "coverage_percentage": (primitives_with_criteria / total_primitives * 100) if total_primitives > 0 else 0,
            "uee_distribution": uee_counts,
            "weight_distribution": weight_distribution,
            "has_balanced_uee": all(count > 0 for count in uee_counts.values()),
            "recommendations": self._generate_coverage_recommendations(uee_counts, weight_distribution, total_primitives)
        }
    
    def _generate_coverage_recommendations(self, uee_counts: Dict[str, int], weight_distribution: Dict[int, int], total_primitives: int) -> List[str]:
        """Generate recommendations for improving mastery criteria coverage."""
        recommendations = []
        
        # Check for missing UEE levels
        missing_uee = [level for level, count in uee_counts.items() if count == 0]
        if missing_uee:
            recommendations.append(f"Consider adding criteria for UEE levels: {', '.join(missing_uee)}")
        
        # Check for weight distribution balance
        total_criteria = sum(weight_distribution.values())
        if total_criteria > 0:
            high_weight_percentage = (weight_distribution[4] + weight_distribution[5]) / total_criteria
            if high_weight_percentage > 0.7:
                recommendations.append("High concentration of high-weight criteria. Consider redistributing weights for better balance.")
            elif high_weight_percentage < 0.2:
                recommendations.append("Consider adding more high-importance criteria to ensure key learning objectives are prioritized.")
        
        # Check overall coverage
        if total_primitives > 0:
            primitives_with_criteria = sum(1 for p in (
                self.key_propositions_and_facts + 
                self.key_entities_and_definitions + 
                self.described_processes_and_steps
            ) if p.mastery_criteria)
            coverage = primitives_with_criteria / total_primitives
            if coverage < 0.8:
                recommendations.append(f"Only {coverage:.1%} of primitives have mastery criteria. Consider adding criteria to more primitives.")
        
        return recommendations


class LearningBlueprint(BaseModel):
    """Root model representing a complete LearningBlueprint."""
    source_id: str = Field(..., description="Unique identifier for the source")
    source_title: str = Field(..., description="Title of the learning source")
    source_type: str = Field(..., description="Type of source (e.g., chapter, article, video)")
    source_summary: dict = Field(..., description="Summary information about the source")
    # Optional raw content and serialized form retained for compatibility with tests and services
    content: Optional[str] = Field(default=None, description="Raw source content (optional)")
    blueprint_json: Optional[Dict[str, Any]] = Field(default=None, description="Serialized blueprint payload (optional)")
    sections: List[Section] = Field(default_factory=list, description="Hierarchical sections of the source")
    tags: List[str] = Field(default_factory=list, description="Optional tags associated with the blueprint")
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