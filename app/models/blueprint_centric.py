"""
Blueprint-Centric Models for AI API

This module defines models that align with the Core API's blueprint-centric architecture,
ensuring seamless integration between AI API content generation and Core API mastery tracking.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, Dict, Any, Union
from datetime import datetime
from enum import Enum


# Enums that match Core API
class UueStage(str, Enum):
    UNDERSTAND = "UNDERSTAND"
    USE = "USE"
    EXPLORE = "EXPLORE"


class TrackingIntensity(str, Enum):
    DENSE = "DENSE"
    NORMAL = "NORMAL"
    SPARSE = "SPARSE"


class DifficultyLevel(str, Enum):
    BEGINNER = "BEGINNER"
    INTERMEDIATE = "INTERMEDIATE"
    ADVANCED = "ADVANCED"


class AssessmentType(str, Enum):
    QUESTION_BASED = "QUESTION_BASED"
    EXPLANATION_BASED = "EXPLANATION_BASED"
    APPLICATION_BASED = "APPLICATION_BASED"
    MULTIMODAL = "MULTIMODAL"


# Core Models that align with Core API
class BlueprintSection(BaseModel):
    """Blueprint section that aligns with Core API BlueprintSection model."""
    id: Optional[int] = Field(None, description="Database ID (auto-generated)")
    title: str = Field(..., description="Section name")
    description: Optional[str] = Field(None, description="Section description")
    blueprint_id: int = Field(..., description="Reference to parent blueprint")
    parent_section_id: Optional[int] = Field(None, description="For hierarchical nesting")
    depth: int = Field(default=0, description="Nesting depth")
    order_index: int = Field(default=0, description="Display order")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.BEGINNER, description="Difficulty level")
    estimated_time_minutes: Optional[int] = Field(None, description="Estimated study time")
    user_id: int = Field(..., description="Owner")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    # Hierarchical relationships
    children: List['BlueprintSection'] = Field(default_factory=list, description="Child sections")
    
    @field_validator('title')
    @classmethod
    def validate_title_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Section title cannot be empty')
        return v.strip()
    
    @field_validator('depth')
    @classmethod
    def validate_depth_non_negative(cls, v):
        if v < 0:
            raise ValueError('Depth cannot be negative')
        return v
    
    @field_validator('order_index')
    @classmethod
    def validate_order_index_non_negative(cls, v):
        if v < 0:
            raise ValueError('Order index cannot be negative')
        return v


class MasteryCriterion(BaseModel):
    """Mastery criterion that aligns with Core API MasteryCriterion model."""
    id: Optional[int] = Field(None, description="Database ID (auto-generated)")
    title: str = Field(..., description="The question family or learning objective")
    description: Optional[str] = Field(None, description="Detailed description")
    weight: float = Field(default=1.0, description="Importance weight (1.0-5.0)")
    uue_stage: UueStage = Field(default=UueStage.UNDERSTAND, description="UUE stage for SR and learning pathways")
    complexity_score: Optional[float] = Field(None, description="AI-calculated complexity (1-10)")
    knowledge_primitive_id: str = Field(..., description="Links to knowledge primitive")
    blueprint_section_id: int = Field(..., description="Links to blueprint section")
    user_id: int = Field(..., description="Owner")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    # Enhanced fields
    assessment_type: AssessmentType = Field(default=AssessmentType.QUESTION_BASED, description="Type of assessment")
    mastery_threshold: float = Field(default=0.8, description="Score needed to master (0.6, 0.8, or 0.95)")
    time_limit: Optional[int] = Field(None, description="Time limit in seconds")
    attempts_allowed: int = Field(default=3, description="Number of attempts allowed")
    
    @field_validator('title')
    @classmethod
    def validate_title_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Criterion title cannot be empty')
        return v.strip()
    
    @field_validator('weight')
    @classmethod
    def validate_weight_range(cls, v):
        if v < 1.0 or v > 5.0:
            raise ValueError('Weight must be between 1.0 and 5.0')
        return v
    
    @field_validator('mastery_threshold')
    @classmethod
    def validate_mastery_threshold(cls, v):
        valid_thresholds = [0.6, 0.8, 0.95]
        if v not in valid_thresholds:
            raise ValueError(f'Mastery threshold must be one of: {valid_thresholds}')
        return v
    
    @field_validator('complexity_score')
    @classmethod
    def validate_complexity_score(cls, v):
        if v is not None and (v < 1.0 or v > 10.0):
            raise ValueError('Complexity score must be between 1.0 and 10.0')
        return v


class KnowledgePrimitive(BaseModel):
    """Knowledge primitive that aligns with Core API KnowledgePrimitive model."""
    primitive_id: str = Field(..., description="Unique primitive ID")
    title: str = Field(..., description="Primitive title")
    description: Optional[str] = Field(None, description="Primitive description")
    primitive_type: str = Field(..., description="Primitive type: fact, concept, process")
    difficulty_level: DifficultyLevel = Field(default=DifficultyLevel.BEGINNER, description="Difficulty level")
    estimated_time_minutes: Optional[int] = Field(None, description="Estimated time in minutes")
    tracking_intensity: TrackingIntensity = Field(default=TrackingIntensity.NORMAL, description="Tracking intensity")
    blueprint_section_id: int = Field(..., description="Links to blueprint section")
    user_id: int = Field(..., description="Owner")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    # Relations
    mastery_criteria: List[MasteryCriterion] = Field(default_factory=list, description="Associated mastery criteria")
    
    @field_validator('title')
    @classmethod
    def validate_title_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Primitive title cannot be empty')
        return v.strip()
    
    @field_validator('primitive_type')
    @classmethod
    def validate_primitive_type(cls, v):
        valid_types = ["fact", "concept", "process"]
        if v not in valid_types:
            raise ValueError(f'Primitive type must be one of: {valid_types}')
        return v


class LearningBlueprint(BaseModel):
    """Learning blueprint that aligns with Core API LearningBlueprint model."""
    id: Optional[int] = Field(None, description="Database ID (auto-generated)")
    title: str = Field(..., description="Blueprint title")
    description: Optional[str] = Field(None, description="Blueprint description")
    user_id: int = Field(..., description="Owner")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    # Content
    content: Optional[str] = Field(None, description="Raw source content")
    source_type: Optional[str] = Field(None, description="Type of source (e.g., chapter, article, video)")
    source_summary: Optional[Dict[str, Any]] = Field(None, description="Summary information about the source")
    
    # Hierarchical structure
    blueprint_sections: List[BlueprintSection] = Field(default_factory=list, description="Blueprint sections")
    knowledge_primitives: List[KnowledgePrimitive] = Field(default_factory=list, description="Knowledge primitives")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Optional tags")
    
    @field_validator('title')
    @classmethod
    def validate_title_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Blueprint title cannot be empty')
        return v.strip()


# Relationship Models for Knowledge Graph
class MasteryCriterionRelationship(BaseModel):
    """Relationship between mastery criteria for learning pathways."""
    id: Optional[int] = Field(None, description="Database ID (auto-generated)")
    source_criterion_id: int = Field(..., description="Source criterion ID")
    target_criterion_id: int = Field(..., description="Target criterion ID")
    relationship_type: str = Field(..., description="Type of relationship (prerequisite, related, etc.)")
    description: Optional[str] = Field(None, description="Description of the relationship")
    strength: float = Field(default=1.0, description="Relationship strength (0.0-1.0)")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    
    @field_validator('relationship_type')
    @classmethod
    def validate_relationship_type(cls, v):
        valid_types = ["prerequisite", "related", "builds_on", "extends", "contradicts"]
        if v not in valid_types:
            raise ValueError(f'Relationship type must be one of: {valid_types}')
        return v
    
    @field_validator('strength')
    @classmethod
    def validate_strength_range(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Strength must be between 0.0 and 1.0')
        return v


class QuestionInstance(BaseModel):
    """Question instance that aligns with Core API QuestionInstance model."""
    id: Optional[int] = Field(None, description="Database ID (auto-generated)")
    question_text: str = Field(..., description="The question text")
    answer: str = Field(..., description="The correct answer")
    explanation: Optional[str] = Field(None, description="Explanation of the answer")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.BEGINNER, description="Question difficulty")
    question_type: str = Field(..., description="Type of question (multiple_choice, fill_blank, etc.)")
    mastery_criterion_id: int = Field(..., description="Links to mastery criterion")
    user_id: int = Field(..., description="Owner")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    # Question metadata
    tags: List[str] = Field(default_factory=list, description="Question tags")
    context: Optional[str] = Field(None, description="Additional context for the question")
    
    @field_validator('question_text')
    @classmethod
    def validate_question_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Question text cannot be empty')
        return v.strip()
    
    @field_validator('answer')
    @classmethod
    def validate_answer_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Answer cannot be empty')
        return v.strip()


# Content Generation Models
class ContentGenerationRequest(BaseModel):
    """Request for AI-powered content generation."""
    blueprint_id: int = Field(..., description="Target blueprint ID")
    section_id: Optional[int] = Field(None, description="Target section ID (if generating for specific section)")
    content_type: str = Field(..., description="Type of content to generate")
    instructions: Dict[str, Any] = Field(default_factory=dict, description="Generation instructions")
    user_id: int = Field(..., description="User requesting generation")
    
    @field_validator('content_type')
    @classmethod
    def validate_content_type(cls, v):
        valid_types = ["mastery_criteria", "questions", "sections", "primitives", "relationships"]
        if v not in valid_types:
            raise ValueError(f'Content type must be one of: {valid_types}')
        return v


class ContentGenerationResponse(BaseModel):
    """Response from AI-powered content generation."""
    success: bool = Field(..., description="Whether generation was successful")
    content: Optional[Dict[str, Any]] = Field(None, description="Generated content")
    message: str = Field(..., description="Response message")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Generation metadata")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


# Section Hierarchy Models
class SectionTree(BaseModel):
    """Tree representation of blueprint sections."""
    section: BlueprintSection = Field(..., description="The section")
    children: List['SectionTree'] = Field(default_factory=list, description="Child sections")
    depth: int = Field(..., description="Depth in the tree")
    
    def get_all_sections(self) -> List[BlueprintSection]:
        """Get all sections in the tree (including self and children)."""
        sections = [self.section]
        for child in self.children:
            sections.extend(child.get_all_sections())
        return sections
    
    def get_section_by_id(self, section_id: int) -> Optional[BlueprintSection]:
        """Find a section by ID in the tree."""
        if self.section.id == section_id:
            return self.section
        for child in self.children:
            result = child.get_section_by_id(section_id)
            if result:
                return result
        return None


# Validation and Utility Models
class BlueprintValidationResult(BaseModel):
    """Result of blueprint validation."""
    is_valid: bool = Field(..., description="Whether the blueprint is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    
    def add_error(self, error: str):
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a validation warning."""
        self.warnings.append(warning)
    
    def add_recommendation(self, recommendation: str):
        """Add an improvement recommendation."""
        self.recommendations.append(recommendation)


# Update forward references
BlueprintSection.model_rebuild()
SectionTree.model_rebuild()

