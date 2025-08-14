"""
Enhanced Content Generation Models for AI API

This module defines models for AI-powered content generation that aligns with
the blueprint-centric architecture, including UUE stage progression and mastery
criteria generation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from .blueprint_centric import UueStage, DifficultyLevel, AssessmentType


class ContentType(str, Enum):
    """Types of content that can be generated."""
    MASTERY_CRITERIA = "mastery_criteria"
    QUESTIONS = "questions"
    SECTIONS = "sections"
    PRIMITIVES = "primitives"
    RELATIONSHIPS = "relationships"


class GenerationStyle(str, Enum):
    """Content generation style preferences."""
    CONCISE = "concise"
    THOROUGH = "thorough"
    EXPLORATIVE = "explorative"
    PRACTICAL = "practical"


class QuestionType(str, Enum):
    """Types of questions that can be generated."""
    MULTIPLE_CHOICE = "multiple_choice"
    FILL_BLANK = "fill_blank"
    SHORT_ANSWER = "short_answer"
    EXPLANATION = "explanation"
    APPLICATION = "application"


# Content Generation Request Models
class ContentGenerationRequest(BaseModel):
    """Request for AI-powered content generation."""
    blueprint_id: int = Field(..., description="Target blueprint ID")
    section_id: Optional[int] = Field(None, description="Target section ID")
    content_type: ContentType = Field(..., description="Type of content to generate")
    user_id: int = Field(..., description="User requesting generation")
    
    # Generation instructions
    style: GenerationStyle = Field(default=GenerationStyle.THOROUGH, description="Content generation style")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.BEGINNER, description="Target difficulty level")
    target_uue_stage: Optional[UueStage] = Field(None, description="Target UUE stage for content")
    
    # Content constraints
    max_items: int = Field(default=10, description="Maximum number of items to generate")
    
    @field_validator('max_items')
    @classmethod
    def validate_max_items(cls, v):
        if v < 1 or v > 100:
            raise ValueError('Max items must be between 1 and 100')
        return v


class MasteryCriteriaGenerationRequest(ContentGenerationRequest):
    """Request for mastery criteria generation."""
    content_type: ContentType = Field(default=ContentType.MASTERY_CRITERIA, description="Content type")
    target_mastery_threshold: float = Field(default=0.8, description="Target mastery threshold")
    balance_uue_stages: bool = Field(default=True, description="Balance UUE stages")
    
    @field_validator('target_mastery_threshold')
    @classmethod
    def validate_mastery_threshold(cls, v):
        valid_thresholds = [0.6, 0.8, 0.95]
        if v not in valid_thresholds:
            raise ValueError(f'Mastery threshold must be one of: {valid_thresholds}')
        return v


class QuestionGenerationRequest(ContentGenerationRequest):
    """Request for question generation."""
    content_type: ContentType = Field(default=ContentType.QUESTIONS, description="Content type")
    question_types: List[QuestionType] = Field(default_factory=list, description="Preferred question types")
    include_explanations: bool = Field(default=True, description="Include answer explanations")
    generate_question_families: bool = Field(default=True, description="Generate question families")
    variations_per_family: int = Field(default=3, description="Variations per question family")
    
    @field_validator('variations_per_family')
    @classmethod
    def validate_variations_per_family(cls, v):
        if v < 1 or v > 10:
            raise ValueError('Variations per family must be between 1 and 10')
        return v


# Content Generation Response Models
class ContentGenerationResponse(BaseModel):
    """Response from AI-powered content generation."""
    request: ContentGenerationRequest = Field(..., description="Original generation request")
    success: bool = Field(..., description="Whether generation was successful")
    content: Optional[Dict[str, Any]] = Field(None, description="Generated content")
    message: str = Field(..., description="Response message")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    created_at: datetime = Field(default_factory=datetime.now, description="Response timestamp")


# Generated Content Models
class GeneratedMasteryCriterion(BaseModel):
    """Generated mastery criterion."""
    title: str = Field(..., description="Criterion title")
    description: str = Field(..., description="Criterion description")
    uue_stage: UueStage = Field(..., description="UUE stage")
    weight: float = Field(default=1.0, description="Importance weight")
    complexity_score: float = Field(default=5.0, description="Complexity score")
    assessment_type: AssessmentType = Field(default=AssessmentType.QUESTION_BASED, description="Assessment type")
    mastery_threshold: float = Field(default=0.8, description="Mastery threshold")
    
    @field_validator('weight')
    @classmethod
    def validate_weight(cls, v):
        if v < 1.0 or v > 5.0:
            raise ValueError('Weight must be between 1.0 and 5.0')
        return v


class GeneratedQuestion(BaseModel):
    """Generated question."""
    question_text: str = Field(..., description="Question text")
    answer: str = Field(..., description="Correct answer")
    explanation: Optional[str] = Field(None, description="Answer explanation")
    question_type: QuestionType = Field(..., description="Question type")
    difficulty: DifficultyLevel = Field(..., description="Question difficulty")
    uue_stage: UueStage = Field(..., description="Target UUE stage")
    mastery_criterion_id: Optional[str] = Field(None, description="Associated mastery criterion")


class QuestionFamily(BaseModel):
    """Family of related questions for a mastery criterion."""
    id: str = Field(..., description="Question family ID")
    mastery_criterion_id: str = Field(..., description="Associated mastery criterion")
    base_question: str = Field(..., description="Base question text")
    variations: List[GeneratedQuestion] = Field(default_factory=list, description="Question variations")
    difficulty: DifficultyLevel = Field(..., description="Family difficulty level")
    question_type: QuestionType = Field(..., description="Question type")
    uue_stage: UueStage = Field(..., description="Target UUE stage")
