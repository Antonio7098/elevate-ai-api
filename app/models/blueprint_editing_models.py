"""
Blueprint Editing Models

Defines the data models for blueprint editing requests and responses,
including blueprints, primitives, mastery criteria, and questions.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================================
# BLUEPRINT EDITING MODELS
# ============================================================================

class BlueprintEditingRequest(BaseModel):
    """Request model for editing a blueprint."""
    blueprint_id: int = Field(..., description="ID of the blueprint to edit")
    edit_type: str = Field(..., description="Type of edit operation")
    edit_instruction: str = Field(..., description="Detailed editing instructions")
    preserve_original_structure: bool = Field(True, description="Whether to preserve original structure")
    include_reasoning: bool = Field(False, description="Whether to include AI reasoning")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences for editing")


class BlueprintEditingResponse(BaseModel):
    """Response model for blueprint editing operations."""
    success: bool = Field(..., description="Whether the operation was successful")
    edited_content: Optional[Dict[str, Any]] = Field(None, description="The edited blueprint content")
    edit_summary: Optional[str] = Field(None, description="Summary of changes made")
    reasoning: Optional[str] = Field(None, description="AI reasoning for the changes")
    version: Optional[int] = Field(None, description="New version number")
    granular_edits: Optional[List[Any]] = Field(None, description="List of granular edits made")
    processing_time: Optional[float] = Field(None, description="Time taken to process the edit")
    message: str = Field(..., description="Human-readable message about the operation")


class BlueprintEditingSuggestionsResponse(BaseModel):
    """Response model for blueprint editing suggestions."""
    success: bool = Field(..., description="Whether suggestions were generated successfully")
    suggestions: List[Any] = Field(..., description="List of editing suggestions")
    blueprint_id: int = Field(..., description="ID of the blueprint")
    message: str = Field(..., description="Human-readable message about the suggestions")


# ============================================================================
# PRIMITIVE EDITING MODELS
# ============================================================================

class PrimitiveEditingRequest(BaseModel):
    """Request model for editing a knowledge primitive."""
    primitive_id: int = Field(..., description="ID of the primitive to edit")
    edit_type: str = Field(..., description="Type of edit operation")
    edit_instruction: str = Field(..., description="Detailed editing instructions")
    preserve_original_structure: bool = Field(True, description="Whether to preserve original structure")
    include_reasoning: bool = Field(False, description="Whether to include AI reasoning")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences for editing")


class PrimitiveEditingResponse(BaseModel):
    """Response model for primitive editing operations."""
    success: bool = Field(..., description="Whether the operation was successful")
    edited_content: Optional[Dict[str, Any]] = Field(None, description="The edited primitive content")
    edit_summary: Optional[str] = Field(None, description="Summary of changes made")
    reasoning: Optional[str] = Field(None, description="AI reasoning for the changes")
    version: Optional[int] = Field(None, description="New version number")
    processing_time: Optional[float] = Field(None, description="Time taken to process the edit")
    message: str = Field(..., description="Human-readable message about the operation")


class PrimitiveEditingSuggestionsResponse(BaseModel):
    """Response model for primitive editing suggestions."""
    success: bool = Field(..., description="Whether suggestions were generated successfully")
    suggestions: List[Any] = Field(..., description="List of editing suggestions")
    primitive_id: int = Field(..., description="ID of the primitive")
    message: str = Field(..., description="Human-readable message about the suggestions")


# ============================================================================
# MASTERY CRITERION EDITING MODELS
# ============================================================================

class MasteryCriterionEditingRequest(BaseModel):
    """Request model for editing a mastery criterion."""
    criterion_id: int = Field(..., description="ID of the criterion to edit")
    edit_type: str = Field(..., description="Type of edit operation")
    edit_instruction: str = Field(..., description="Detailed editing instructions")
    preserve_original_structure: bool = Field(True, description="Whether to preserve original structure")
    include_reasoning: bool = Field(False, description="Whether to include AI reasoning")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences for editing")


class MasteryCriterionEditingResponse(BaseModel):
    """Response model for mastery criterion editing operations."""
    success: bool = Field(..., description="Whether the operation was successful")
    edited_content: Optional[Dict[str, Any]] = Field(None, description="The edited criterion content")
    edit_summary: Optional[str] = Field(None, description="Summary of changes made")
    reasoning: Optional[str] = Field(None, description="AI reasoning for the changes")
    version: Optional[int] = Field(None, description="New version number")
    processing_time: Optional[float] = Field(None, description="Time taken to process the edit")
    message: str = Field(..., description="Human-readable message about the operation")


class MasteryCriterionEditingSuggestionsResponse(BaseModel):
    """Response model for mastery criterion editing suggestions."""
    success: bool = Field(..., description="Whether suggestions were generated successfully")
    suggestions: List[Any] = Field(..., description="List of editing suggestions")
    criterion_id: int = Field(..., description="ID of the criterion")
    message: str = Field(..., description="Human-readable message about the suggestions")


# ============================================================================
# QUESTION EDITING MODELS
# ============================================================================

class QuestionEditingRequest(BaseModel):
    """Request model for editing a question."""
    question_id: int = Field(..., description="ID of the question to edit")
    edit_type: str = Field(..., description="Type of edit operation")
    edit_instruction: str = Field(..., description="Detailed editing instructions")
    preserve_original_structure: bool = Field(True, description="Whether to preserve original structure")
    include_reasoning: bool = Field(False, description="Whether to include AI reasoning")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences for editing")


class QuestionEditingResponse(BaseModel):
    """Response model for question editing operations."""
    success: bool = Field(..., description="Whether the operation was successful")
    edited_content: Optional[Dict[str, Any]] = Field(None, description="The edited question content")
    edit_summary: Optional[str] = Field(None, description="Summary of changes made")
    reasoning: Optional[str] = Field(None, description="AI reasoning for the changes")
    version: Optional[int] = Field(None, description="New version number")
    processing_time: Optional[float] = Field(None, description="Time taken to process the edit")
    message: str = Field(..., description="Human-readable message about the operation")


class QuestionEditingSuggestionsResponse(BaseModel):
    """Response model for question editing suggestions."""
    success: bool = Field(..., description="Whether suggestions were generated successfully")
    suggestions: List[Any] = Field(..., description="List of editing suggestions")
    question_id: int = Field(..., description="ID of the question")
    message: str = Field(..., description="Human-readable message about the suggestions")


# ============================================================================
# SHARED MODELS
# ============================================================================

class EditingSuggestion(BaseModel):
    """Model for editing suggestions."""
    suggestion_id: str = Field(..., description="Unique identifier for the suggestion")
    type: str = Field(..., description="Type of suggestion (e.g., 'clarity', 'structure')")
    description: str = Field(..., description="Description of the suggested change")
    suggested_change: str = Field(..., description="The specific change to make")
    confidence: float = Field(0.8, description="Confidence level of the suggestion (0.0 to 1.0)")
    reasoning: str = Field(..., description="Reasoning behind the suggestion")
    priority: Optional[str] = Field(None, description="Priority level of the suggestion")


class BlueprintContext(BaseModel):
    """Context information for blueprint editing."""
    blueprint_id: int = Field(..., description="ID of the blueprint")
    title: str = Field(..., description="Title of the blueprint")
    description: Optional[str] = Field(None, description="Description of the blueprint")
    version: int = Field(1, description="Current version of the blueprint")
    sections_count: int = Field(..., description="Number of sections in the blueprint")
    primitives_count: int = Field(..., description="Number of knowledge primitives")
    criteria_count: int = Field(..., description="Number of mastery criteria")
    questions_count: int = Field(..., description="Number of questions")
    difficulty_distribution: Dict[str, float] = Field(..., description="Distribution of difficulty levels")
    estimated_time_hours: int = Field(..., description="Estimated time to complete")
    learning_objectives: List[str] = Field(..., description="Learning objectives")


class GranularEditResult(BaseModel):
    """Result of a granular edit operation."""
    edit_id: str = Field(..., description="Unique identifier for the edit")
    edit_type: str = Field(..., description="Type of edit performed")
    target_position: Optional[str] = Field(None, description="Position where edit was applied")
    original_content: Optional[str] = Field(None, description="Original content before edit")
    new_content: Optional[str] = Field(None, description="New content after edit")
    confidence: float = Field(0.8, description="Confidence in the edit")
    reasoning: str = Field(..., description="Reasoning for the edit")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the edit")


# ============================================================================
# ENUMS
# ============================================================================

class EditType(str):
    """Types of edit operations."""
    # Blueprint edits
    EDIT_SECTION = "edit_section"
    ADD_SECTION = "add_section"
    REMOVE_SECTION = "remove_section"
    REORDER_SECTIONS = "reorder_sections"
    
    # Primitive edits
    EDIT_PRIMITIVE = "edit_primitive"
    ADD_PRIMITIVE = "add_primitive"
    REMOVE_PRIMITIVE = "remove_primitive"
    REORDER_PRIMITIVES = "reorder_primitives"
    
    # Mastery criterion edits
    EDIT_CRITERION = "edit_criterion"
    ADD_CRITERION = "add_criterion"
    REMOVE_CRITERION = "remove_criterion"
    REORDER_CRITERIA = "reorder_criteria"
    
    # Question edits
    EDIT_QUESTION = "edit_question"
    ADD_QUESTION = "add_question"
    REMOVE_QUESTION = "remove_question"
    REORDER_QUESTIONS = "reorder_questions"
    
    # Content edits
    IMPROVE_CLARITY = "improve_clarity"
    IMPROVE_STRUCTURE = "improve_structure"
    ADD_EXAMPLES = "add_examples"
    SIMPLIFY_LANGUAGE = "simplify_language"
    ENHANCE_DETAIL = "enhance_detail"
    CORRECT_ERRORS = "correct_errors"


class SuggestionType(str):
    """Types of editing suggestions."""
    CLARITY = "clarity"
    STRUCTURE = "structure"
    CONTENT = "content"
    GRAMMAR = "grammar"
    COMPLEXITY = "complexity"
    RELATIONSHIPS = "relationships"
    ASSESSMENT = "assessment"
    QUALITY = "quality"





