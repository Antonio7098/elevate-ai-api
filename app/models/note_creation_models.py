"""
Data models for the Note Creation Agent.
Defines request/response schemas and data structures.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


class ContentFormat(str, Enum):
    """Supported content formats for input conversion."""
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    HTML = "html"
    RICH_TEXT = "rich_text"


class NoteStyle(str, Enum):
    """Note generation styles."""
    CONCISE = "concise"
    DETAILED = "detailed"
    BULLET_POINTS = "bullet_points"
    NARRATIVE = "narrative"
    ACADEMIC = "academic"
    PRACTICAL = "practical"


class UserPreferences(BaseModel):
    """User preferences for note generation."""
    preferred_style: NoteStyle = NoteStyle.CONCISE
    include_examples: bool = True
    include_definitions: bool = True
    max_note_length: Optional[int] = 2000
    focus_on_key_concepts: bool = True
    include_cross_references: bool = False


class ChunkingStrategy(BaseModel):
    """Configuration for source chunking."""
    max_chunk_size: Optional[int] = Field(8000, description="Maximum tokens per chunk")
    chunk_overlap: int = Field(500, description="Overlap between chunks for context")
    semantic_boundaries: bool = Field(True, description="Prefer topic-based breaks")
    preserve_structure: bool = Field(True, description="Maintain hierarchical relationships")
    use_algorithmic_detection: bool = Field(True, description="Use fast font/markup detection")
    llm_validation_threshold: float = Field(0.8, description="Confidence level for LLM review")


class SourceChunk(BaseModel):
    """Represents a chunk of source content."""
    chunk_id: str
    content: str
    start_position: int
    end_position: int
    topic: str
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = Field(default_factory=list)
    cross_references: List[str] = Field(default_factory=list)


class AlgorithmicDetectionResult(BaseModel):
    """Result of algorithmic section detection."""
    detected_sections: List[Dict[str, Any]]
    confidence_scores: List[float]
    suggested_chunks: List[SourceChunk]
    needs_llm_validation: bool


class NoteGenerationRequest(BaseModel):
    """Request for generating notes from source."""
    source_content: str = Field(..., description="Source text to generate notes from")
    note_style: NoteStyle = NoteStyle.CONCISE
    user_preferences: UserPreferences = Field(default_factory=UserPreferences)
    target_length: Optional[int] = Field(None, description="Target note length in words")
    focus_areas: List[str] = Field(default_factory=list, description="Specific areas to focus on")
    create_blueprint: bool = Field(True, description="Always create blueprint for RAG context")
    chunking_strategy: Optional[ChunkingStrategy] = Field(None, description="Chunking configuration")


class ContentToNoteRequest(BaseModel):
    """Request for converting user content to notes."""
    user_content: str = Field(..., description="User's input content")
    content_format: ContentFormat = Field(..., description="Format of input content")
    note_style: NoteStyle = NoteStyle.CONCISE
    user_preferences: UserPreferences = Field(default_factory=UserPreferences)
    create_blueprint: bool = Field(True, description="Extract knowledge primitives for blueprint")


class InputConversionRequest(BaseModel):
    """Request for converting input to BlockNote format."""
    input_content: str = Field(..., description="Content to convert")
    input_format: ContentFormat = Field(..., description="Current format of input")
    target_format: Literal["blocknote"] = "blocknote"
    preserve_structure: bool = Field(True, description="Maintain document structure")
    include_metadata: bool = Field(False, description="Include content metadata")


class NoteEditingRequest(BaseModel):
    """Request for agentic note editing."""
    note_id: str = Field(..., description="ID of note to edit")
    edit_instruction: str = Field(..., description="Natural language editing instruction")
    edit_type: Literal["rewrite", "expand", "condense", "restructure", "clarify"] = "rewrite"
    preserve_original_structure: bool = Field(True, description="Keep original organization")
    include_reasoning: bool = Field(False, description="Include AI reasoning for changes")


class NoteGenerationResponse(BaseModel):
    """Response from note generation."""
    success: bool
    note_content: Optional[str] = Field(None, description="Generated note content in BlockNote format")
    plain_text: Optional[str] = Field(None, description="Plain text version of note")
    blueprint_id: Optional[str] = Field(None, description="ID of created/linked blueprint")
    chunks_processed: Optional[List[SourceChunk]] = Field(None, description="Source chunks used")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    message: str = Field(..., description="Success/error message")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ContentConversionResponse(BaseModel):
    """Response from content conversion."""
    success: bool
    converted_content: Optional[str] = Field(None, description="Content in BlockNote format")
    plain_text: Optional[str] = Field(None, description="Plain text version")
    blueprint_id: Optional[str] = Field(None, description="ID of created blueprint")
    conversion_notes: Optional[str] = Field(None, description="Notes about conversion process")
    message: str = Field(..., description="Success/error message")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class NoteEditingResponse(BaseModel):
    """Response from note editing."""
    success: bool
    edited_content: Optional[str] = Field(None, description="Edited note in BlockNote format")
    plain_text: Optional[str] = Field(None, description="Plain text version")
    edit_summary: Optional[str] = Field(None, description="Summary of changes made")
    reasoning: Optional[str] = Field(None, description="AI reasoning for changes")
    message: str = Field(..., description="Success/error message")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class EditingSuggestion(BaseModel):
    """Suggestion for note editing."""
    suggestion_id: str
    type: Literal["grammar", "clarity", "structure", "content", "style"]
    description: str
    suggested_change: str
    confidence: float
    reasoning: str


class NoteEditingSuggestionsResponse(BaseModel):
    """Response with editing suggestions."""
    success: bool
    suggestions: List[EditingSuggestion]
    note_id: str
    message: str


class ChunkingResult(BaseModel):
    """Result of source chunking process."""
    success: bool
    chunks: List[SourceChunk]
    total_chunks: int
    processing_strategy: str
    llm_validation_used: bool
    processing_time: float
    message: str


class BlueprintCreationResult(BaseModel):
    """Result of blueprint creation process."""
    success: bool
    blueprint_id: str
    blueprint_summary: str
    knowledge_primitives: List[str]
    cross_references: List[str]
    processing_time: float
    message: str
