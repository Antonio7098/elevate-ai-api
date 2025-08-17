"""
Data models for the Note Creation Agent.
Defines request/response schemas and data structures.
Updated to align with new NoteSection schema.
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
    """Request for agentic note editing with enhanced granularity."""
    note_id: int = Field(..., description="ID of note to edit (integer from NoteSection)")
    blueprint_section_id: int = Field(..., description="ID of blueprint section this note belongs to")
    edit_instruction: str = Field(..., description="Natural language editing instruction")
    
    # Enhanced granularity options
    edit_type: Literal[
        # Note-level (existing)
        "rewrite", "expand", "condense", "restructure", "clarify",
        # Line-level (new)
        "edit_line", "add_line", "remove_line", "replace_line",
        # Section-level (new)
        "edit_section", "add_section", "remove_section", "reorder_sections",
        # Block-level (new)
        "edit_block", "add_block", "remove_block", "move_block"
    ] = "rewrite"
    
    # Granularity-specific fields
    target_line_number: Optional[int] = Field(None, description="Target line number for line-level edits")
    target_section_title: Optional[str] = Field(None, description="Target section title for section-level edits")
    target_block_id: Optional[str] = Field(None, description="Target block ID for block-level edits")
    insertion_position: Optional[int] = Field(None, description="Position for insertions (line number or section order)")
    
    # Content for additions/replacements
    new_content: Optional[str] = Field(None, description="New content to add or replace")
    
    # General options
    preserve_original_structure: bool = Field(True, description="Keep original organization")
    preserve_context: bool = Field(True, description="Maintain context with surrounding content")
    include_reasoning: bool = Field(False, description="Include AI reasoning for changes")
    user_preferences: Optional[UserPreferences] = Field(None, description="User preferences for editing")


class GranularEditResult(BaseModel):
    """Result of a granular edit operation."""
    edit_type: str
    target_position: Optional[int] = None
    target_identifier: Optional[str] = None  # line number, section title, or block ID
    original_content: Optional[str] = None
    new_content: Optional[str] = None
    context_preserved: bool = True
    surrounding_context: Optional[str] = None


class NoteGenerationResponse(BaseModel):
    """Response from note generation."""
    success: bool
    note_content: Optional[str] = Field(None, description="Generated note content in BlockNote format")
    plain_text: Optional[str] = Field(None, description="Plain text version of note")
    blueprint_id: Optional[int] = Field(None, description="ID of created/linked blueprint")
    blueprint_section_id: Optional[int] = Field(None, description="ID of blueprint section")
    note_section_id: Optional[int] = Field(None, description="ID of created note section")
    chunks_processed: Optional[List[SourceChunk]] = Field(None, description="Source chunks used")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    message: str = Field(..., description="Success/error message")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ContentConversionResponse(BaseModel):
    """Response from content conversion."""
    success: bool
    converted_content: Optional[str] = Field(None, description="Content in BlockNote format")
    plain_text: Optional[str] = Field(None, description="Plain text version")
    blueprint_id: Optional[int] = Field(None, description="ID of created blueprint")
    blueprint_section_id: Optional[int] = Field(None, description="ID of blueprint section")
    note_section_id: Optional[int] = Field(None, description="ID of created note section")
    conversion_notes: Optional[str] = Field(None, description="Notes about conversion process")
    message: str = Field(..., description="Success/error message")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class NoteEditingResponse(BaseModel):
    """Response from note editing with granular edit details."""
    success: bool
    edited_content: Optional[str] = Field(None, description="Edited note in BlockNote format")
    plain_text: Optional[str] = Field(None, description="Plain text version")
    edit_summary: Optional[str] = Field(None, description="Summary of changes made")
    reasoning: Optional[str] = Field(None, description="AI reasoning for changes")
    content_version: Optional[int] = Field(None, description="New content version after editing")
    
    # Granular edit details
    granular_edits: List[GranularEditResult] = Field(default_factory=list, description="Details of granular edits made")
    edit_positions: List[int] = Field(default_factory=list, description="Positions where edits were made")
    
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
    note_id: int = Field(..., description="ID of note section")
    blueprint_section_id: int = Field(..., description="ID of blueprint section")
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
    blueprint_id: int = Field(..., description="ID of created blueprint")
    blueprint_section_id: Optional[int] = Field(None, description="ID of created blueprint section")
    blueprint_summary: str
    knowledge_primitives: List[str]
    cross_references: List[str]
    processing_time: float
    message: str


class NoteSectionContext(BaseModel):
    """Context information for a note section."""
    note_section_id: int
    blueprint_section_id: int
    blueprint_id: int
    section_hierarchy: List[Dict[str, Any]] = Field(default_factory=list, description="Parent sections hierarchy")
    related_notes: List[Dict[str, Any]] = Field(default_factory=list, description="Related notes in same section")
    knowledge_primitives: List[str] = Field(default_factory=list, description="Knowledge primitives in this section")
