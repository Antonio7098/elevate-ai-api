from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any


class DeconstructRequest(BaseModel):
    """Request schema for the /deconstruct endpoint."""
    source_text: str = Field(..., description="Raw text content to be deconstructed")
    source_type_hint: Optional[str] = Field(None, description="Hint about the type of source (e.g., chapter, article, video)")


class DeconstructResponse(BaseModel):
    """Response schema for the /deconstruct endpoint."""
    blueprint_id: str = Field(..., description="Unique identifier for the generated blueprint")
    source_text: str = Field(..., description="Original source text")
    blueprint_json: Dict[str, Any] = Field(..., description="Generated LearningBlueprint JSON")
    created_at: str = Field(..., description="Timestamp of creation")
    status: str = Field(..., description="Status of the deconstruction process")


class ChatMessageRequest(BaseModel):
    """Request schema for the /chat endpoint."""
    user_id: str = Field(..., description="User ID")
    message_content: str = Field(..., description="User's message content")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(default=[], description="Previous conversation messages")
    context: Optional[Dict[str, Any]] = Field(None, description="Context for the conversation")
    session_id: str = Field(..., description="Session ID for conversation tracking")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")
    max_tokens: Optional[int] = Field(default=1000, description="Maximum tokens for response")
    temperature: Optional[float] = Field(default=0.7, description="Temperature for response generation")


class ChatMessageResponse(BaseModel):
    """Response schema for the /chat endpoint."""
    role: str = Field(..., description="Role of the message sender (assistant)")
    content: str = Field(..., description="AI assistant's response")
    retrieved_context: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved context used for response")


class GenerateNotesRequest(BaseModel):
    """Request schema for the /generate/notes endpoint."""
    blueprint_id: str = Field(..., description="ID of the LearningBlueprint to use")
    name: str = Field(..., description="Name for the generated note")
    folder_id: Optional[int] = Field(None, description="ID of the folder to store the note")


class GenerateQuestionsRequest(BaseModel):
    """Request schema for the /generate/questions endpoint."""
    blueprint_id: str = Field(..., description="ID of the LearningBlueprint to use")
    name: str = Field(..., description="Name for the generated question set")
    folder_id: Optional[int] = Field(None, description="ID of the folder to store the question set")
    question_options: Optional[Dict[str, Any]] = Field(None, description="Options for question generation")


# New schemas for the question generation endpoint
class GenerateQuestionsFromBlueprintDto(BaseModel):
    """Request schema for the /api/ai-rag/learning-blueprints/:blueprintId/question-sets endpoint."""
    name: str = Field(..., description="The title for the new QuestionSet")
    folder_id: Optional[int] = Field(None, description="ID of the folder to store the new question set in")
    question_options: Optional[Dict[str, Any]] = Field(None, description="Additional parameters to guide the AI's question generation process")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()
    
    @field_validator('folder_id')
    @classmethod
    def validate_folder_id(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Folder ID must be a positive integer')
        return v


class QuestionDto(BaseModel):
    """Schema for individual questions in a QuestionSet."""
    text: str = Field(..., description="The question text")
    answer: str = Field(..., description="The correct answer or explanation")
    question_type: str = Field(..., description="Type of question (understand/use/explore)")
    total_marks_available: int = Field(..., description="Total marks available for this question")
    marking_criteria: str = Field(..., description="Detailed marking criteria for scoring")


class QuestionSetResponseDto(BaseModel):
    """Response schema for the question generation endpoint."""
    id: int = Field(..., description="Unique identifier for the QuestionSet")
    name: str = Field(..., description="Name of the question set")
    blueprint_id: str = Field(..., description="ID of the source LearningBlueprint")
    folder_id: Optional[int] = Field(None, description="ID of the folder containing the question set")
    questions: List[QuestionDto] = Field(..., description="List of generated questions")
    created_at: str = Field(..., description="Timestamp when the question set was created")
    updated_at: str = Field(..., description="Timestamp when the question set was last updated")


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code for client handling")


# New schemas for the answer evaluation endpoint
class QuestionContextDto(BaseModel):
    """Schema for question context sent by Core API."""
    questionId: int = Field(..., description="The ID of the question to be evaluated")
    questionText: str = Field(..., description="The question text")
    expectedAnswer: str = Field(..., description="The expected/correct answer")
    questionType: str = Field(..., description="Type of question (e.g., SHORT_ANSWER, MULTIPLE_CHOICE)")
    marksAvailable: int = Field(..., description="Total marks available for this question")
    markingCriteria: Optional[str] = Field(None, description="Detailed marking criteria for scoring")

class EvaluationContextDto(BaseModel):
    """Schema for evaluation context sent by Core API."""
    questionSetName: Optional[str] = Field(None, description="Name of the question set")
    folderName: Optional[str] = Field(None, description="Name of the folder")
    learningStage: Optional[int] = Field(None, description="Current learning stage")
    conceptTags: Optional[List[str]] = Field(None, description="Concept tags for context")

class EvaluateAnswerDto(BaseModel):
    """Request schema for the /api/ai/evaluate-answer endpoint."""
    questionContext: QuestionContextDto = Field(..., description="Complete question context from Core API")
    userAnswer: str = Field(..., description="The answer provided by the user")
    context: Optional[EvaluationContextDto] = Field(None, description="Additional context for evaluation")
    
    @field_validator('userAnswer')
    @classmethod
    def validate_user_answer(cls, v):
        if not v or not v.strip():
            raise ValueError('User answer cannot be empty')
        return v.strip()


class EvaluateAnswerResponseDto(BaseModel):
    """Response schema for the answer evaluation endpoint."""
    corrected_answer: str = Field(..., description="The ideal/correct answer as determined by the AI")
    marks_available: int = Field(..., description="The total marks available for this question")
    marks_achieved: int = Field(..., description="The marks awarded to the user's answer (rounded integer)")
    feedback: str = Field(..., description="Encouraging feedback and suggestions for improvement")


# New schemas for the blueprint indexing endpoint
class IndexBlueprintRequest(BaseModel):
    """Request schema for the /index-blueprint endpoint."""
    blueprint_json: Dict[str, Any] = Field(..., description="LearningBlueprint JSON to index")
    force_reindex: bool = Field(False, description="Force reindexing even if blueprint already exists")
    
    @field_validator('blueprint_json')
    @classmethod
    def validate_blueprint_json(cls, v):
        if not v or not isinstance(v, dict):
            raise ValueError('Blueprint JSON must be a non-empty dictionary')
        return v


class IndexBlueprintResponse(BaseModel):
    """Response schema for the /index-blueprint endpoint."""
    blueprint_id: str = Field(..., description="ID of the indexed blueprint")
    blueprint_title: str = Field(..., description="Title of the blueprint")
    indexing_completed: bool = Field(..., description="Whether indexing was completed successfully")
    nodes_processed: int = Field(..., description="Number of TextNodes processed")
    embeddings_generated: int = Field(..., description="Number of embeddings generated")
    vectors_stored: int = Field(..., description="Number of vectors stored in database")
    success_rate: float = Field(..., description="Success rate of the indexing operation")
    elapsed_seconds: float = Field(..., description="Time taken for indexing in seconds")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered during indexing")
    created_at: str = Field(..., description="Timestamp when indexing was completed")


class IndexingStatsResponse(BaseModel):
    """Response schema for indexing statistics."""
    total_nodes: int = Field(..., description="Total number of nodes in the vector database")
    total_blueprints: int = Field(..., description="Total number of blueprints indexed")
    blueprint_specific: Optional[Dict[str, Any]] = Field(None, description="Blueprint-specific statistics")
    created_at: str = Field(..., description="Timestamp when stats were generated")


class SearchRequest(BaseModel):
    """Request schema for vector search with metadata filtering."""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    
    # Metadata filtering options
    blueprint_id: Optional[str] = Field(None, description="Filter by specific blueprint ID")
    locus_type: Optional[str] = Field(None, description="Filter by locus type (foundational_concept, use_case, etc.)")
    uue_stage: Optional[str] = Field(None, description="Filter by UUE stage (understand, use, evaluate)")
    
    # Relationship filtering
    related_to_locus: Optional[str] = Field(None, description="Filter by loci related to this locus ID")
    relationship_type: Optional[str] = Field(None, description="Filter by relationship type (prerequisite, supports, etc.)")
    
    # Content filtering
    min_chunk_size: Optional[int] = Field(None, ge=1, description="Minimum chunk size in words")
    max_chunk_size: Optional[int] = Field(None, ge=1, description="Maximum chunk size in words")
    
    @field_validator('locus_type')
    @classmethod
    def validate_locus_type(cls, v):
        if v and v not in ['foundational_concept', 'use_case', 'exploration', 'key_term', 'common_misconception']:
            raise ValueError('Invalid locus type. Must be one of: foundational_concept, use_case, exploration, key_term, common_misconception')
        return v
    
    @field_validator('uue_stage')
    @classmethod
    def validate_uue_stage(cls, v):
        if v and v not in ['understand', 'use', 'evaluate']:
            raise ValueError('Invalid UUE stage. Must be one of: understand, use, evaluate')
        return v


class SearchResultItem(BaseModel):
    """Individual search result item."""
    id: str = Field(..., description="TextNode ID")
    content: str = Field(..., description="Node content")
    score: float = Field(..., description="Similarity score")
    
    # Metadata
    blueprint_id: str = Field(..., description="Source blueprint ID")
    locus_id: str = Field(..., description="Source locus ID")
    locus_type: str = Field(..., description="Type of locus")
    uue_stage: str = Field(..., description="UUE stage")
    
    # Chunk information
    chunk_index: Optional[int] = Field(None, description="Chunk index if content was chunked")
    chunk_total: Optional[int] = Field(None, description="Total chunks for this locus")
    word_count: int = Field(..., description="Word count of the content")
    
    # Relationships
    relationships: List[Dict[str, Any]] = Field(default_factory=list, description="Related loci")
    
    # Timestamps
    created_at: str = Field(..., description="When the node was created")
    indexed_at: str = Field(..., description="When the node was indexed")


class SearchResponse(BaseModel):
    """Response schema for vector search results."""
    results: List[SearchResultItem] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    query: str = Field(..., description="Original search query")
    
    # Applied filters
    filters_applied: Dict[str, Any] = Field(..., description="Filters that were applied")
    
    # Performance metrics
    search_time_ms: float = Field(..., description="Search time in milliseconds")
    embedding_time_ms: float = Field(..., description="Time to generate query embedding")
    
    # Metadata
    created_at: str = Field(..., description="When the search was performed")


class RelatedLocusSearchRequest(BaseModel):
    """Request schema for finding related loci."""
    locus_id: str = Field(..., description="Source locus ID to find relationships for")
    relationship_types: Optional[List[str]] = Field(None, description="Filter by relationship types")
    max_depth: int = Field(1, ge=1, le=3, description="Maximum relationship depth to traverse")
    include_reverse: bool = Field(True, description="Include reverse relationships")
    

class RelatedLocusItem(BaseModel):
    """Related locus information."""
    locus_id: str = Field(..., description="Related locus ID")
    relationship_type: str = Field(..., description="Type of relationship")
    relationship_strength: float = Field(..., description="Strength of the relationship")
    depth: int = Field(..., description="Relationship depth from source")
    path: List[str] = Field(..., description="Path from source to this locus")
    
    # Locus metadata
    locus_type: str = Field(..., description="Type of the related locus")
    blueprint_id: str = Field(..., description="Blueprint containing the locus")
    content_preview: str = Field(..., description="Preview of the locus content")


class RelatedLocusSearchResponse(BaseModel):
    """Response schema for related locus search."""
    source_locus_id: str = Field(..., description="Source locus ID")
    related_loci: List[RelatedLocusItem] = Field(..., description="Related loci")
    total_related: int = Field(..., description="Total number of related loci found")
    max_depth_reached: int = Field(..., description="Maximum depth reached in traversal")
    created_at: str = Field(..., description="When the search was performed") 