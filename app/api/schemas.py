from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Dict, Any, Literal


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
    sourceContent: str = Field(..., description="Source text content to generate questions from")
    questionCount: Optional[int] = Field(5, description="Number of questions to generate")
    questionTypes: Optional[List[str]] = Field(default_factory=lambda: ["short_answer", "multiple_choice"], description="Types of questions to generate")
    difficultyLevel: Optional[str] = Field("intermediate", description="Difficulty level for questions")
    topicFocus: Optional[str] = Field(None, description="Specific topic focus for questions")
    includeAnswerKeys: Optional[bool] = Field(True, description="Whether to include answer keys")
    question_options: Optional[Dict[str, Any]] = Field(None, description="Additional options for question generation")


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
    criterion_id: Optional[str] = Field(None, description="ID of mastery criterion this question tests")
    uee_level: Optional[str] = Field(None, description="UEE level this question targets")
    question_complexity: Optional[int] = Field(None, ge=1, le=5, description="Question complexity level (1=basic, 5=advanced)")
    estimated_time_minutes: Optional[int] = Field(None, ge=1, le=60, description="Estimated time to answer in minutes")


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
    blueprint_id: str = Field(..., description="ID of the blueprint to index")
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


# Primitive-Centric API Schemas
class MasteryCriterionDto(BaseModel):
    """Schema for mastery criteria in API requests/responses (Core API Prisma compatible)."""
    criterionId: str = Field(..., description="Unique criterion ID")
    primitiveId: Optional[str] = Field(None, description="Primitive ID this criterion belongs to")
    title: str = Field(..., description="Criterion title")
    description: Optional[str] = Field(None, description="Criterion description")
    ueeLevel: Literal["UNDERSTAND", "USE", "EXPLORE"] = Field(..., description="UEE level")
    weight: float = Field(..., description="Criterion importance weight")
    isRequired: bool = Field(default=True, description="Whether criterion is required")
    
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
    
    @field_validator('weight')
    @classmethod
    def validate_weight_range(cls, v):
        if v < 1.0 or v > 5.0:
            raise ValueError('Weight must be between 1.0 and 5.0')
        return v


class KnowledgePrimitiveDto(BaseModel):
    """Schema for knowledge primitives in API requests/responses (Core API Prisma compatible)."""
    primitiveId: str = Field(..., description="Unique primitive ID (matches Prisma primitiveId)")
    title: str = Field(..., description="Primitive title")
    description: Optional[str] = Field(None, description="Primitive description")
    primitiveType: str = Field(..., description="Primitive type: fact, concept, process (matches Prisma)")
    difficultyLevel: str = Field(..., description="Difficulty level: beginner, intermediate, advanced")
    estimatedTimeMinutes: Optional[int] = Field(None, description="Estimated time in minutes")
    trackingIntensity: Literal["DENSE", "NORMAL", "SPARSE"] = Field(default="NORMAL", description="Tracking intensity")
    masteryCriteria: List[MasteryCriterionDto] = Field(default_factory=list, description="Associated mastery criteria")
    
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


class BlueprintPrimitivesRequest(BaseModel):
    """Request schema for getting primitives from a blueprint."""
    include_criteria: bool = Field(default=True, description="Include mastery criteria in response")
    primitive_types: Optional[List[str]] = Field(None, description="Filter by primitive types")
    blueprintId: str = Field(..., description="Blueprint ID to sync")
    primitives: List[KnowledgePrimitiveDto] = Field(..., description="Primitives to sync")


# Sprint 31 API Schemas for Primitive Services

class PrimitiveGenerationRequest(BaseModel):
    """Request schema for generating primitives with mastery criteria from source content."""
    sourceContent: str = Field(..., description="Raw source content to analyze")
    sourceType: str = Field(..., description="Type of source (textbook, article, video, etc.)")
    userPreferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User learning preferences")
    targetPrimitiveCount: Optional[int] = Field(default=None, description="Target number of primitives to generate")
    generateCriteria: bool = Field(default=True, description="Whether to generate mastery criteria")
    criteriaPerPrimitive: int = Field(default=4, description="Target criteria per primitive")

    @field_validator('sourceType')
    @classmethod
    def validate_source_type(cls, v):
        valid_types = ["textbook", "article", "video", "lecture", "documentation", "tutorial", "other"]
        if v not in valid_types:
            raise ValueError(f'Source type must be one of: {", ".join(valid_types)}')
        return v


class PrimitiveGenerationResponse(BaseModel):
    """Response schema for primitive generation."""
    success: bool = Field(..., description="Whether generation was successful")
    primitives: List[KnowledgePrimitiveDto] = Field(..., description="Generated primitives with criteria")
    generatedCount: int = Field(..., description="Number of primitives generated")
    totalCriteria: int = Field(..., description="Total number of criteria across all primitives")
    processingTime: float = Field(default=0.0, description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    warnings: List[str] = Field(default_factory=list, description="Any generation warnings")


class CriterionQuestionRequest(BaseModel):
    """Request schema for generating questions mapped to mastery criteria."""
    criterionIds: List[str] = Field(..., description="Criterion IDs to generate questions for")
    criteria: List[MasteryCriterionDto] = Field(..., description="Mastery criteria data")
    primitive: KnowledgePrimitiveDto = Field(..., description="Parent primitive context")
    sourceContent: str = Field(..., description="Source content for context")
    questionsPerCriterion: int = Field(default=3, description="Number of questions per criterion")
    userPreferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User preferences")
    useSemanticMapping: bool = Field(default=True, description="Use semantic mapping for question-criterion alignment")
    difficultyPreference: str = Field(default="intermediate", description="Preferred question difficulty")

    @field_validator('difficultyPreference')
    @classmethod
    def validate_difficulty_preference(cls, v):
        valid_difficulties = ["beginner", "intermediate", "advanced", "expert"]
        if v not in valid_difficulties:
            raise ValueError(f'Difficulty preference must be one of: {", ".join(valid_difficulties)}')
        return v

    @field_validator('questionsPerCriterion')
    @classmethod
    def validate_questions_per_criterion(cls, v):
        if v < 1 or v > 10:
            raise ValueError('Questions per criterion must be between 1 and 10')
        return v


class CriterionQuestionDto(BaseModel):
    """Data Transfer Object for criterion-based questions."""
    questionId: str = Field(..., description="Unique question identifier")
    questionText: str = Field(..., description="Question text")
    questionType: str = Field(..., description="Type of question")
    correctAnswer: str = Field(..., description="Correct answer")
    options: List[str] = Field(default_factory=list, description="Multiple choice options")
    explanation: Optional[str] = Field(None, description="Answer explanation")
    difficulty: str = Field(default="intermediate", description="Question difficulty")
    estimatedTime: int = Field(default=120, description="Estimated time in seconds")
    tags: List[str] = Field(default_factory=list, description="Question tags")
    criterionId: Optional[str] = Field(None, description="Associated criterion ID")
    primitiveId: Optional[str] = Field(None, description="Associated primitive ID")
    ueeLevel: Optional[str] = Field(None, description="UEE level")
    weight: float = Field(default=1.0, description="Question weight")

    @field_validator('questionType')
    @classmethod
    def validate_question_type(cls, v):
        valid_types = [
            "multiple_choice", "true_false", "fill_blank", "definition", "matching",
            "problem_solving", "application", "calculation", "scenario", "case_study",
            "analysis", "synthesis", "evaluation", "design", "critique"
        ]
        if v not in valid_types:
            raise ValueError(f'Question type must be one of: {", ".join(valid_types)}')
        return v

    @field_validator('difficulty')
    @classmethod
    def validate_difficulty(cls, v):
        valid_difficulties = ["beginner", "intermediate", "advanced", "expert"]
        if v not in valid_difficulties:
            raise ValueError(f'Difficulty must be one of: {", ".join(valid_difficulties)}')
        return v

    @field_validator('estimatedTime')
    @classmethod
    def validate_estimated_time(cls, v):
        if v < 10 or v > 1800:  # 10 seconds to 30 minutes
            raise ValueError('Estimated time must be between 10 and 1800 seconds')
        return v


class CriterionQuestionResponse(BaseModel):
    """Response schema for criterion question generation."""
    success: bool = Field(..., description="Whether generation was successful")
    questionsByCriterion: Dict[str, List[CriterionQuestionDto]] = Field(..., description="Questions mapped by criterion ID")
    totalQuestions: int = Field(..., description="Total number of questions generated")
    mappingConfidence: float = Field(default=0.0, description="Average mapping confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    warnings: List[str] = Field(default_factory=list, description="Any generation warnings")


class SyncStatusResponse(BaseModel):
    """Response schema for synchronization status."""
    success: bool = Field(..., description="Whether operation was successful")
    status: str = Field(..., description="Current sync status")
    message: str = Field(..., description="Status message")
    primitivesProcessed: int = Field(default=0, description="Number of primitives processed")
    criteriaProcessed: int = Field(default=0, description="Number of criteria processed")
    questionsProcessed: int = Field(default=0, description="Number of questions processed")
    estimatedCompletion: Optional[int] = Field(None, description="Estimated completion time in seconds")
    lastSync: Optional[str] = Field(None, description="Last sync timestamp")
    errors: List[str] = Field(default_factory=list, description="Any sync errors")
    warnings: List[str] = Field(default_factory=list, description="Any sync warnings")

    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        valid_statuses = ["pending", "in_progress", "completed", "failed", "cancelled"]
        if v not in valid_statuses:
            raise ValueError(f'Status must be one of: {", ".join(valid_statuses)}')
        return v


class MappingValidationResponse(BaseModel):
    """Response schema for question-criterion mapping validation."""
    success: bool = Field(..., description="Whether validation was successful")
    validationIssues: Dict[str, List[str]] = Field(..., description="Issues by criterion ID")
    mappingStatistics: Dict[str, Any] = Field(..., description="Mapping quality statistics")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    overallQuality: str = Field(..., description="Overall mapping quality")

    @field_validator('overallQuality')
    @classmethod
    def validate_overall_quality(cls, v):
        valid_qualities = ["excellent", "good", "fair", "poor"]
        if v not in valid_qualities:
            raise ValueError(f'Overall quality must be one of: {", ".join(valid_qualities)}')
        return v


# Sprint 32 API Schemas for Blueprint Primitive Data Access

class BlueprintPrimitivesResponse(BaseModel):
    """Response schema for blueprint primitive data access."""
    success: bool = Field(..., description="Whether retrieval was successful")
    blueprintId: str = Field(..., description="Blueprint ID")
    primitives: List[KnowledgePrimitiveDto] = Field(..., description="Core API compatible primitives")
    primitiveCount: int = Field(..., description="Number of primitives")
    totalCriteria: int = Field(..., description="Total number of criteria")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    warnings: List[str] = Field(default_factory=list, description="Any retrieval warnings")


class BatchBlueprintPrimitivesRequest(BaseModel):
    """Request schema for batch blueprint primitive retrieval."""
    blueprintIds: List[str] = Field(..., description="List of blueprint IDs to process")
    includeMetadata: bool = Field(default=True, description="Include metadata in response")
    validateCompatibility: bool = Field(default=True, description="Validate Core API compatibility")
    
    @field_validator('blueprintIds')
    @classmethod
    def validate_blueprint_ids(cls, v):
        if len(v) == 0:
            raise ValueError('At least one blueprint ID is required')
        if len(v) > 50:  # Reasonable batch limit
            raise ValueError('Maximum 50 blueprint IDs per batch request')
        return v


class BatchBlueprintPrimitivesResponse(BaseModel):
    """Response schema for batch blueprint primitive retrieval."""
    success: bool = Field(..., description="Whether batch operation was successful")
    results: Dict[str, List[KnowledgePrimitiveDto]] = Field(..., description="Results by blueprint ID")
    totalPrimitives: int = Field(..., description="Total primitives across all blueprints")
    totalCriteria: int = Field(..., description="Total criteria across all primitives")
    processedCount: int = Field(..., description="Number of successfully processed blueprints")
    failedCount: int = Field(..., description="Number of failed blueprints")
    errors: List[str] = Field(default_factory=list, description="Any processing errors")


class PrimitiveValidationRequest(BaseModel):
    """Request schema for primitive validation."""
    validateSchema: bool = Field(default=True, description="Validate Prisma schema compliance")
    validateIntegrity: bool = Field(default=True, description="Validate data integrity")
    validateUeeDistribution: bool = Field(default=True, description="Validate UEE level distribution")
    strictMode: bool = Field(default=False, description="Enable strict validation mode")


class PrimitiveValidationResponse(BaseModel):
    """Response schema for primitive validation."""
    success: bool = Field(..., description="Whether validation was successful")
    isValid: bool = Field(..., description="Whether primitives are valid")
    validationQuality: str = Field(..., description="Overall validation quality")
    primitiveCount: int = Field(..., description="Number of primitives validated")
    criteriaCount: int = Field(..., description="Number of criteria validated")
    issues: List[str] = Field(default_factory=list, description="Validation issues")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Validation metadata")
    
    @field_validator('validationQuality')
    @classmethod
    def validate_validation_quality(cls, v):
        valid_qualities = ["excellent", "good", "fair", "poor"]
        if v not in valid_qualities:
            raise ValueError(f'Validation quality must be one of: {", ".join(valid_qualities)}')
        return v


# Sprint 32 API Schemas for Core API Compatible Question Generation

class CoreApiQuestionRequest(BaseModel):
    """Request schema for generating Core API compatible questions."""
    criterionId: str = Field(..., description="Core API criterion ID")
    criterionTitle: str = Field(..., description="Criterion title")
    criterionDescription: Optional[str] = Field(None, description="Criterion description")
    primitiveId: str = Field(..., description="Core API primitive ID")
    primitiveTitle: str = Field(..., description="Primitive title")
    primitiveDescription: Optional[str] = Field(None, description="Primitive description")
    ueeLevel: str = Field(..., description="UEE level (UNDERSTAND|USE|EXPLORE)")
    weight: float = Field(..., description="Criterion weight (1.0-5.0)")
    isRequired: bool = Field(default=True, description="Whether criterion is required")
    sourceContent: str = Field(..., description="Source content for context")
    questionCount: int = Field(default=3, description="Number of questions to generate")
    difficultyLevel: Optional[str] = Field(default="intermediate", description="Difficulty level")
    userPreferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User preferences")
    
    @field_validator('ueeLevel')
    @classmethod
    def validate_uee_level(cls, v):
        valid_levels = ['UNDERSTAND', 'USE', 'EXPLORE']
        if v not in valid_levels:
            raise ValueError(f'UEE level must be one of: {", ".join(valid_levels)}')
        return v
    
    @field_validator('weight')
    @classmethod
    def validate_weight(cls, v):
        if not (1.0 <= v <= 5.0):
            raise ValueError('Weight must be between 1.0 and 5.0')
        return v
    
    @field_validator('questionCount')
    @classmethod
    def validate_question_count(cls, v):
        if not (1 <= v <= 10):
            raise ValueError('Question count must be between 1 and 10')
        return v


class CoreApiQuestionDto(BaseModel):
    """Core API compatible question DTO."""
    questionId: str = Field(..., description="Unique question identifier")
    questionText: str = Field(..., description="Question text")
    questionType: str = Field(..., description="Type of question")
    correctAnswer: str = Field(..., description="Correct answer")
    options: List[str] = Field(default_factory=list, description="Multiple choice options")
    explanation: Optional[str] = Field(None, description="Answer explanation")
    difficulty: str = Field(default="intermediate", description="Question difficulty")
    estimatedTime: int = Field(default=120, description="Estimated time in seconds")
    tags: List[str] = Field(default_factory=list, description="Question tags")
    criterionId: str = Field(..., description="Core API criterion ID")
    primitiveId: str = Field(..., description="Core API primitive ID")
    ueeLevel: str = Field(..., description="UEE level")
    weight: float = Field(default=1.0, description="Question weight")
    coreApiCompatible: bool = Field(default=True, description="Core API compatibility flag")
    
    @field_validator('ueeLevel')
    @classmethod
    def validate_uee_level(cls, v):
        valid_levels = ['UNDERSTAND', 'USE', 'EXPLORE']
        if v not in valid_levels:
            raise ValueError(f'UEE level must be one of: {", ".join(valid_levels)}')
        return v


class CoreApiQuestionResponse(BaseModel):
    """Response schema for Core API question generation."""
    success: bool = Field(..., description="Whether generation was successful")
    questions: List[CoreApiQuestionDto] = Field(..., description="Generated questions")
    questionCount: int = Field(..., description="Number of questions generated")
    criterionId: str = Field(..., description="Core API criterion ID")
    primitiveId: str = Field(..., description="Core API primitive ID")
    ueeLevel: str = Field(..., description="UEE level")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    warnings: List[str] = Field(default_factory=list, description="Any generation warnings")


class BatchCoreApiQuestionRequest(BaseModel):
    """Request schema for batch Core API question generation."""
    criterionRequests: List[CoreApiQuestionRequest] = Field(..., description="List of criterion requests")
    sourceContent: str = Field(..., description="Source content for context")
    questionsPerCriterion: int = Field(default=3, description="Questions per criterion")
    userPreferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User preferences")
    
    @field_validator('criterionRequests')
    @classmethod
    def validate_criterion_requests(cls, v):
        if len(v) == 0:
            raise ValueError('At least one criterion request is required')
        if len(v) > 20:  # Reasonable batch limit
            raise ValueError('Maximum 20 criterion requests per batch')
        return v


class BatchCoreApiQuestionResponse(BaseModel):
    """Response schema for batch Core API question generation."""
    success: bool = Field(..., description="Whether batch operation was successful")
    results: Dict[str, List[CoreApiQuestionDto]] = Field(..., description="Results by criterion ID")
    totalQuestions: int = Field(..., description="Total questions generated")
    processedCount: int = Field(..., description="Number of successfully processed criteria")
    failedCount: int = Field(..., description="Number of failed criteria")
    errors: List[str] = Field(default_factory=list, description="Any processing errors")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CoreApiQuestionValidationRequest(BaseModel):
    """Request schema for Core API question validation."""
    questions: List[CoreApiQuestionDto] = Field(..., description="Questions to validate")
    strictMode: bool = Field(default=False, description="Enable strict validation mode")
    validatePrismaSchema: bool = Field(default=True, description="Validate Prisma schema compatibility")
    
    @field_validator('questions')
    @classmethod
    def validate_questions(cls, v):
        if len(v) == 0:
            raise ValueError('At least one question is required for validation')
        return v


class CoreApiQuestionValidationResponse(BaseModel):
    """Response schema for Core API question validation."""
    success: bool = Field(..., description="Whether validation was successful")
    isValid: bool = Field(..., description="Whether all questions are valid")
    validationQuality: str = Field(..., description="Overall validation quality")
    totalQuestions: int = Field(..., description="Total questions validated")
    validQuestions: int = Field(..., description="Number of valid questions")
    invalidQuestions: int = Field(..., description="Number of invalid questions")
    issues: List[str] = Field(default_factory=list, description="Validation issues")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Validation metadata")
    format_for_core_api: bool = Field(default=False, description="Format response for Core API storage")
    
    @field_validator('validationQuality')
    @classmethod
    def validate_validation_quality(cls, v):
        valid_qualities = ["excellent", "good", "fair", "poor"]
        if v not in valid_qualities:
            raise ValueError(f'Validation quality must be one of: {", ".join(valid_qualities)}')
        return v


class BlueprintPrimitivesResponse(BaseModel):
    """Response schema for blueprint primitives."""
    blueprint_id: str = Field(..., description="Source blueprint ID")
    blueprint_title: str = Field(..., description="Blueprint title")
    primitives: List[KnowledgePrimitiveDto] = Field(..., description="Extracted primitives")
    total_primitives: int = Field(..., description="Total number of primitives")
    mastery_criteria_coverage: Dict[str, Any] = Field(..., description="Coverage analysis")
    created_at: str = Field(..., description="When primitives were extracted")


class CriterionQuestionRequest(BaseModel):
    """Request schema for generating questions for specific mastery criteria."""
    primitive_id: str = Field(..., description="Target primitive ID")
    criterion_id: str = Field(..., description="Target mastery criterion ID")
    question_count: int = Field(default=3, ge=1, le=10, description="Number of questions")
    question_types: Optional[List[str]] = Field(None, description="Preferred question types")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    
    @field_validator('question_types')
    @classmethod
    def validate_question_types(cls, v):
        if v is not None:
            valid_types = ["multiple_choice", "short_answer", "essay", "true_false", "fill_blank"]
            invalid_types = [t for t in v if t not in valid_types]
            if invalid_types:
                raise ValueError(f'Invalid question types: {", ".join(invalid_types)}. Valid types: {", ".join(valid_types)}')
        return v


class CriterionQuestionResponse(BaseModel):
    """Response schema for criterion-specific questions."""
    primitive_id: str = Field(..., description="Source primitive ID")
    criterion_id: str = Field(..., description="Source criterion ID")
    criterion_description: str = Field(..., description="What the criterion tests")
    uee_level: str = Field(..., description="UEE level of the criterion")
    questions: List[CriterionQuestionDto] = Field(..., description="Generated questions")
    quality_score: float = Field(..., description="Overall question quality score")
    created_at: str = Field(..., description="When questions were generated")


class EnhancedDeconstructRequest(BaseModel):
    """Enhanced request schema for blueprint creation with primitive generation."""
    source_text: str = Field(..., description="Raw text content to be deconstructed")
    source_type_hint: Optional[str] = Field(None, description="Hint about the type of source")
    generate_mastery_criteria: bool = Field(default=True, description="Generate mastery criteria for primitives")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User learning preferences")
    primitive_options: Optional[Dict[str, Any]] = Field(None, description="Primitive generation options")
    sync_to_core_api: bool = Field(default=False, description="Automatically sync to Core API")
    
    @field_validator('source_text')
    @classmethod
    def validate_source_text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Source text cannot be empty')
        return v.strip()


class EnhancedDeconstructResponse(BaseModel):
    """Enhanced response schema for blueprint creation with primitives."""
    blueprint_id: str = Field(..., description="Unique identifier for the generated blueprint")
    source_text: str = Field(..., description="Original source text")
    blueprint_json: Dict[str, Any] = Field(..., description="Generated LearningBlueprint JSON")
    primitives: List[KnowledgePrimitiveDto] = Field(..., description="Extracted primitives with mastery criteria")
    mastery_criteria_coverage: Dict[str, Any] = Field(..., description="Coverage analysis")
    quality_metrics: Dict[str, Any] = Field(..., description="Quality assessment metrics")
    core_api_sync_status: Optional[str] = Field(None, description="Core API synchronization status")
    created_at: str = Field(..., description="Timestamp of creation")
    status: str = Field(..., description="Status of the deconstruction process") 
# Import from answer evaluation schemas
from app.api.answer_evaluation_schemas import (
    PrismaCriterionEvaluationRequest,
    PrismaCriterionEvaluationResponse
)
from app.api.criterion_question_schemas import CriterionQuestionDto


# Blueprint Section Schemas
class BlueprintSectionRequest(BaseModel):
    """Request schema for blueprint section operations."""
    title: str = Field(..., description="Section title")
    description: Optional[str] = Field(None, description="Section description")
    content: Optional[str] = Field(None, description="Section content")
    order_index: Optional[int] = Field(None, description="Order index within parent")
    parent_section_id: Optional[int] = Field(None, description="Parent section ID")
    difficulty_level: Optional[str] = Field("intermediate", description="Difficulty level")
    estimated_time_minutes: Optional[int] = Field(None, description="Estimated completion time")
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if not v or not v.strip():
            raise ValueError('Section title cannot be empty')
        return v.strip()
    
    @field_validator('difficulty_level')
    @classmethod
    def validate_difficulty_level(cls, v):
        valid_levels = ["beginner", "intermediate", "advanced", "expert"]
        if v not in valid_levels:
            raise ValueError(f'Difficulty level must be one of: {", ".join(valid_levels)}')
        return v


class BlueprintSectionResponse(BaseModel):
    """Response schema for blueprint section operations."""
    id: int = Field(..., description="Section ID")
    title: str = Field(..., description="Section title")
    description: Optional[str] = Field(None, description="Section description")
    content: Optional[str] = Field(None, description="Section content")
    order_index: int = Field(..., description="Order index within parent")
    depth: int = Field(..., description="Hierarchical depth")
    parent_section_id: Optional[int] = Field(None, description="Parent section ID")
    blueprint_id: str = Field(..., description="Associated blueprint ID")
    difficulty_level: str = Field(..., description="Difficulty level")
    estimated_time_minutes: Optional[int] = Field(None, description="Estimated completion time")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")


class BlueprintSectionTreeResponse(BaseModel):
    """Response schema for blueprint section hierarchy tree."""
    blueprint_id: str = Field(..., description="Blueprint ID")
    sections: List[BlueprintSectionResponse] = Field(..., description="All sections")
    hierarchy: Dict[str, Any] = Field(..., description="Hierarchical structure")
    total_sections: int = Field(..., description="Total number of sections")
    max_depth: int = Field(..., description="Maximum hierarchy depth")
    created_at: str = Field(..., description="When tree was generated")


class SectionMoveRequest(BaseModel):
    """Request schema for moving sections within hierarchy."""
    section_id: int = Field(..., description="Section to move")
    new_parent_id: Optional[int] = Field(None, description="New parent section ID")
    new_order_index: Optional[int] = Field(None, description="New order index")
    
    @field_validator('section_id')
    @classmethod
    def validate_section_id(cls, v):
        if v <= 0:
            raise ValueError('Section ID must be a positive integer')
        return v


class SectionReorderRequest(BaseModel):
    """Request schema for reordering sections."""
    section_orders: List[Dict[str, int]] = Field(..., description="List of section ID to order index mappings")
    
    @field_validator('section_orders')
    @classmethod
    def validate_section_orders(cls, v):
        if not v:
            raise ValueError('Section orders cannot be empty')
        for item in v:
            if not isinstance(item, dict) or 'section_id' not in item or 'order_index' not in item:
                raise ValueError('Each item must contain section_id and order_index')
        return v


class SectionContentRequest(BaseModel):
    """Request schema for retrieving section content."""
    section_id: int = Field(..., description="Section ID")
    include_metadata: bool = Field(default=True, description="Include section metadata")
    include_primitives: bool = Field(default=True, description="Include associated primitives")
    include_criteria: bool = Field(default=True, description="Include mastery criteria")
    
    @field_validator('section_id')
    @classmethod
    def validate_section_id(cls, v):
        if v <= 0:
            raise ValueError('Section ID must be a positive integer')
        return v


class SectionContentResponse(BaseModel):
    """Response schema for section content."""
    section: BlueprintSectionResponse = Field(..., description="Section information")
    primitives: List[Dict[str, Any]] = Field(default_factory=list, description="Associated primitives")
    mastery_criteria: List[Dict[str, Any]] = Field(default_factory=list, description="Mastery criteria")
    content_summary: Optional[str] = Field(None, description="Content summary")
    learning_progress: Optional[Dict[str, Any]] = Field(None, description="Learning progress data")
    related_sections: List[Dict[str, Any]] = Field(default_factory=list, description="Related sections")


class SectionStatsResponse(BaseModel):
    """Response schema for section statistics."""
    section_id: int = Field(..., description="Section ID")
    total_primitives: int = Field(..., description="Total primitives in section")
    total_criteria: int = Field(..., description="Total mastery criteria")
    difficulty_distribution: Dict[str, int] = Field(..., description="Difficulty level distribution")
    uue_stage_distribution: Dict[str, int] = Field(..., description="UUE stage distribution")
    estimated_completion_time: int = Field(..., description="Total estimated completion time")
    created_at: str = Field(..., description="When stats were generated")


class BlueprintSectionSyncRequest(BaseModel):
    """Request schema for syncing blueprint sections with Core API."""
    blueprint_id: str = Field(..., description="Blueprint ID to sync")
    sections: List[BlueprintSectionRequest] = Field(..., description="Sections to sync")
    user_id: str = Field(..., description="User ID for ownership")
    force_update: bool = Field(default=False, description="Force update existing sections")
    
    @field_validator('blueprint_id')
    @classmethod
    def validate_blueprint_id(cls, v):
        if not v or not v.strip():
            raise ValueError('Blueprint ID cannot be empty')
        return v.strip()


class BlueprintSectionSyncResponse(BaseModel):
    """Response schema for blueprint section sync operations."""
    blueprint_id: str = Field(..., description="Blueprint ID")
    sync_success: bool = Field(..., description="Overall sync success")
    sections_created: int = Field(..., description="Number of sections created")
    sections_updated: int = Field(..., description="Number of sections updated")
    errors: List[str] = Field(default_factory=list, description="Sync errors")
    created_section_ids: List[int] = Field(default_factory=list, description="IDs of created sections")
    updated_section_ids: List[int] = Field(default_factory=list, description="IDs of updated sections")
    sync_timestamp: str = Field(..., description="When sync was completed")

# Section-Aware Primitive Schemas

class SectionPrimitivesRequest(BaseModel):
    """Request model for section-specific primitive operations."""
    section_id: str = Field(..., description="ID of the section")
    include_metadata: bool = Field(True, description="Include primitive metadata")
    include_criteria: bool = Field(True, description="Include mastery criteria")
    include_relationships: bool = Field(False, description="Include primitive relationships")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "section_id": "section-123",
                "include_metadata": True,
                "include_criteria": True,
                "include_relationships": False
            }
        }
    )

class SectionPrimitivesResponse(BaseModel):
    """Response model for section-specific primitive operations."""
    section_id: str = Field(..., description="ID of the section")
    blueprint_id: str = Field(..., description="ID of the blueprint")
    primitives: List[KnowledgePrimitiveDto] = Field(..., description="Section primitives")
    mastery_criteria: List[MasteryCriterionDto] = Field(..., description="Section mastery criteria")
    total_primitives: int = Field(..., description="Total number of primitives")
    total_criteria: int = Field(..., description="Total number of mastery criteria")
    section_info: Dict[str, Any] = Field(..., description="Section metadata")
    extraction_timestamp: str = Field(..., description="When primitives were extracted")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "section_id": "section-123",
                "blueprint_id": "blueprint-456",
                "primitives": [],
                "mastery_criteria": [],
                "total_primitives": 0,
                "total_criteria": 0,
                "section_info": {
                    "section_title": "Introduction",
                    "section_depth": 0,
                    "parent_section_id": None
                },
                "extraction_timestamp": "2025-08-13T14:45:00Z"
            }
        }
    )

class SectionSearchRequest(BaseModel):
    """Request model for section-specific search operations."""
    query: str = Field(..., description="Search query")
    search_type: str = Field("semantic", description="Type of search: semantic, vector, or hybrid")
    limit: int = Field(10, description="Maximum number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional search filters")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "machine learning algorithms",
                "search_type": "semantic",
                "limit": 10,
                "filters": {
                    "primitive_type": "process",
                    "difficulty_level": "intermediate"
                }
            }
        }
    )

class SectionSearchResponse(BaseModel):
    """Response model for section-specific search operations."""
    query: str = Field(..., description="Search query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_type: str = Field(..., description="Type of search performed")
    section_context: Dict[str, Any] = Field(..., description="Section context for search")
    search_metadata: Dict[str, Any] = Field(default_factory=dict, description="Search metadata")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "machine learning algorithms",
                "results": [],
                "total_results": 0,
                "search_type": "semantic",
                "section_context": {
                    "section_id": "section-123",
                    "section_title": "Introduction",
                    "blueprint_id": "blueprint-456"
                },
                "search_metadata": {
                    "search_duration": 0.15,
                    "vector_store_type": "pinecone"
                }
            }
        }
    )
