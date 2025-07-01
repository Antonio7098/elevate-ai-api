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
    message_content: str = Field(..., description="User's message content")
    context: Optional[Dict[str, Any]] = Field(None, description="Context for the conversation")


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
class EvaluateAnswerDto(BaseModel):
    """Request schema for the /api/ai/evaluate-answer endpoint."""
    question_id: int = Field(..., description="The ID of the question to be evaluated")
    user_answer: str = Field(..., description="The answer provided by the user")
    
    @field_validator('question_id')
    @classmethod
    def validate_question_id(cls, v):
        if v <= 0:
            raise ValueError('Question ID must be a positive integer')
        return v
    
    @field_validator('user_answer')
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