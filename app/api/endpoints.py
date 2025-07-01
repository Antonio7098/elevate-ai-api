from fastapi import APIRouter, HTTPException, status, Path
from app.api.schemas import (
    DeconstructRequest,
    DeconstructResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    GenerateNotesRequest,
    GenerateQuestionsRequest,
    GenerateQuestionsFromBlueprintDto,
    QuestionSetResponseDto,
    EvaluateAnswerDto,
    EvaluateAnswerResponseDto,
    ErrorResponse
)
from app.core.deconstruction import deconstruct_text
from app.core.chat import process_chat_message
from app.core.indexing import generate_notes, generate_questions, generate_questions_from_blueprint, evaluate_answer
from app.core.usage_tracker import usage_tracker
from typing import Dict, Any, Optional
import uuid
from datetime import datetime, date

router = APIRouter()


@router.post("/deconstruct", response_model=DeconstructResponse)
async def deconstruct_endpoint(request: DeconstructRequest):
    """
    Deconstruct raw text into a structured LearningBlueprint.
    
    This is the core endpoint that transforms raw educational content into
    a structured JSON blueprint containing knowledge primitives and relationships.
    """
    try:
        # Use the actual deconstruction logic
        blueprint = await deconstruct_text(request.source_text, request.source_type_hint)
        
        return DeconstructResponse(
            blueprint_id=blueprint.source_id,
            source_text=request.source_text,
            blueprint_json=blueprint.model_dump(),
            created_at=datetime.utcnow().isoformat(),
            status="completed"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deconstruction failed: {str(e)}"
        )


@router.post("/chat/message", response_model=ChatMessageResponse)
async def chat_endpoint(request: ChatMessageRequest):
    """
    Process a chat message with the AI assistant.
    
    Provides intelligent responses using RAG (Retrieval-Augmented Generation)
    based on the user's knowledge base and conversation context.
    """
    try:
        # TODO: Implement actual chat logic
        response_content = "This is a placeholder response. Chat functionality will be implemented in future sprints."
        
        return ChatMessageResponse(
            role="assistant",
            content=response_content,
            retrieved_context=[]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}"
        )


@router.post("/generate/notes")
async def generate_notes_endpoint(request: GenerateNotesRequest):
    """
    Generate personalized notes from a LearningBlueprint.
    
    Creates tailored notes based on the blueprint content and user preferences.
    """
    try:
        # TODO: Implement actual note generation logic
        note_id = str(uuid.uuid4())
        
        return {
            "note_id": note_id,
            "name": request.name,
            "content": "Placeholder note content",
            "blueprint_id": request.blueprint_id,
            "folder_id": request.folder_id,
            "created_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Note generation failed: {str(e)}"
        )


@router.post("/generate/questions")
async def generate_questions_endpoint(request: GenerateQuestionsRequest):
    """
    Generate question sets from a LearningBlueprint.
    
    Creates personalized questions for spaced repetition and assessment.
    """
    try:
        # TODO: Implement actual question generation logic
        question_set_id = str(uuid.uuid4())
        
        return {
            "question_set_id": question_set_id,
            "name": request.name,
            "questions": [],
            "blueprint_id": request.blueprint_id,
            "folder_id": request.folder_id,
            "created_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question generation failed: {str(e)}"
        )


@router.post("/ai-rag/learning-blueprints/{blueprint_id}/question-sets", response_model=QuestionSetResponseDto)
async def generate_questions_from_blueprint_endpoint(
    request: GenerateQuestionsFromBlueprintDto,
    blueprint_id: str = Path(..., description="ID of the LearningBlueprint to use for question generation")
):
    """
    Generate question sets from a LearningBlueprint.
    
    Creates personalized questions for spaced repetition and assessment based on
    the content of a specific LearningBlueprint.
    """
    try:
        # Call the actual question generation logic
        result = await generate_questions_from_blueprint(
            blueprint_id=blueprint_id,
            name=request.name,
            folder_id=request.folder_id,
            question_options=request.question_options
        )
        
        # Convert the result to QuestionDto objects
        from app.api.schemas import QuestionDto
        
        questions = []
        for q in result.get("questions", []):
            questions.append(QuestionDto(
                text=q.get("text", ""),
                answer=q.get("answer", ""),
                question_type=q.get("question_type", "understand"),
                total_marks_available=q.get("total_marks_available", 1),
                marking_criteria=q.get("marking_criteria", "")
            ))
        
        return QuestionSetResponseDto(
            id=result.get("id", 1),
            name=result.get("name", request.name),
            blueprint_id=result.get("blueprint_id", blueprint_id),
            folder_id=result.get("folder_id"),
            questions=questions,
            created_at=result.get("created_at", datetime.utcnow().isoformat()),
            updated_at=result.get("updated_at", datetime.utcnow().isoformat())
        )
        
    except ValueError as e:
        # Handle validation errors from the DTO
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except HTTPException:
        # Re-raise HTTPExceptions as-is (e.g., from the core function)
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question generation failed: {str(e)}"
        )


@router.post("/suggest/inline")
async def inline_suggestions_endpoint(request: Dict[str, Any]):
    """
    Provide real-time suggestions during note-taking.
    
    This is a highly optimized, low-latency endpoint for providing
    suggestions as users type in the note editor.
    """
    try:
        # TODO: Implement actual inline suggestion logic
        return {
            "suggestions": [],
            "links": [],
            "completions": []
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inline suggestions failed: {str(e)}"
        )


@router.post("/ai/evaluate-answer", response_model=EvaluateAnswerResponseDto)
async def evaluate_answer_endpoint(request: EvaluateAnswerDto):
    """
    Evaluate a user's answer to a question using AI.
    
    This endpoint evaluates a user's answer against the expected answer and
    marking criteria, returning marks achieved and feedback.
    """
    try:
        # Call the actual answer evaluation logic
        result = await evaluate_answer(
            question_id=request.question_id,
            user_answer=request.user_answer
        )
        
        return EvaluateAnswerResponseDto(
            corrected_answer=result.get("corrected_answer", ""),
            marks_available=result.get("marks_available", 0),
            marks_achieved=result.get("marks_achieved", 0)
        )
        
    except ValueError as e:
        # Handle validation errors from the DTO
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except HTTPException:
        # Re-raise HTTPExceptions as-is (e.g., from the core function)
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer evaluation failed: {str(e)}"
        )


@router.get("/usage")
async def get_usage_stats(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get LLM API usage statistics.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    """
    try:
        # Parse dates if provided
        start = None
        end = None
        if start_date:
            start = date.fromisoformat(start_date)
        if end_date:
            end = date.fromisoformat(end_date)
        
        # Get usage summary
        summary = usage_tracker.get_usage_summary(start, end)
        
        return {
            "summary": summary,
            "period": {
                "start_date": start_date,
                "end_date": end_date
            }
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date format: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get usage stats: {str(e)}"
        )


@router.get("/usage/recent")
async def get_recent_usage(limit: int = 50):
    """
    Get recent LLM API usage records.
    
    Args:
        limit: Maximum number of records to return (default: 50)
    """
    try:
        recent_usage = usage_tracker.get_recent_usage(limit)
        return {
            "recent_usage": recent_usage,
            "count": len(recent_usage)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recent usage: {str(e)}"
        ) 