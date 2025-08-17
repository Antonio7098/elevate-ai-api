"""
Generation Orchestrator API Endpoints

Provides REST API endpoints for the sequential generation workflow:
source → blueprint → sections → primitives → mastery criteria → questions
with user editing capabilities between each step.
"""

import logging
import uuid
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from app.core.generation_orchestrator import (
    generation_orchestrator, 
    UserEditRequest, 
    GenerationStep,
    GenerationStatus
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/orchestrator", tags=["generation-orchestrator"])


# Request/Response Models
class StartGenerationRequest(BaseModel):
    """Request to start a new generation session."""
    source_content: str = Field(..., description="Raw source content to process")
    source_type: str = Field(..., description="Type of source (e.g., 'textbook', 'article')")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User learning preferences")
    session_title: Optional[str] = Field(None, description="Optional session title")


class StartGenerationResponse(BaseModel):
    """Response from starting a generation session."""
    session_id: str
    current_step: GenerationStep
    status: GenerationStatus
    message: str
    next_actions: List[str]


class ProceedToNextStepRequest(BaseModel):
    """Request to proceed to the next generation step."""
    session_id: str = Field(..., description="Session identifier")


class UserEditContentRequest(BaseModel):
    """Request for user to edit generated content."""
    step: GenerationStep = Field(..., description="Current generation step")
    content_id: str = Field(..., description="ID of content to edit")
    edited_content: Dict[str, Any] = Field(..., description="Edited content")
    user_notes: Optional[str] = Field(None, description="User notes about the edit")


class GenerationProgressResponse(BaseModel):
    """Response showing current generation progress."""
    session_id: str
    current_step: GenerationStep
    status: GenerationStatus
    completed_steps: List[GenerationStep]
    current_content: Optional[Dict[str, Any]]
    user_edits: List[Dict[str, Any]]
    errors: List[str]
    started_at: str
    last_updated: str
    next_actions: List[str]


class CompleteLearningPathResponse(BaseModel):
    """Response containing the complete generated learning path."""
    session_id: str
    status: str
    generated_at: str
    completed_at: str
    content: Dict[str, Any]
    user_edits: List[Dict[str, Any]]


def get_current_user():
    """Get current user (placeholder for authentication)."""
    return {"id": "debug-user", "username": "debug"}


@router.post("/start", response_model=StartGenerationResponse)
async def start_generation_session(
    request: StartGenerationRequest,
    current_user: dict = Depends(get_current_user)
) -> StartGenerationResponse:
    """
    Start a new sequential generation session.
    
    This begins the workflow:
    source → blueprint → sections → primitives → mastery criteria → questions
    """
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        logger.info(f"Starting generation session {session_id} for user {current_user.get('id')}")
        
        # Start the generation session
        progress = await generation_orchestrator.start_generation_session(
            session_id=session_id,
            source_content=request.source_content,
            source_type=request.source_type,
            user_preferences=request.user_preferences
        )
        
        # Determine next actions
        next_actions = _get_next_actions(progress)
        
        return StartGenerationResponse(
            session_id=session_id,
            current_step=progress.current_step,
            status=progress.status,
            message=f"Generation session started successfully. Current step: {progress.current_step.value}",
            next_actions=next_actions
        )
        
    except Exception as e:
        logger.error(f"Failed to start generation session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start generation session: {str(e)}"
        )


@router.post("/proceed", response_model=GenerationProgressResponse)
async def proceed_to_next_step(
    request: ProceedToNextStepRequest,
    current_user: dict = Depends(get_current_user)
) -> GenerationProgressResponse:
    """
    Proceed to the next step in the generation workflow.
    
    This moves the session from the current step to the next step
    in the sequence: source → blueprint → sections → primitives → mastery criteria → questions
    """
    try:
        logger.info(f"Proceeding to next step for session {request.session_id}")
        
        # Proceed to next step
        progress = await generation_orchestrator.proceed_to_next_step(request.session_id)
        
        # Determine next actions
        next_actions = _get_next_actions(progress)
        
        return GenerationProgressResponse(
            session_id=progress.session_id,
            current_step=progress.current_step,
            status=progress.status,
            completed_steps=progress.completed_steps,
            current_content=progress.current_content,
            user_edits=[edit.dict() for edit in progress.user_edits],
            errors=progress.errors,
            started_at=progress.started_at.isoformat(),
            last_updated=progress.last_updated.isoformat(),
            next_actions=next_actions
        )
        
    except ValueError as e:
        logger.warning(f"Invalid request to proceed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to proceed to next step: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to proceed to next step: {str(e)}"
        )


@router.post("/edit", response_model=GenerationProgressResponse)
async def edit_generated_content(
    session_id: str,
    edit_request: UserEditContentRequest,
    current_user: dict = Depends(get_current_user)
) -> GenerationProgressResponse:
    """
    Edit generated content at the current step.
    
    This allows users to modify the generated content before proceeding
    to the next step in the workflow.
    """
    try:
        logger.info(f"User editing content for session {session_id}, step {edit_request.step}")
        
        # Create user edit request
        user_edit = UserEditRequest(
            step=edit_request.step,
            content_id=edit_request.content_id,
            edited_content=edit_request.edited_content,
            user_notes=edit_request.user_notes
        )
        
        # Apply user edits
        progress = await generation_orchestrator.user_edit_content(session_id, user_edit)
        
        # Determine next actions
        next_actions = _get_next_actions(progress)
        
        return GenerationProgressResponse(
            session_id=progress.session_id,
            current_step=progress.current_step,
            status=progress.status,
            completed_steps=progress.completed_steps,
            current_content=progress.current_content,
            user_edits=[edit.dict() for edit in progress.user_edits],
            errors=progress.errors,
            started_at=progress.started_at.isoformat(),
            last_updated=progress.last_updated.isoformat(),
            next_actions=next_actions
        )
        
    except ValueError as e:
        logger.warning(f"Invalid edit request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to apply user edits: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply user edits: {str(e)}"
        )


@router.get("/progress/{session_id}", response_model=GenerationProgressResponse)
async def get_generation_progress(
    session_id: str,
    current_user: dict = Depends(get_current_user)
) -> GenerationProgressResponse:
    """
    Get current progress for a generation session.
    
    This returns the current status, completed steps, and generated content
    for the specified session.
    """
    try:
        logger.info(f"Getting progress for session {session_id}")
        
        # Get current progress
        progress = await generation_orchestrator.get_generation_progress(session_id)
        
        # Determine next actions
        next_actions = _get_next_actions(progress)
        
        return GenerationProgressResponse(
            session_id=progress.session_id,
            current_step=progress.current_step,
            status=progress.status,
            completed_steps=progress.completed_steps,
            current_content=progress.current_content,
            user_edits=[edit.dict() for edit in progress.user_edits],
            errors=progress.errors,
            started_at=progress.started_at.isoformat(),
            last_updated=progress.last_updated.isoformat(),
            next_actions=next_actions
        )
        
    except ValueError as e:
        logger.warning(f"Session not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to get progress: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get progress: {str(e)}"
        )


@router.get("/complete/{session_id}", response_model=CompleteLearningPathResponse)
async def get_complete_learning_path(
    session_id: str,
    current_user: dict = Depends(get_current_user)
) -> CompleteLearningPathResponse:
    """
    Get the complete generated learning path.
    
    This returns all generated content once the session is complete:
    blueprint, sections, primitives, mastery criteria, questions, and notes.
    """
    try:
        logger.info(f"Getting complete learning path for session {session_id}")
        
        # Get complete learning path
        complete_path = await generation_orchestrator.get_complete_learning_path(session_id)
        
        return CompleteLearningPathResponse(**complete_path)
        
    except ValueError as e:
        logger.warning(f"Session not complete: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to get complete learning path: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get complete learning path: {str(e)}"
        )


@router.delete("/session/{session_id}")
async def delete_generation_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a generation session.
    
    This removes the session and all associated data.
    """
    try:
        logger.info(f"Deleting generation session {session_id}")
        
        # Remove session from orchestrator
        if session_id in generation_orchestrator.generation_sessions:
            del generation_orchestrator.generation_sessions[session_id]
        
        return {"message": f"Session {session_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}"
        )


def _get_next_actions(progress) -> List[str]:
    """Determine what actions the user can take next."""
    actions = []
    
    if progress.status == GenerationStatus.READY_FOR_NEXT:
        if progress.current_step == GenerationStep.COMPLETE:
            actions.append("View complete learning path")
        else:
            actions.append("Proceed to next step")
            actions.append("Edit current content")
    
    elif progress.status == GenerationStatus.USER_EDITING:
        actions.append("Continue editing")
        actions.append("Mark edits complete")
    
    elif progress.status == GenerationStatus.FAILED:
        actions.append("Review errors")
        actions.append("Restart session")
    
    actions.append("View current progress")
    actions.append("Delete session")
    
    return actions
