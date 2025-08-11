"""
API endpoints for the Note Creation Agent.
Provides endpoints for note generation, content conversion, and editing.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List

from app.models.note_creation_models import (
    NoteGenerationRequest, NoteGenerationResponse,
    ContentToNoteRequest, ContentConversionResponse,
    InputConversionRequest, NoteEditingRequest, NoteEditingResponse,
    NoteEditingSuggestionsResponse
)
from app.core.note_services.note_agent_orchestrator import NoteAgentOrchestrator
from app.services.llm_service import create_llm_service

# Create router
router = APIRouter(prefix="/api/v1", tags=["note-creation"])

# Initialize orchestrator with LLM service
llm_service = create_llm_service(provider="gemini")  # Use Gemini by default, falls back to mock
orchestrator = NoteAgentOrchestrator(llm_service)


@router.post("/generate-notes-from-source", response_model=NoteGenerationResponse)
async def generate_notes_from_source(
    request: NoteGenerationRequest
) -> NoteGenerationResponse:
    """
    Generate notes from source content via blueprint creation.
    
    This endpoint:
    1. Chunks the source content using hybrid approach
    2. Creates a learning blueprint from the chunks
    3. Generates structured notes from the blueprint
    4. Links notes to the blueprint for RAG context
    """
    try:
        response = await orchestrator.create_notes_from_source(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate notes from source: {str(e)}"
        )


@router.post("/generate-notes-from-content", response_model=ContentConversionResponse)
async def generate_notes_from_content(
    request: ContentToNoteRequest
) -> ContentConversionResponse:
    """
    Generate notes from user content via blueprint creation.
    
    This endpoint:
    1. Creates a learning blueprint from user input
    2. Generates structured notes from the blueprint
    3. Links notes to the blueprint for RAG context
    """
    try:
        response = await orchestrator.create_notes_from_content(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate notes from content: {str(e)}"
        )


@router.post("/convert-input-to-blocks", response_model=ContentConversionResponse)
async def convert_input_to_blocks(
    request: InputConversionRequest
) -> ContentConversionResponse:
    """
    Convert user input directly to BlockNote format.
    
    This endpoint:
    1. Converts input from various formats to BlockNote JSON
    2. Preserves document structure if requested
    3. Optionally includes metadata
    """
    try:
        response = await orchestrator.convert_input_to_blocknote(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to convert input to BlockNote format: {str(e)}"
        )


@router.post("/edit-note-agentically", response_model=NoteEditingResponse)
async def edit_note_agentically(
    request: NoteEditingRequest
) -> NoteEditingResponse:
    """
    Edit a note using AI agentic capabilities.
    
    This endpoint:
    1. Analyzes the current note content
    2. Applies the requested edit instruction
    3. Returns the edited note with reasoning
    """
    try:
        response = await orchestrator.edit_note_agentically(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to edit note agentically: {str(e)}"
        )


@router.get("/note-editing-suggestions", response_model=NoteEditingSuggestionsResponse)
async def get_note_editing_suggestions(
    note_id: str,
    include_grammar: bool = True,
    include_clarity: bool = True,
    include_structure: bool = True
) -> NoteEditingSuggestionsResponse:
    """
    Get AI-powered editing suggestions for a note.
    
    This endpoint:
    1. Analyzes the note content
    2. Generates improvement suggestions
    3. Provides confidence scores and reasoning
    """
    try:
        response = await orchestrator.get_editing_suggestions(
            note_id, include_grammar, include_clarity, include_structure
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get editing suggestions: {str(e)}"
        )


@router.post("/batch-process-notes")
async def batch_process_notes(
    requests: List[dict],
    workflow_type: str = "auto"
):
    """
    Process multiple note creation requests in batch.
    
    This endpoint:
    1. Accepts multiple note creation requests
    2. Processes them efficiently with proper resource management
    3. Returns batch processing results with success/failure details
    
    Args:
        requests: List of note creation request objects
        workflow_type: Type of workflow to apply ("auto", "source", "content", "conversion")
    """
    try:
        # Validate workflow type
        valid_types = ["auto", "source", "content", "conversion"]
        if workflow_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid workflow_type. Must be one of: {valid_types}"
            )
        
        response = await orchestrator.batch_process_notes(requests, workflow_type)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to batch process notes: {str(e)}"
        )


@router.get("/workflow-status")
async def get_workflow_status():
    """
    Get the current status of all note creation workflows and services.
    
    This endpoint provides:
    1. Service availability status
    2. Workflow status information
    3. System health indicators
    """
    try:
        response = await orchestrator.get_workflow_status()
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get workflow status: {str(e)}"
        )


@router.get("/service-info")
async def get_service_info():
    """
    Get information about the Note Creation Agent service.
    
    This endpoint provides:
    1. Service capabilities and features
    2. Supported formats and note styles
    3. Version and description information
    """
    try:
        response = orchestrator.get_service_info()
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get service info: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for the note creation agent."""
    try:
        # Get basic health status
        status = await orchestrator.get_workflow_status()
        
        return {
            "status": "healthy" if status.get('orchestrator_status') == 'healthy' else "degraded",
            "service": "note-creation-agent",
            "version": "1.0.0",
            "endpoints": [
                "POST /api/v1/generate-notes-from-source",
                "POST /api/v1/generate-notes-from-content",
                "POST /api/v1/convert-input-to-blocks",
                "POST /api/v1/edit-note-agentically",
                "GET /api/v1/note-editing-suggestions",
                "POST /api/v1/batch-process-notes",
                "GET /api/v1/workflow-status",
                "GET /api/v1/service-info"
            ],
            "services_status": status.get('services', {}),
            "workflows_status": status.get('workflows', {})
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "note-creation-agent",
            "version": "1.0.0",
            "error": str(e)
        }
