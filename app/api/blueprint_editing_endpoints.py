"""
Blueprint Editing API Endpoints

Provides REST API endpoints for editing blueprints, primitives, 
mastery criteria, and questions with AI-powered capabilities.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from app.core.blueprint_editing_service import BlueprintEditingService
from app.models.blueprint_editing_models import (
    BlueprintEditingRequest, BlueprintEditingResponse, BlueprintEditingSuggestionsResponse,
    PrimitiveEditingRequest, PrimitiveEditingResponse, PrimitiveEditingSuggestionsResponse,
    MasteryCriterionEditingRequest, MasteryCriterionEditingResponse, MasteryCriterionEditingSuggestionsResponse,
    QuestionEditingRequest, QuestionEditingResponse, QuestionEditingSuggestionsResponse
)
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_current_user():
    """Get current user (placeholder for authentication)."""
    return {"id": "debug-user", "username": "debug"}

router = APIRouter(prefix="/api/v1/blueprint-editing", tags=["blueprint-editing"])


# ============================================================================
# BLUEPRINT EDITING ENDPOINTS
# ============================================================================

@router.post("/blueprint/edit", response_model=BlueprintEditingResponse)
async def edit_blueprint(
    request: BlueprintEditingRequest,
    current_user: dict = Depends(get_current_user)
) -> BlueprintEditingResponse:
    """
    Edit a blueprint using AI agentic capabilities.
    
    This endpoint provides comprehensive blueprint editing with:
    - Context-aware editing based on blueprint structure
    - Granular editing for sections, primitives, criteria, and questions
    - AI-powered suggestions and reasoning
    - Preservation of original structure when requested
    """
    try:
        llm_service = LLMService()
        blueprint_service = BlueprintEditingService(llm_service)
        
        response = await blueprint_service.edit_blueprint_agentically(request)
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=response.message
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error editing blueprint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/blueprint/{blueprint_id}/suggestions", response_model=BlueprintEditingSuggestionsResponse)
async def get_blueprint_editing_suggestions(
    blueprint_id: int,
    include_structure: bool = True,
    include_content: bool = True,
    include_relationships: bool = True,
    current_user: dict = Depends(get_current_user)
) -> BlueprintEditingSuggestionsResponse:
    """
    Get AI-powered editing suggestions for a blueprint.
    
    Provides intelligent suggestions for:
    - Structural improvements
    - Content enhancements
    - Relationship optimizations
    - Quality improvements
    """
    try:
        llm_service = LLMService()
        blueprint_service = BlueprintEditingService(llm_service)
        
        response = await blueprint_service.get_blueprint_editing_suggestions(
            blueprint_id=blueprint_id,
            include_structure=include_structure,
            include_content=include_content,
            include_relationships=include_relationships
        )
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=response.message
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting blueprint suggestions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================================
# PRIMITIVE EDITING ENDPOINTS
# ============================================================================

@router.post("/primitive/edit", response_model=PrimitiveEditingResponse)
async def edit_primitive(
    request: PrimitiveEditingRequest,
    current_user: dict = Depends(get_current_user)
) -> PrimitiveEditingResponse:
    """
    Edit a knowledge primitive using AI agentic capabilities.
    
    Provides intelligent editing for:
    - Concept definitions and descriptions
    - Difficulty level adjustments
    - Relationship mappings
    - Complexity scoring
    """
    try:
        llm_service = LLMService()
        blueprint_service = BlueprintEditingService(llm_service)
        
        response = await blueprint_service.edit_primitive_agentically(request)
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=response.message
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error editing primitive: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/primitive/{primitive_id}/suggestions", response_model=PrimitiveEditingSuggestionsResponse)
async def get_primitive_editing_suggestions(
    primitive_id: int,
    include_clarity: bool = True,
    include_complexity: bool = True,
    include_relationships: bool = True,
    current_user: dict = Depends(get_current_user)
) -> PrimitiveEditingSuggestionsResponse:
    """
    Get AI-powered editing suggestions for a knowledge primitive.
    
    Provides suggestions for:
    - Clarity improvements
    - Complexity adjustments
    - Relationship optimizations
    - Content enhancements
    """
    try:
        llm_service = LLMService()
        blueprint_service = BlueprintEditingService(llm_service)
        
        response = await blueprint_service.get_primitive_editing_suggestions(
            primitive_id=primitive_id,
            include_clarity=include_clarity,
            include_complexity=include_complexity,
            include_relationships=include_relationships
        )
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=response.message
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting primitive suggestions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================================
# MASTERY CRITERION EDITING ENDPOINTS
# ============================================================================

@router.post("/criterion/edit", response_model=MasteryCriterionEditingResponse)
async def edit_mastery_criterion(
    request: MasteryCriterionEditingRequest,
    current_user: dict = Depends(get_current_user)
) -> MasteryCriterionEditingResponse:
    """
    Edit a mastery criterion using AI agentic capabilities.
    
    Provides intelligent editing for:
    - Assessment criteria and thresholds
    - Difficulty and complexity adjustments
    - Learning pathway relationships
    - UUE stage progression
    """
    try:
        llm_service = LLMService()
        blueprint_service = BlueprintEditingService(llm_service)
        
        response = await blueprint_service.edit_mastery_criterion_agentically(request)
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=response.message
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error editing mastery criterion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/criterion/{criterion_id}/suggestions", response_model=MasteryCriterionEditingSuggestionsResponse)
async def get_mastery_criterion_editing_suggestions(
    criterion_id: int,
    include_clarity: bool = True,
    include_difficulty: bool = True,
    include_assessment: bool = True,
    current_user: dict = Depends(get_current_user)
) -> MasteryCriterionEditingSuggestionsResponse:
    """
    Get AI-powered editing suggestions for a mastery criterion.
    
    Provides suggestions for:
    - Clarity improvements
    - Difficulty adjustments
    - Assessment optimizations
    - Learning pathway enhancements
    """
    try:
        llm_service = LLMService()
        blueprint_service = BlueprintEditingService(llm_service)
        
        response = await blueprint_service.get_mastery_criterion_editing_suggestions(
            criterion_id=criterion_id,
            include_clarity=include_clarity,
            include_difficulty=include_difficulty,
            include_assessment=include_assessment
        )
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=response.message
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting mastery criterion suggestions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================================
# QUESTION EDITING ENDPOINTS
# ============================================================================

@router.post("/question/edit", response_model=QuestionEditingResponse)
async def edit_question(
    request: QuestionEditingRequest,
    current_user: dict = Depends(get_current_user)
) -> QuestionEditingResponse:
    """
    Edit a question using AI agentic capabilities.
    
    Provides intelligent editing for:
    - Question clarity and difficulty
    - Answer accuracy and explanations
    - Assessment quality
    - Learning objective alignment
    """
    try:
        llm_service = LLMService()
        blueprint_service = BlueprintEditingService(llm_service)
        
        response = await blueprint_service.edit_question_agentically(request)
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=response.message
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error editing question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/question/{question_id}/suggestions", response_model=QuestionEditingSuggestionsResponse)
async def get_question_editing_suggestions(
    question_id: int,
    include_clarity: bool = True,
    include_difficulty: bool = True,
    include_quality: bool = True,
    current_user: dict = Depends(get_current_user)
) -> QuestionEditingSuggestionsResponse:
    """
    Get AI-powered editing suggestions for a question.
    
    Provides suggestions for:
    - Clarity improvements
    - Difficulty adjustments
    - Quality enhancements
    - Assessment optimizations
    """
    try:
        llm_service = LLMService()
        blueprint_service = BlueprintEditingService(llm_service)
        
        response = await blueprint_service.get_question_editing_suggestions(
            question_id=question_id,
            include_clarity=include_clarity,
            include_difficulty=include_difficulty,
            include_quality=include_quality
        )
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=response.message
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting question suggestions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for the blueprint editing service."""
    return {"status": "healthy", "service": "blueprint-editing"}


@router.get("/capabilities")
async def get_capabilities() -> Dict[str, Any]:
    """Get information about the blueprint editing service capabilities."""
    return {
        "service": "blueprint-editing",
        "version": "1.0.0",
        "capabilities": {
            "blueprint_editing": {
                "granular_editing": True,
                "ai_powered_suggestions": True,
                "context_aware_editing": True,
                "structure_preservation": True
            },
            "primitive_editing": {
                "content_editing": True,
                "difficulty_adjustment": True,
                "relationship_mapping": True,
                "complexity_scoring": True
            },
            "mastery_criterion_editing": {
                "assessment_criteria": True,
                "difficulty_adjustment": True,
                "learning_pathways": True,
                "uue_progression": True
            },
            "question_editing": {
                "clarity_improvement": True,
                "difficulty_adjustment": True,
                "quality_enhancement": True,
                "assessment_optimization": True
            }
        },
        "supported_edit_types": [
            "edit_section", "add_section", "remove_section", "reorder_sections",
            "edit_primitive", "add_primitive", "remove_primitive", "reorder_primitives",
            "edit_criterion", "add_criterion", "remove_criterion", "reorder_criteria",
            "edit_question", "add_question", "remove_question", "reorder_questions",
            "improve_clarity", "improve_structure", "add_examples", "simplify_language",
            "enhance_detail", "correct_errors"
        ]
    }





