"""
Sprint 31 API Endpoints for Primitive Services Logic & Integration.

These endpoints integrate the enhanced primitive generation, mastery criteria creation,
question generation, and Core API synchronization services.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Path
from typing import List, Dict, Any, Optional
import logging

from app.api.schemas import (
    PrimitiveGenerationRequest, PrimitiveGenerationResponse,
    MasteryCriterionDto, KnowledgePrimitiveDto,
    CriterionQuestionRequest, CriterionQuestionResponse,
    SyncStatusResponse, BlueprintPrimitivesRequest
)
from app.core.deconstruction import generate_primitives_with_criteria_from_source
from app.core.mastery_criteria_service import mastery_criteria_service
from app.core.question_generation_service import question_generation_service
from app.core.question_mapping_service import question_mapping_service
from app.core.core_api_sync_service import core_api_sync_service
from app.core.primitive_transformation import primitive_transformer
from app.core.blueprint_lifecycle import BlueprintLifecycleService

# Mock auth function for debug mode (auth is disabled)
def get_current_user():
    """Mock user for debug mode - returns default user info."""
    return {"id": "debug-user", "username": "debug"}

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/primitives", tags=["primitives"])


@router.post("/generate", response_model=PrimitiveGenerationResponse)
async def generate_primitives_with_criteria(
    request: PrimitiveGenerationRequest,
    current_user: dict = Depends(get_current_user)
) -> PrimitiveGenerationResponse:
    """
    Generate knowledge primitives with mastery criteria from source content.
    
    This endpoint uses LLM to analyze source content and generate Core API
    compatible primitives with UEE-progressive mastery criteria.
    """
    try:
        logger.info(f"Generating primitives for user {current_user.get('id')} from {request.sourceType}")
        
        # Generate enhanced blueprint with Core API primitives
        enhanced_blueprint = await generate_primitives_with_criteria_from_source(
            source_content=request.sourceContent,
            source_type=request.sourceType,
            user_preferences=request.userPreferences
        )
        
        # Extract Core API primitives from blueprint metadata
        core_api_primitives = getattr(enhanced_blueprint, '_core_api_primitives', [])
        
        if not core_api_primitives:
            # Fallback: transform existing blueprint primitives
            core_api_primitives = primitive_transformer.transform_blueprint_to_primitives(enhanced_blueprint)
        
        # Convert to DTOs for response
        primitive_dtos = []
        for primitive in core_api_primitives:
            criteria_dtos = [
                MasteryCriterionDto(
                    criterionId=criterion.criterionId,
                    title=criterion.title,
                    description=criterion.description,
                    ueeLevel=criterion.ueeLevel,
                    weight=criterion.weight,
                    isRequired=criterion.isRequired
                )
                for criterion in primitive.masteryCriteria
            ]
            
            primitive_dto = KnowledgePrimitiveDto(
                primitiveId=primitive.primitiveId,
                title=primitive.title,
                description=primitive.description,
                primitiveType=primitive.primitiveType,
                difficultyLevel=primitive.difficultyLevel,
                estimatedTimeMinutes=primitive.estimatedTimeMinutes,
                trackingIntensity=primitive.trackingIntensity,
                masteryCriteria=criteria_dtos
            )
            primitive_dtos.append(primitive_dto)
        
        response = PrimitiveGenerationResponse(
            success=True,
            primitives=primitive_dtos,
            generatedCount=len(primitive_dtos),
            totalCriteria=sum(len(p.masteryCriteria) for p in primitive_dtos),
            processingTime=0.0,  # Could add actual timing
            metadata={
                'sourceType': request.sourceType,
                'userPreferences': request.userPreferences,
                'ueeDistribution': _calculate_uee_distribution(core_api_primitives)
            }
        )
        
        logger.info(f"Generated {len(primitive_dtos)} primitives with {response.totalCriteria} criteria")
        return response
        
    except Exception as e:
        logger.error(f"Failed to generate primitives: {e}")
        raise HTTPException(status_code=500, detail=f"Primitive generation failed: {str(e)}")


@router.post("/questions/generate", response_model=CriterionQuestionResponse)
async def generate_criterion_questions(
    request: CriterionQuestionRequest,
    current_user: dict = Depends(get_current_user)
) -> CriterionQuestionResponse:
    """
    Generate questions mapped to specific mastery criteria.
    
    Creates UEE-level appropriate questions that assess specific mastery criteria
    for use in the spaced repetition system.
    """
    try:
        logger.info(f"Generating questions for {len(request.criterionIds)} criteria")
        
        # Get primitives and criteria (would typically fetch from database)
        # For now, assume they're provided in request or fetch from Core API
        questions_by_criterion = {}
        total_questions = 0
        
        for criterion_data in request.criteria:
            # Create criterion object
            criterion = primitive_transformer._create_mastery_criterion_from_dict(criterion_data)
            primitive = primitive_transformer._create_primitive_from_dict(request.primitive)
            
            # Generate questions for this criterion
            questions = await question_generation_service.generate_questions_for_criterion(
                criterion=criterion,
                primitive=primitive,
                source_content=request.sourceContent,
                question_count=request.questionsPerCriterion,
                user_preferences=request.userPreferences
            )
            
            questions_by_criterion[criterion.criterionId] = questions
            total_questions += len(questions)
        
        # Map questions to criteria using semantic analysis
        if request.useSemanticMapping:
            all_questions = [q for questions in questions_by_criterion.values() for q in questions]
            all_criteria = [
                primitive_transformer._create_mastery_criterion_from_dict(c) 
                for c in request.criteria
            ]
            
            mappings = await question_mapping_service.map_questions_to_criteria(
                questions=all_questions,
                criteria=all_criteria,
                primitive=primitive,
                source_content=request.sourceContent
            )
            
            # Update questions_by_criterion with optimized mappings
            # (Implementation would reorganize based on semantic mappings)
        
        response = CriterionQuestionResponse(
            success=True,
            questionsByCriterion=questions_by_criterion,
            totalQuestions=total_questions,
            mappingConfidence=0.85,  # Would calculate actual confidence
            metadata={
                'semanticMappingUsed': request.useSemanticMapping,
                'questionsPerCriterion': request.questionsPerCriterion,
                'userPreferences': request.userPreferences
            }
        )
        
        logger.info(f"Generated {total_questions} questions across {len(request.criterionIds)} criteria")
        return response
        
    except Exception as e:
        logger.error(f"Failed to generate criterion questions: {e}")
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")


@router.post("/sync", response_model=SyncStatusResponse)
async def sync_primitives_to_core_api(
    request: BlueprintPrimitivesRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
) -> SyncStatusResponse:
    """
    Synchronize AI-generated primitives and criteria with Core API database.
    
    Creates primitives and mastery criteria in the Core API Prisma database
    for use in the spaced repetition system.
    """
    try:
        user_id = current_user.get('id')
        logger.info(f"Starting sync for blueprint {request.blueprintId}, user {user_id}")
        
        # Convert request primitives to internal format
        primitives = []
        for primitive_data in request.primitives:
            primitive = primitive_transformer._create_primitive_from_dict(primitive_data)
            primitives.append(primitive)
        
        # Start synchronization (run in background for large datasets)
        if len(primitives) > 5:
            background_tasks.add_task(
                _sync_primitives_background,
                primitives,
                request.blueprintId,
                user_id
            )
            
            return SyncStatusResponse(
                success=True,
                status="in_progress",
                message="Sync started in background",
                primitivesProcessed=0,
                criteriaProcessed=0,
                estimatedCompletion=len(primitives) * 2  # Rough estimate in seconds
            )
        else:
            # Sync immediately for small datasets
            sync_result = await core_api_sync_service.sync_primitives_and_criteria(
                primitives=primitives,
                blueprint_id=request.blueprintId,
                user_id=user_id
            )
            
            return SyncStatusResponse(
                success=sync_result['success'],
                status="completed" if sync_result['success'] else "failed",
                message="Sync completed" if sync_result['success'] else "Sync failed",
                primitivesProcessed=sync_result['primitives_created'],
                criteriaProcessed=sync_result['criteria_created'],
                errors=sync_result['errors']
            )
        
    except Exception as e:
        logger.error(f"Failed to sync primitives: {e}")
        raise HTTPException(status_code=500, detail=f"Sync operation failed: {str(e)}")


@router.get("/sync/status/{blueprint_id}", response_model=SyncStatusResponse)
async def get_sync_status(
    blueprint_id: str,
    current_user: dict = Depends(get_current_user)
) -> SyncStatusResponse:
    """Get the current synchronization status for a blueprint."""
    try:
        user_id = current_user.get('id')
        status_result = await core_api_sync_service.get_sync_status(blueprint_id, user_id)
        
        if status_result['success']:
            return SyncStatusResponse(
                success=True,
                status=status_result['status'],
                message=f"Blueprint has {status_result['primitive_count']} primitives",
                primitivesProcessed=status_result['primitive_count'],
                criteriaProcessed=status_result['criteria_count'],
                lastSync=status_result['last_sync'],
                errors=status_result['errors']
            )
        else:
            return SyncStatusResponse(
                success=False,
                status="error",
                message=status_result['error'],
                primitivesProcessed=0,
                criteriaProcessed=0
            )
        
    except Exception as e:
        logger.error(f"Failed to get sync status for {blueprint_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.post("/blueprints/{blueprint_id}/extract", response_model=PrimitiveGenerationResponse)
async def extract_primitives_from_blueprint(
    blueprint_id: str,
    background_tasks: BackgroundTasks,
    user_preferences: Optional[Dict[str, Any]] = None,
    current_user: dict = Depends(get_current_user)
) -> PrimitiveGenerationResponse:
    """
    Extract and enhance primitives from an existing indexed blueprint.
    
    Takes an already-indexed blueprint and generates Core API compatible
    primitives with enhanced mastery criteria.
    """
    try:
        user_id = current_user.get('id')
        logger.info(f"Extracting primitives from blueprint {blueprint_id}")
        
        # Get blueprint from vector store
        # TODO: Fix blueprint retrieval - blueprint_manager doesn't exist
        # blueprint_service = BlueprintLifecycleService()
        # blueprint = await blueprint_service.get_blueprint_status(blueprint_id)
        # For now, skip this functionality to allow server to start
        raise HTTPException(status_code=501, detail="Blueprint extraction not implemented - blueprint_manager missing")
        
        # Reconstruct source content from blueprint sections
        source_content = _reconstruct_source_content(blueprint)
        
        # Generate enhanced primitives
        enhanced_blueprint = await generate_primitives_with_criteria_from_source(
            source_content=source_content,
            source_type=blueprint.source_type,
            user_preferences=user_preferences or {}
        )
        
        # Extract and convert primitives
        core_api_primitives = getattr(enhanced_blueprint, '_core_api_primitives', [])
        
        # Start background sync if requested
        if len(core_api_primitives) > 0:
            background_tasks.add_task(
                _sync_primitives_background,
                core_api_primitives,
                blueprint_id,
                user_id
            )
        
        # Convert to DTOs for response
        primitive_dtos = [
            _convert_primitive_to_dto(primitive) 
            for primitive in core_api_primitives
        ]
        
        response = PrimitiveGenerationResponse(
            success=True,
            primitives=primitive_dtos,
            generatedCount=len(primitive_dtos),
            totalCriteria=sum(len(p.masteryCriteria) for p in primitive_dtos),
            metadata={
                'blueprintId': blueprint_id,
                'sourceType': blueprint.source_type,
                'extractedFromExistingBlueprint': True
            }
        )
        
        logger.info(f"Extracted {len(primitive_dtos)} primitives from blueprint {blueprint_id}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to extract primitives from blueprint {blueprint_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Primitive extraction failed: {str(e)}")


# Helper functions

async def _sync_primitives_background(
    primitives: List,
    blueprint_id: str,
    user_id: str
) -> None:
    """Background task for syncing primitives to Core API."""
    try:
        logger.info(f"Starting background sync for {len(primitives)} primitives")
        
        sync_result = await core_api_sync_service.sync_primitives_and_criteria(
            primitives=primitives,
            blueprint_id=blueprint_id,
            user_id=user_id
        )
        
        if sync_result['success']:
            logger.info(f"Background sync completed successfully: {sync_result['primitives_created']} primitives")
        else:
            logger.error(f"Background sync failed: {sync_result['errors']}")
            
    except Exception as e:
        logger.error(f"Background sync task failed: {e}")


def _calculate_uee_distribution(primitives: List) -> Dict[str, float]:
    """Calculate UEE level distribution across primitives."""
    total_criteria = sum(len(p.masteryCriteria) for p in primitives)
    if total_criteria == 0:
        return {'UNDERSTAND': 0.0, 'USE': 0.0, 'EXPLORE': 0.0}
    
    understand_count = sum(
        1 for p in primitives for c in p.masteryCriteria if c.ueeLevel == 'UNDERSTAND'
    )
    use_count = sum(
        1 for p in primitives for c in p.masteryCriteria if c.ueeLevel == 'USE'
    )
    explore_count = sum(
        1 for p in primitives for c in p.masteryCriteria if c.ueeLevel == 'EXPLORE'
    )
    
    return {
        'UNDERSTAND': understand_count / total_criteria,
        'USE': use_count / total_criteria,
        'EXPLORE': explore_count / total_criteria
    }


def _reconstruct_source_content(blueprint) -> str:
    """Reconstruct source content from blueprint sections."""
    content_parts = []
    
    if hasattr(blueprint, 'source_summary') and blueprint.source_summary:
        content_parts.append(blueprint.source_summary)
    
    if hasattr(blueprint, 'sections'):
        for section in blueprint.sections:
            if hasattr(section, 'content'):
                content_parts.append(section.content)
    
    return '\n\n'.join(content_parts)


def _convert_primitive_to_dto(primitive) -> KnowledgePrimitiveDto:
    """Convert internal primitive to DTO."""
    criteria_dtos = [
        MasteryCriterionDto(
            criterionId=criterion.criterionId,
            title=criterion.title,
            description=criterion.description,
            ueeLevel=criterion.ueeLevel,
            weight=criterion.weight,
            isRequired=criterion.isRequired
        )
        for criterion in primitive.masteryCriteria
    ]
    
    return KnowledgePrimitiveDto(
        primitiveId=primitive.primitiveId,
        title=primitive.title,
        description=primitive.description,
        primitiveType=primitive.primitiveType,
        difficultyLevel=primitive.difficultyLevel,
        estimatedTimeMinutes=primitive.estimatedTimeMinutes,
        trackingIntensity=primitive.trackingIntensity,
        masteryCriteria=criteria_dtos
    )

# Section-Aware Primitive Endpoints

@router.post("/sections/{section_id}/generate", response_model=PrimitiveGenerationResponse)
async def generate_section_primitives_endpoint(
    section_id: str = Path(..., description="ID of the section to generate primitives for"),
    request: PrimitiveGenerationRequest = ...,
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> PrimitiveGenerationResponse:
    """
    Generate primitives and mastery criteria for a specific section.
    
    This endpoint extends the primitive generation functionality to be section-aware,
    allowing generation of primitives for specific sections rather than entire blueprints.
    """
    try:
        from app.services.blueprint_section_service import BlueprintSectionService
        
        # Get section service
        section_service = BlueprintSectionService()
        
        # Validate section exists
        section = await section_service.get_section(section_id)
        if not section:
            raise HTTPException(
                status_code=404,
                detail=f"Section {section_id} not found"
            )
        
        # Generate primitives for the section
        primitives_result = await generate_primitives_with_criteria_from_source(
            source_content=request.source_content,
            source_type=request.source_type,
            user_id=request.user_id,
            section_id=section_id,
            blueprint_id=str(section.blueprint_id)
        )
        
        # Add background task for section content update
        background_tasks.add_task(
            section_service.update_section_content,
            section_id=section_id,
            content=request.source_content,
            primitives=primitives_result.primitives
        )
        
        return PrimitiveGenerationResponse(
            primitives=primitives_result.primitives,
            mastery_criteria=primitives_result.mastery_criteria,
            total_primitives=len(primitives_result.primitives),
            total_criteria=len(primitives_result.mastery_criteria),
            section_id=section_id,
            blueprint_id=str(section.blueprint_id),
            generation_timestamp=primitives_result.generation_timestamp
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate section primitives: {str(e)}"
        )

@router.get("/sections/{section_id}/primitives", response_model=List[KnowledgePrimitiveDto])
async def get_section_primitives_endpoint(
    section_id: str = Path(..., description="ID of the section to get primitives from")
) -> List[KnowledgePrimitiveDto]:
    """
    Get all primitives associated with a specific section.
    
    This endpoint retrieves all knowledge primitives that have been generated
    or mapped to a particular section.
    """
    try:
        from app.services.blueprint_section_service import BlueprintSectionService
        
        # Get section service
        section_service = BlueprintSectionService()
        
        # Validate section exists
        section = await section_service.get_section(section_id)
        if not section:
            raise HTTPException(
                status_code=404,
                detail=f"Section {section_id} not found"
            )
        
        # Get section primitives
        section_primitives = await section_service.get_section_primitives(section_id)
        
        return section_primitives
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get section primitives: {str(e)}"
        )

@router.post("/sections/{section_id}/primitives/sync", response_model=SyncStatusResponse)
async def sync_section_primitives_endpoint(
    section_id: str = Path(..., description="ID of the section to sync primitives for")
) -> SyncStatusResponse:
    """
    Synchronize primitives from a specific section with the Core API.
    
    This endpoint synchronizes all primitives and mastery criteria from a specific
    section with the Core API system.
    """
    try:
        from app.services.blueprint_section_service import BlueprintSectionService
        from app.core.core_api_sync_service import core_api_sync_service
        
        # Get section service
        section_service = BlueprintSectionService()
        
        # Validate section exists
        section = await section_service.get_section(section_id)
        if not section:
            raise HTTPException(
                status_code=404,
                detail=f"Section {section_id} not found"
            )
        
        # Get section primitives
        section_primitives = await section_service.get_section_primitives(section_id)
        
        # Sync with Core API
        sync_result = await core_api_sync_service.sync_section_primitives(
            section_primitives, section_id, str(section.blueprint_id)
        )
        
        return SyncStatusResponse(
            blueprint_id=str(section.blueprint_id),
            section_id=section_id,
            sync_success=sync_result["success"],
            primitives_synced=sync_result["primitives_synced"],
            criteria_synced=sync_result["criteria_synced"],
            errors=sync_result.get("errors", []),
            sync_timestamp=sync_result["sync_timestamp"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to sync section primitives: {str(e)}"
        )
