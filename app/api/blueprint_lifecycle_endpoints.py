"""
Blueprint Lifecycle API Endpoints - Extends core API with blueprint update management.

These endpoints integrate with the existing FastAPI structure to provide
blueprint update, change detection, and lifecycle management capabilities.
"""

from fastapi import APIRouter, HTTPException, status, Path, Query
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from app.api.schemas import (
    ErrorResponse,
    IndexBlueprintRequest,  # Reuse existing schemas
    IndexBlueprintResponse
)
from app.core.blueprint_lifecycle import (
    BlueprintLifecycleService,
    update_blueprint,
    delete_blueprint,
    get_blueprint_info,
    ChangeType
)
from app.models.learning_blueprint import LearningBlueprint
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Create router for blueprint lifecycle endpoints
lifecycle_router = APIRouter()


# New Pydantic schemas for lifecycle endpoints
class UpdateBlueprintRequest(BaseModel):
    """Request to update an existing blueprint."""
    blueprint: Dict[str, Any]  # LearningBlueprint as dict
    strategy: Optional[str] = "incremental"  # or "full_reindex"
    
    class Config:
        schema_extra = {
            "example": {
                "blueprint": {
                    "source_id": "python-basics-v2",
                    "source_title": "Python Basics Updated",
                    "source_type": "tutorial",
                    "sections": [{"section_name": "Variables", "description": "..."}]
                },
                "strategy": "incremental"
            }
        }


class UpdateBlueprintResponse(BaseModel):
    """Response from blueprint update operation."""
    blueprint_id: str
    status: str
    strategy: str
    changes_applied: int
    changes_failed: int
    changeset_summary: Dict[str, int]
    operations: list
    timestamp: str
    
    class Config:
        schema_extra = {
            "example": {
                "blueprint_id": "python-basics-v2",
                "status": "success",
                "strategy": "incremental", 
                "changes_applied": 3,
                "changes_failed": 0,
                "changeset_summary": {"added": 1, "modified": 2, "deleted": 0},
                "operations": ["Added section_5", "Updated section_2", "Updated section_3"],
                "timestamp": "2024-01-20T10:30:00Z"
            }
        }


class BlueprintStatusResponse(BaseModel):
    """Response for blueprint status check."""
    blueprint_id: str
    status: str
    is_indexed: bool
    node_count: int
    last_updated: Optional[str]
    locus_types: Dict[str, int]
    
    class Config:
        schema_extra = {
            "example": {
                "blueprint_id": "python-basics-v2",
                "status": "indexed",
                "is_indexed": True,
                "node_count": 28,
                "last_updated": "2024-01-20T10:30:00Z",
                "locus_types": {"foundational_concept": 15, "use_case": 8, "key_term": 5}
            }
        }


class ChangePreviewResponse(BaseModel):
    """Response for change preview (dry run)."""
    blueprint_id: str
    has_changes: bool
    total_changes: int
    summary: Dict[str, int]
    changes: list
    timestamp: str
    
    class Config:
        schema_extra = {
            "example": {
                "blueprint_id": "python-basics-v2",
                "has_changes": True,
                "total_changes": 3,
                "summary": {"added": 1, "modified": 2, "deleted": 0},
                "changes": [
                    {"locus_id": "section_5", "change_type": "added", "title": "New Section"},
                    {"locus_id": "section_2", "change_type": "modified", "title": "Updated Content"}
                ],
                "timestamp": "2024-01-20T10:30:00Z"
            }
        }


@lifecycle_router.put("/blueprints/{blueprint_id}", response_model=UpdateBlueprintResponse)
async def update_blueprint_endpoint(
    blueprint_id: str = Path(..., description="ID of the blueprint to update"),
    request: UpdateBlueprintRequest = None
):
    """
    Update an existing blueprint in the vector database.
    
    This endpoint:
    1. Detects changes between current and new blueprint versions
    2. Applies changes using incremental or full reindex strategy
    3. Returns detailed results of the update operation
    
    Strategy options:
    - 'incremental': Only updates changed loci (faster, recommended)
    - 'full_reindex': Deletes and re-indexes entire blueprint (slower, guaranteed consistency)
    """
    try:
        logger.info(f"Updating blueprint {blueprint_id} with strategy {request.strategy}")
        
        # Convert dict back to LearningBlueprint object
        blueprint_dict = request.blueprint
        blueprint = LearningBlueprint(**blueprint_dict)
        
        # Validate blueprint_id matches
        if blueprint.source_id != blueprint_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Blueprint ID mismatch: URL has '{blueprint_id}' but blueprint has '{blueprint.source_id}'"
            )
        
        # Perform update using lifecycle service
        result = await update_blueprint(
            blueprint_id=blueprint_id,
            new_blueprint=blueprint,
            strategy=request.strategy or "incremental"
        )
        
        # Handle different result types
        if result.get("status") == "no_changes":
            return UpdateBlueprintResponse(
                blueprint_id=blueprint_id,
                status="no_changes",
                strategy=request.strategy,
                changes_applied=0,
                changes_failed=0,
                changeset_summary={"added": 0, "modified": 0, "deleted": 0},
                operations=["No changes detected"],
                timestamp=datetime.utcnow().isoformat()
            )
        
        return UpdateBlueprintResponse(
            blueprint_id=blueprint_id,
            status="success" if result.get("changes_failed", 0) == 0 else "partial_failure",
            strategy=result.get("strategy", request.strategy),
            changes_applied=result.get("changes_applied", 0),
            changes_failed=result.get("changes_failed", 0),
            changeset_summary=result.get("changeset_summary", {}),
            operations=result.get("operations", []),
            timestamp=result.get("timestamp", datetime.utcnow().isoformat())
        )
        
    except Exception as e:
        logger.error(f"Failed to update blueprint {blueprint_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Blueprint update failed: {str(e)}"
        )


@lifecycle_router.get("/blueprints/{blueprint_id}/status", response_model=BlueprintStatusResponse)
async def get_blueprint_status_endpoint(
    blueprint_id: str = Path(..., description="ID of the blueprint to check")
):
    """
    Get current status and information about a blueprint in the vector database.
    
    Returns:
    - Indexing status (indexed/not_indexed/error)
    - Number of nodes indexed
    - Distribution by locus types
    - Last update timestamp (if available)
    """
    try:
        logger.info(f"Getting status for blueprint {blueprint_id}")
        
        status_info = await get_blueprint_info(blueprint_id)
        
        return BlueprintStatusResponse(
            blueprint_id=blueprint_id,
            status=status_info.get("status", "unknown"),
            is_indexed=status_info.get("is_indexed", False),
            node_count=status_info.get("node_count", 0),
            last_updated=status_info.get("last_updated"),
            locus_types=status_info.get("locus_types", {})
        )
        
    except Exception as e:
        logger.error(f"Failed to get blueprint status {blueprint_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Status check failed: {str(e)}"
        )


@lifecycle_router.post("/blueprints/{blueprint_id}/changes", response_model=ChangePreviewResponse)
async def preview_blueprint_changes_endpoint(
    blueprint_id: str = Path(..., description="ID of the blueprint to preview changes for"),
    request: UpdateBlueprintRequest = None
):
    """
    Preview changes that would be made when updating a blueprint (dry run).
    
    This is a read-only operation that:
    1. Detects changes between current and new blueprint versions
    2. Returns detailed change information WITHOUT applying changes
    3. Useful for reviewing updates before applying them
    """
    try:
        logger.info(f"Previewing changes for blueprint {blueprint_id}")
        
        # Convert dict back to LearningBlueprint object
        blueprint_dict = request.blueprint
        blueprint = LearningBlueprint(**blueprint_dict)
        
        # Validate blueprint_id matches
        if blueprint.source_id != blueprint_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Blueprint ID mismatch: URL has '{blueprint_id}' but blueprint has '{blueprint.source_id}'"
            )
        
        # Detect changes using lifecycle service (dry run)
        service = BlueprintLifecycleService()
        changeset = await service.detect_blueprint_changes(blueprint_id, blueprint)
        
        # Format changes for API response
        changes_list = []
        for change in changeset.changes:
            change_info = {
                "locus_id": change.locus_id,
                "change_type": change.change_type.value,
                "content_hash_changed": change.content_hash_changed
            }
            
            if change.new_node:
                change_info["title"] = change.new_node.locus_title or "Untitled"
                change_info["content_preview"] = change.new_node.content[:100] + "..." if len(change.new_node.content) > 100 else change.new_node.content
            
            if change.old_node:
                change_info["old_title"] = change.old_node.locus_title or "Untitled"
            
            changes_list.append(change_info)
        
        return ChangePreviewResponse(
            blueprint_id=blueprint_id,
            has_changes=changeset.has_changes,
            total_changes=changeset.total_changes,
            summary=changeset.summary,
            changes=changes_list,
            timestamp=changeset.timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to preview changes for blueprint {blueprint_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Change preview failed: {str(e)}"
        )


@lifecycle_router.delete("/blueprints/{blueprint_id}")
async def delete_blueprint_endpoint(
    blueprint_id: str = Path(..., description="ID of the blueprint to delete")
):
    """
    Delete a blueprint and all its indexed content from the vector database.
    
    This operation:
    1. Removes all TextNodes and vectors associated with the blueprint
    2. Cleans up any related metadata and relationships
    3. Cannot be undone - use with caution
    """
    try:
        logger.info(f"Deleting blueprint {blueprint_id}")
        
        result = await delete_blueprint(blueprint_id)
        
        return {
            "blueprint_id": blueprint_id,
            "status": "deleted",
            "nodes_deleted": result.get("nodes_deleted", 0),
            "deletion_completed": result.get("deletion_completed", True),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to delete blueprint {blueprint_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Blueprint deletion failed: {str(e)}"
        )


# Health check endpoint for lifecycle service
@lifecycle_router.get("/blueprints/health")
async def lifecycle_health_check():
    """
    Health check endpoint for blueprint lifecycle service.
    
    Returns service status and basic connectivity information.
    """
    try:
        # Basic connectivity test
        service = BlueprintLifecycleService()
        await service._initialize()
        
        return {
            "status": "healthy",
            "service": "BlueprintLifecycleService",
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Lifecycle service is operational"
        }
        
    except Exception as e:
        logger.error(f"Lifecycle health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "BlueprintLifecycleService", 
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }
