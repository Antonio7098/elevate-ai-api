"""
Blueprint service for managing blueprint lifecycle operations.

This module provides the core service layer for blueprint operations.
"""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from app.models.blueprint import Blueprint, BlueprintCreateRequest, BlueprintUpdateRequest, BlueprintStatus
from app.core.blueprint.blueprint_repository import BlueprintRepository
from app.core.blueprint.blueprint_validator import BlueprintValidator


class BlueprintService:
    def __init__(self, repository: BlueprintRepository, validator: BlueprintValidator):
        self.repository = repository
        self.validator = validator
    
    async def create_blueprint(self, request: BlueprintCreateRequest, author_id: str) -> Blueprint:
        """Create a new blueprint"""
        # Validate the request
        await self.validator.validate_create_request(request)
        
        # Create blueprint instance
        blueprint = Blueprint(
            id=str(uuid.uuid4()),
            title=request.title,
            description=request.description,
            content=request.content,
            type=request.type,
            tags=request.tags,
            metadata=request.metadata,
            is_public=request.is_public,
            author_id=author_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Save to repository
        saved_blueprint = await self.repository.create(blueprint)
        return saved_blueprint
    
    async def get_blueprint(self, blueprint_id: str) -> Optional[Blueprint]:
        """Get a blueprint by ID"""
        return await self.repository.get_by_id(blueprint_id)
    
    async def get_blueprints_by_author(self, author_id: str, limit: int = 100, offset: int = 0) -> List[Blueprint]:
        """Get blueprints by author"""
        return await self.repository.get_by_author(author_id, limit, offset)
    
    async def get_public_blueprints(self, limit: int = 100, offset: int = 0) -> List[Blueprint]:
        """Get public blueprints"""
        return await self.repository.get_public(limit, offset)
    
    async def update_blueprint(self, blueprint_id: str, request: BlueprintUpdateRequest, user_id: str) -> Optional[Blueprint]:
        """Update a blueprint"""
        # Get existing blueprint
        blueprint = await self.repository.get_by_id(blueprint_id)
        if not blueprint:
            return None
        
        # Check authorization
        if blueprint.author_id != user_id:
            raise PermissionError("User not authorized to update this blueprint")
        
        # Validate the update request
        await self.validator.validate_update_request(request)
        
        # Update fields
        update_data = request.dict(exclude_unset=True)
        update_data['updated_at'] = datetime.utcnow()
        
        # Update blueprint
        updated_blueprint = await self.repository.update(blueprint_id, update_data)
        return updated_blueprint
    
    async def delete_blueprint(self, blueprint_id: str, user_id: str) -> bool:
        """Delete a blueprint (soft delete)"""
        # Get existing blueprint
        blueprint = await self.repository.get_by_id(blueprint_id)
        if not blueprint:
            return False
        
        # Check authorization
        if blueprint.author_id != user_id:
            raise PermissionError("User not authorized to delete this blueprint")
        
        # Soft delete by updating status
        await self.repository.update(blueprint_id, {
            'status': BlueprintStatus.DELETED,
            'updated_at': datetime.utcnow()
        })
        return True
    
    async def archive_blueprint(self, blueprint_id: str, user_id: str) -> bool:
        """Archive a blueprint"""
        # Get existing blueprint
        blueprint = await self.repository.get_by_id(blueprint_id)
        if not blueprint:
            return False
        
        # Check authorization
        if blueprint.author_id != user_id:
            raise PermissionError("User not authorized to archive this blueprint")
        
        # Update status to archived
        await self.repository.update(blueprint_id, {
            'status': BlueprintStatus.ARCHIVED,
            'updated_at': datetime.utcnow()
        })
        return True
    
    async def activate_blueprint(self, blueprint_id: str, user_id: str) -> bool:
        """Activate a blueprint"""
        # Get existing blueprint
        blueprint = await self.repository.get_by_id(blueprint_id)
        if not blueprint:
            return False
        
        # Check authorization
        if blueprint.author_id != user_id:
            raise PermissionError("User not authorized to activate this blueprint")
        
        # Update status to active
        await self.repository.update(blueprint_id, {
            'status': BlueprintStatus.ACTIVE,
            'updated_at': datetime.utcnow()
        })
        return True
    
    async def search_blueprints(self, query: str, limit: int = 100, offset: int = 0) -> List[Blueprint]:
        """Search blueprints by text query"""
        return await self.repository.search(query, limit, offset)
    
    async def get_blueprints_by_type(self, blueprint_type: str, limit: int = 100, offset: int = 0) -> List[Blueprint]:
        """Get blueprints by type"""
        return await self.repository.get_by_type(blueprint_type, limit, offset)
    
    async def get_blueprints_by_tags(self, tags: List[str], limit: int = 100, offset: int = 0) -> List[Blueprint]:
        """Get blueprints by tags"""
        return await self.repository.get_by_tags(tags, limit, offset)
