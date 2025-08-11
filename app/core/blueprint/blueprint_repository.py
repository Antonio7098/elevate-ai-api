"""
Blueprint repository for data access operations.

This module provides the data access layer for blueprint operations.
"""

from typing import List, Optional, Dict, Any
from app.models.blueprint import Blueprint, BlueprintStatus


class BlueprintRepository:
    """Repository class for blueprint data operations."""
    
    def __init__(self):
        # In a real implementation, this would be a database connection
        # For now, we'll use an in-memory storage
        self._blueprints: Dict[str, Blueprint] = {}
    
    async def create(self, blueprint: Blueprint) -> Blueprint:
        """Create a new blueprint."""
        self._blueprints[blueprint.id] = blueprint
        return blueprint
    
    async def get_by_id(self, blueprint_id: str) -> Optional[Blueprint]:
        """Get a blueprint by ID."""
        return self._blueprints.get(blueprint_id)
    
    async def get_by_author(self, author_id: str, limit: int = 100, offset: int = 0) -> List[Blueprint]:
        """Get blueprints by author."""
        blueprints = [
            bp for bp in self._blueprints.values()
            if bp.author_id == author_id and bp.status != BlueprintStatus.DELETED
        ]
        return blueprints[offset:offset + limit]
    
    async def get_public(self, limit: int = 100, offset: int = 0) -> List[Blueprint]:
        """Get public blueprints."""
        blueprints = [
            bp for bp in self._blueprints.values()
            if bp.is_public and bp.status == BlueprintStatus.ACTIVE
        ]
        return blueprints[offset:offset + limit]
    
    async def update(self, blueprint_id: str, update_data: Dict[str, Any]) -> Optional[Blueprint]:
        """Update a blueprint."""
        if blueprint_id not in self._blueprints:
            return None
        
        blueprint = self._blueprints[blueprint_id]
        
        # Update fields
        for key, value in update_data.items():
            if hasattr(blueprint, key):
                setattr(blueprint, key, value)
        
        self._blueprints[blueprint_id] = blueprint
        return blueprint
    
    async def delete(self, blueprint_id: str) -> bool:
        """Delete a blueprint."""
        if blueprint_id in self._blueprints:
            del self._blueprints[blueprint_id]
            return True
        return False
    
    async def search(self, query: str, limit: int = 100, offset: int = 0) -> List[Blueprint]:
        """Search blueprints by text query."""
        query_lower = query.lower()
        results = []
        
        for blueprint in self._blueprints.values():
            if blueprint.status == BlueprintStatus.DELETED:
                continue
                
            # Search in title, description, and tags
            if (query_lower in blueprint.title.lower() or
                (blueprint.description and query_lower in blueprint.description.lower()) or
                any(query_lower in tag.lower() for tag in blueprint.tags)):
                results.append(blueprint)
        
        return results[offset:offset + limit]
    
    async def get_by_type(self, blueprint_type: str, limit: int = 100, offset: int = 0) -> List[Blueprint]:
        """Get blueprints by type."""
        blueprints = [
            bp for bp in self._blueprints.values()
            if bp.type.value == blueprint_type and bp.status != BlueprintStatus.DELETED
        ]
        return blueprints[offset:offset + limit]
    
    async def get_by_tags(self, tags: List[str], limit: int = 100, offset: int = 0) -> List[Blueprint]:
        """Get blueprints by tags."""
        tag_set = set(tags)
        blueprints = [
            bp for bp in self._blueprints.values()
            if bp.status != BlueprintStatus.DELETED and
            any(tag in bp.tags for tag in tag_set)
        ]
        return blueprints[offset:offset + limit]
    
    async def list_all(self, limit: int = 100, offset: int = 0) -> List[Blueprint]:
        """List all non-deleted blueprints."""
        blueprints = [
            bp for bp in self._blueprints.values()
            if bp.status != BlueprintStatus.DELETED
        ]
        return blueprints[offset:offset + limit]
    
    async def count_by_status(self, status: BlueprintStatus) -> int:
        """Count blueprints by status."""
        return sum(1 for bp in self._blueprints.values() if bp.status == status)
    
    async def count_by_author(self, author_id: str) -> int:
        """Count blueprints by author."""
        return sum(1 for bp in self._blueprints.values() 
                  if bp.author_id == author_id and bp.status != BlueprintStatus.DELETED)
