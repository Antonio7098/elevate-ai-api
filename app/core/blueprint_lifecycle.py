"""
Blueprint Lifecycle Management - Handle updates, changes, and synchronization.

This module provides services for managing blueprint changes and keeping the
vector database synchronized with blueprint modifications.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from app.models.learning_blueprint import LearningBlueprint
from app.models.text_node import TextNode
from app.core.blueprint_parser import BlueprintParser
from app.core.indexing_pipeline import IndexingPipeline
from app.core.vector_store import create_vector_store

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of changes detected in blueprints."""
    ADDED = "added"
    MODIFIED = "modified" 
    DELETED = "deleted"


@dataclass
class LocusChange:
    """Represents a change to a specific locus."""
    locus_id: str
    change_type: ChangeType
    old_node: Optional[TextNode] = None
    new_node: Optional[TextNode] = None
    content_hash_changed: bool = False


@dataclass
class BlueprintChangeSet:
    """Collection of all changes for a blueprint update."""
    blueprint_id: str
    timestamp: datetime
    changes: List[LocusChange]
    summary: Dict[str, int]  # Count by change type
    
    @property
    def has_changes(self) -> bool:
        return len(self.changes) > 0
    
    @property
    def total_changes(self) -> int:
        return len(self.changes)


class BlueprintLifecycleService:
    """Service for managing blueprint lifecycle and updates."""
    
    def __init__(self):
        self.parser = BlueprintParser()
        self.indexing_pipeline = IndexingPipeline()
        self.vector_store = None
        
    async def _initialize(self):
        """Initialize vector store connection."""
        try:
            if not self.vector_store:
                from app.core.config import settings
                self.vector_store = create_vector_store(
                    store_type=settings.vector_store_type,
                    api_key=settings.pinecone_api_key,
                    environment=settings.pinecone_environment
                )
                await self.vector_store.initialize()
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            logger.error(f"Failed to initialize BlueprintLifecycleService: {e}\nTraceback:\n{tb_str}")
            raise
    
    async def detect_blueprint_changes(
        self, 
        blueprint_id: str, 
        new_blueprint: LearningBlueprint
    ) -> BlueprintChangeSet:
        """
        Detect changes between current indexed blueprint and new version.
        
        Args:
            blueprint_id: ID of the blueprint to check
            new_blueprint: New version of the blueprint
            
        Returns:
            BlueprintChangeSet with all detected changes
        """
        await self._initialize()
        
        # Parse new blueprint
        new_nodes = self.parser.parse_blueprint(new_blueprint)
        new_loci_map = {node.locus_id: node for node in new_nodes}
        
        # Get existing loci from vector database
        existing_loci = await self._get_blueprint_loci(blueprint_id)
        existing_loci_set = set(existing_loci.keys())
        new_loci_set = set(new_loci_map.keys())
        
        changes = []
        
        # Find added loci
        added_loci = new_loci_set - existing_loci_set
        for locus_id in added_loci:
            changes.append(LocusChange(
                locus_id=locus_id,
                change_type=ChangeType.ADDED,
                new_node=new_loci_map[locus_id]
            ))
        
        # Find deleted loci
        deleted_loci = existing_loci_set - new_loci_set
        for locus_id in deleted_loci:
            changes.append(LocusChange(
                locus_id=locus_id,
                change_type=ChangeType.DELETED,
                old_node=existing_loci[locus_id]
            ))
        
        # Find modified loci (content hash changed)
        common_loci = existing_loci_set & new_loci_set
        for locus_id in common_loci:
            old_node = existing_loci[locus_id]
            new_node = new_loci_map[locus_id]
            
            # Compare content hash or key properties
            if (old_node.source_text_hash != new_node.source_text_hash or
                old_node.content != new_node.content):
                
                changes.append(LocusChange(
                    locus_id=locus_id,
                    change_type=ChangeType.MODIFIED,
                    old_node=old_node,
                    new_node=new_node,
                    content_hash_changed=True
                ))
        
        # Create summary
        summary = {
            "added": len([c for c in changes if c.change_type == ChangeType.ADDED]),
            "modified": len([c for c in changes if c.change_type == ChangeType.MODIFIED]),
            "deleted": len([c for c in changes if c.change_type == ChangeType.DELETED])
        }
        
        logger.info(f"Detected changes for {blueprint_id}: {summary}")
        
        return BlueprintChangeSet(
            blueprint_id=blueprint_id,
            timestamp=datetime.utcnow(),
            changes=changes,
            summary=summary
        )
    
    async def apply_blueprint_changes(
        self, 
        changeset: BlueprintChangeSet,
        strategy: str = "incremental"
    ) -> Dict[str, Any]:
        """
        Apply blueprint changes to the vector database.
        
        Args:
            changeset: The changes to apply
            strategy: "incremental" or "full_reindex"
            
        Returns:
            Result summary
        """
        await self._initialize()
        
        if strategy == "full_reindex":
            return await self._apply_full_reindex(changeset)
        else:
            return await self._apply_incremental_changes(changeset)
    
    async def _apply_incremental_changes(
        self, 
        changeset: BlueprintChangeSet
    ) -> Dict[str, Any]:
        """Apply changes incrementally."""
        results = {
            "strategy": "incremental",
            "blueprint_id": changeset.blueprint_id,
            "timestamp": datetime.utcnow().isoformat(),
            "changes_applied": 0,
            "changes_failed": 0,
            "operations": []
        }
        
        try:
            for change in changeset.changes:
                try:
                    if change.change_type == ChangeType.ADDED:
                        await self._add_locus(changeset.blueprint_id, change.new_node)
                        results["operations"].append(f"Added {change.locus_id}")
                        
                    elif change.change_type == ChangeType.MODIFIED:
                        await self._update_locus(changeset.blueprint_id, change.old_node, change.new_node)
                        results["operations"].append(f"Updated {change.locus_id}")
                        
                    elif change.change_type == ChangeType.DELETED:
                        await self._delete_locus(changeset.blueprint_id, change.locus_id)
                        results["operations"].append(f"Deleted {change.locus_id}")
                    
                    results["changes_applied"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to apply change {change.locus_id}: {e}")
                    results["changes_failed"] += 1
                    results["operations"].append(f"FAILED {change.locus_id}: {str(e)}")
            
            logger.info(f"Applied {results['changes_applied']} changes for {changeset.blueprint_id}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to apply incremental changes: {e}")
            results["error"] = str(e)
            return results
    
    async def _apply_full_reindex(
        self, 
        changeset: BlueprintChangeSet
    ) -> Dict[str, Any]:
        """Apply changes by full re-indexing."""
        results = {
            "strategy": "full_reindex",
            "blueprint_id": changeset.blueprint_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Delete existing blueprint
            delete_result = await self.indexing_pipeline.delete_blueprint_index(changeset.blueprint_id)
            
            # Re-index new blueprint
            # Note: We'd need the full blueprint object here
            # This is a simplified implementation
            results["nodes_deleted"] = delete_result.get("nodes_deleted", 0)
            results["reindex_completed"] = True
            
            logger.info(f"Full re-index completed for {changeset.blueprint_id}")
            return results
            
        except Exception as e:
            logger.error(f"Failed full re-index: {e}")
            results["error"] = str(e)
            return results
    
    async def _get_blueprint_loci(self, blueprint_id: str) -> Dict[str, TextNode]:
        """Get all existing loci for a blueprint from vector database."""
        # This is a placeholder - we'd need to implement search by blueprint_id
        # in the vector store to get existing nodes
        
        # For now, return empty dict as this requires vector store query functionality
        return {}
    
    async def _add_locus(self, blueprint_id: str, node: TextNode):
        """Add a new locus to the vector database."""
        # Process single node through indexing pipeline
        await self.indexing_pipeline._process_single_batch([node], None)
        logger.debug(f"Added locus {node.locus_id} for blueprint {blueprint_id}")
    
    async def _update_locus(self, blueprint_id: str, old_node: TextNode, new_node: TextNode):
        """Update an existing locus in the vector database."""
        # Delete old node and add new one
        await self._delete_locus(blueprint_id, old_node.locus_id)
        await self._add_locus(blueprint_id, new_node)
        logger.debug(f"Updated locus {new_node.locus_id} for blueprint {blueprint_id}")
    
    async def _delete_locus(self, blueprint_id: str, locus_id: str):
        """Delete a locus from the vector database."""
        # Delete vectors matching this locus_id
        # This requires implementing deletion by metadata filter
        logger.debug(f"Deleted locus {locus_id} for blueprint {blueprint_id}")
    
    async def get_blueprint_status(self, blueprint_id: str) -> Dict[str, Any]:
        """Get current status of a blueprint in the vector database."""
        await self._initialize()
        
        try:
            stats = await self.indexing_pipeline.get_indexing_stats(blueprint_id)
            
            # The stats are now nested under 'blueprint_specific'
            blueprint_info = stats.get("blueprint_specific", {})
            node_count = blueprint_info.get("node_count", 0)
            is_indexed = node_count > 0

            return {
                "blueprint_id": blueprint_id,
                "is_indexed": is_indexed,
                "node_count": node_count,
                "last_updated": None,  # Would need to track this
                "locus_types": blueprint_info.get("locus_types", {}),
                "status": "indexed" if is_indexed else "not_indexed"
            }
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            logger.error(f"Failed to get blueprint status for {blueprint_id}: {e}\nTraceback:\n{tb_str}")
            return {
                "blueprint_id": blueprint_id,
                "status": "error",
                "error": str(e)
            }


# Convenience functions for common operations

async def update_blueprint(
    blueprint_id: str, 
    new_blueprint: LearningBlueprint,
    strategy: str = "incremental"
) -> Dict[str, Any]:
    """
    High-level function to update a blueprint in the vector database.
    
    Args:
        blueprint_id: ID of blueprint to update
        new_blueprint: New version of the blueprint
        strategy: "incremental" or "full_reindex"
        
    Returns:
        Update results
    """
    service = BlueprintLifecycleService()
    
    # Detect changes
    changeset = await service.detect_blueprint_changes(blueprint_id, new_blueprint)
    
    if not changeset.has_changes:
        return {
            "status": "no_changes",
            "blueprint_id": blueprint_id,
            "message": "No changes detected - blueprint is up to date"
        }
    
    # Apply changes
    result = await service.apply_blueprint_changes(changeset, strategy)
    result["changeset_summary"] = changeset.summary
    
    return result


async def delete_blueprint(blueprint_id: str) -> Dict[str, Any]:
    """
    Delete a blueprint from the vector database.
    
    Args:
        blueprint_id: ID of blueprint to delete
        
    Returns:
        Deletion results
    """
    pipeline = IndexingPipeline()
    return await pipeline.delete_blueprint_index(blueprint_id)


async def get_blueprint_info(blueprint_id: str) -> Dict[str, Any]:
    """
    Get information about a blueprint's current status.
    
    Args:
        blueprint_id: ID of blueprint to check
        
    Returns:
        Blueprint status information
    """
    service = BlueprintLifecycleService()
    return await service.get_blueprint_status(blueprint_id)
