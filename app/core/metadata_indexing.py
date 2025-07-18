"""
Metadata indexing service for fast filtering and retrieval.

This module provides optimized metadata indexing to improve performance
of filtering operations by locus type, UUE stage, and relationships.
"""

import logging
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
from datetime import datetime
import asyncio
import json

from app.core.vector_store import VectorStore, VectorStoreError
from app.models.text_node import TextNode, LocusType, UUEStage

logger = logging.getLogger(__name__)


class MetadataIndexingError(Exception):
    """Exception raised for metadata indexing errors."""
    pass


class MetadataIndex:
    """
    In-memory metadata index for fast filtering.
    
    This class maintains indexes for common metadata filters to avoid
    scanning all vectors for metadata-based queries.
    """
    
    def __init__(self):
        # Index by locus type
        self.locus_type_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Index by UUE stage
        self.uue_stage_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Index by blueprint ID
        self.blueprint_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Index by locus ID
        self.locus_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Relationship index: locus_id -> set of related locus_ids
        self.relationship_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Reverse relationship index: target_locus_id -> set of source locus_ids
        self.reverse_relationship_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Relationship type index: relationship_type -> set of (source, target) pairs
        self.relationship_type_index: Dict[str, Set[tuple]] = defaultdict(set)
        
        # Word count index for content filtering
        self.word_count_index: Dict[str, int] = {}
        
        # Last updated timestamp
        self.last_updated = datetime.utcnow()
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    async def add_node(self, node_id: str, metadata: Dict[str, Any]) -> None:
        """Add a node to the metadata index."""
        async with self._lock:
            # Index by locus type
            locus_type = metadata.get("locus_type")
            if locus_type:
                self.locus_type_index[locus_type].add(node_id)
            
            # Index by UUE stage
            uue_stage = metadata.get("uue_stage")
            if uue_stage:
                self.uue_stage_index[uue_stage].add(node_id)
            
            # Index by blueprint ID
            blueprint_id = metadata.get("blueprint_id")
            if blueprint_id:
                self.blueprint_index[blueprint_id].add(node_id)
            
            # Index by locus ID
            locus_id = metadata.get("locus_id")
            if locus_id:
                self.locus_index[locus_id].add(node_id)
            
            # Index relationships
            relationships = metadata.get("relationships", [])
            if relationships and locus_id:
                for rel in relationships:
                    target_locus = rel.get("target_locus_id")
                    rel_type = rel.get("relationship_type")
                    
                    if target_locus:
                        self.relationship_index[locus_id].add(target_locus)
                        self.reverse_relationship_index[target_locus].add(locus_id)
                        
                        if rel_type:
                            self.relationship_type_index[rel_type].add((locus_id, target_locus))
            
            # Index word count
            word_count = metadata.get("word_count", 0)
            self.word_count_index[node_id] = word_count
            
            self.last_updated = datetime.utcnow()
    
    async def remove_node(self, node_id: str) -> None:
        """Remove a node from the metadata index."""
        async with self._lock:
            # Remove from all indexes
            for locus_type_nodes in self.locus_type_index.values():
                locus_type_nodes.discard(node_id)
            
            for uue_stage_nodes in self.uue_stage_index.values():
                uue_stage_nodes.discard(node_id)
            
            for blueprint_nodes in self.blueprint_index.values():
                blueprint_nodes.discard(node_id)
            
            for locus_nodes in self.locus_index.values():
                locus_nodes.discard(node_id)
            
            # Remove from relationship indexes
            for related_nodes in self.relationship_index.values():
                related_nodes.discard(node_id)
            
            for related_nodes in self.reverse_relationship_index.values():
                related_nodes.discard(node_id)
            
            # Clean up relationship type index
            for rel_type, pairs in self.relationship_type_index.items():
                pairs_to_remove = {pair for pair in pairs if node_id in pair}
                pairs -= pairs_to_remove
            
            # Remove from word count index
            self.word_count_index.pop(node_id, None)
            
            self.last_updated = datetime.utcnow()
    
    async def filter_by_locus_type(self, locus_type: str) -> Set[str]:
        """Get node IDs filtered by locus type."""
        async with self._lock:
            return self.locus_type_index[locus_type].copy()
    
    async def filter_by_uue_stage(self, uue_stage: str) -> Set[str]:
        """Get node IDs filtered by UUE stage."""
        async with self._lock:
            return self.uue_stage_index[uue_stage].copy()
    
    async def filter_by_blueprint(self, blueprint_id: str) -> Set[str]:
        """Get node IDs filtered by blueprint ID."""
        async with self._lock:
            return self.blueprint_index[blueprint_id].copy()
    
    async def filter_by_locus(self, locus_id: str) -> Set[str]:
        """Get node IDs filtered by locus ID."""
        async with self._lock:
            return self.locus_index[locus_id].copy()
    
    async def get_related_loci(self, locus_id: str) -> Set[str]:
        """Get loci related to the given locus."""
        async with self._lock:
            return self.relationship_index[locus_id].copy()
    
    async def get_reverse_related_loci(self, locus_id: str) -> Set[str]:
        """Get loci that relate to the given locus."""
        async with self._lock:
            return self.reverse_relationship_index[locus_id].copy()
    
    async def filter_by_relationship_type(self, relationship_type: str) -> Set[tuple]:
        """Get (source, target) pairs filtered by relationship type."""
        async with self._lock:
            return self.relationship_type_index[relationship_type].copy()
    
    async def filter_by_word_count(self, min_count: Optional[int] = None,
                                   max_count: Optional[int] = None) -> Set[str]:
        """Get node IDs filtered by word count range."""
        async with self._lock:
            result = set()
            for node_id, count in self.word_count_index.items():
                if min_count is not None and count < min_count:
                    continue
                if max_count is not None and count > max_count:
                    continue
                result.add(node_id)
            return result
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the metadata index."""
        async with self._lock:
            return {
                "total_nodes": len(self.word_count_index),
                "locus_types": {k: len(v) for k, v in self.locus_type_index.items()},
                "uue_stages": {k: len(v) for k, v in self.uue_stage_index.items()},
                "blueprints": {k: len(v) for k, v in self.blueprint_index.items()},
                "total_relationships": sum(len(v) for v in self.relationship_index.values()),
                "relationship_types": {k: len(v) for k, v in self.relationship_type_index.items()},
                "last_updated": self.last_updated.isoformat()
            }
    
    async def clear(self) -> None:
        """Clear all indexes."""
        async with self._lock:
            self.locus_type_index.clear()
            self.uue_stage_index.clear()
            self.blueprint_index.clear()
            self.locus_index.clear()
            self.relationship_index.clear()
            self.reverse_relationship_index.clear()
            self.relationship_type_index.clear()
            self.word_count_index.clear()
            self.last_updated = datetime.utcnow()


class MetadataIndexingService:
    """
    Service for managing metadata indexes for fast filtering.
    """
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.index = MetadataIndex()
        self.index_name = "blueprint-nodes"
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the metadata indexing service."""
        if self._initialized:
            return
        
        try:
            await self.rebuild_index()
            self._initialized = True
            logger.info("Metadata indexing service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize metadata indexing service: {e}")
            raise MetadataIndexingError(f"Initialization failed: {e}")
    
    async def rebuild_index(self) -> None:
        """
        Rebuild the metadata index from the vector store.
        
        This should be called periodically or when significant changes occur.
        """
        try:
            # Clear existing index
            await self.index.clear()
            
            # This is a simplified approach - in a production system,
            # you'd want to implement pagination for large datasets
            stats = await self.vector_store.get_stats(self.index_name)
            total_vectors = stats.get("total_vector_count", 0)
            
            if total_vectors == 0:
                logger.info("No vectors found in index, metadata index is empty")
                return
            
            # For now, we'll use a placeholder approach since we don't have
            # a direct way to iterate through all vectors in the abstract base class
            # In a real implementation, this would be vector store specific
            logger.info(f"Rebuilding metadata index for {total_vectors} vectors")
            
            # TODO: Implement vector store specific methods to iterate through all vectors
            # For now, the index will be populated as new nodes are added
            
        except VectorStoreError as e:
            logger.error(f"Failed to rebuild metadata index: {e}")
            raise MetadataIndexingError(f"Index rebuild failed: {e}")
    
    async def add_node_to_index(self, node_id: str, metadata: Dict[str, Any]) -> None:
        """Add a node to the metadata index."""
        try:
            await self.index.add_node(node_id, metadata)
            logger.debug(f"Added node {node_id} to metadata index")
        except Exception as e:
            logger.error(f"Failed to add node {node_id} to metadata index: {e}")
            raise MetadataIndexingError(f"Failed to add node to index: {e}")
    
    async def remove_node_from_index(self, node_id: str) -> None:
        """Remove a node from the metadata index."""
        try:
            await self.index.remove_node(node_id)
            logger.debug(f"Removed node {node_id} from metadata index")
        except Exception as e:
            logger.error(f"Failed to remove node {node_id} from metadata index: {e}")
            raise MetadataIndexingError(f"Failed to remove node from index: {e}")
    
    async def get_filtered_node_ids(self, filters: Dict[str, Any]) -> Set[str]:
        """
        Get node IDs that match the given filters using the metadata index.
        
        This is much faster than scanning all vectors for metadata matches.
        """
        try:
            # Start with all nodes if no filters
            if not filters:
                return set(self.index.word_count_index.keys())
            
            # Get initial set from the most selective filter
            result_sets = []
            
            # Filter by locus type
            if "locus_type" in filters:
                locus_type_nodes = await self.index.filter_by_locus_type(filters["locus_type"])
                result_sets.append(locus_type_nodes)
            
            # Filter by UUE stage
            if "uue_stage" in filters:
                uue_stage_nodes = await self.index.filter_by_uue_stage(filters["uue_stage"])
                result_sets.append(uue_stage_nodes)
            
            # Filter by blueprint ID
            if "blueprint_id" in filters:
                blueprint_nodes = await self.index.filter_by_blueprint(filters["blueprint_id"])
                result_sets.append(blueprint_nodes)
            
            # Filter by locus ID
            if "locus_id" in filters:
                locus_nodes = await self.index.filter_by_locus(filters["locus_id"])
                result_sets.append(locus_nodes)
            
            # Filter by word count
            min_count = filters.get("min_word_count")
            max_count = filters.get("max_word_count")
            if min_count is not None or max_count is not None:
                word_count_nodes = await self.index.filter_by_word_count(min_count, max_count)
                result_sets.append(word_count_nodes)
            
            # Handle relationship filters
            if "relationships.target_locus_id" in filters:
                target_locus = filters["relationships.target_locus_id"]
                related_nodes = await self.index.get_reverse_related_loci(target_locus)
                if related_nodes:
                    # Get node IDs that have relationships to the target locus
                    nodes_with_relationships = set()
                    for locus_id in related_nodes:
                        locus_nodes = await self.index.filter_by_locus(locus_id)
                        nodes_with_relationships.update(locus_nodes)
                    result_sets.append(nodes_with_relationships)
            
            # Intersect all result sets
            if result_sets:
                result = result_sets[0]
                for result_set in result_sets[1:]:
                    result = result.intersection(result_set)
                return result
            else:
                return set()
            
        except Exception as e:
            logger.error(f"Failed to get filtered node IDs: {e}")
            raise MetadataIndexingError(f"Filter operation failed: {e}")
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the metadata index."""
        try:
            return await self.index.get_stats()
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            raise MetadataIndexingError(f"Failed to get stats: {e}")
    
    async def find_related_loci_fast(self, locus_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Fast relationship traversal using the metadata index.
        
        Args:
            locus_id: Source locus ID
            max_depth: Maximum traversal depth
            
        Returns:
            Dictionary with related loci and their relationships
        """
        try:
            visited = set()
            related_loci = {}
            
            async def traverse(current_locus: str, depth: int):
                if depth > max_depth or current_locus in visited:
                    return
                
                visited.add(current_locus)
                
                # Get directly related loci
                related = await self.index.get_related_loci(current_locus)
                for related_locus in related:
                    if related_locus not in related_loci:
                        related_loci[related_locus] = {
                            "depth": depth,
                            "source": current_locus
                        }
                    
                    # Recursively traverse if not at max depth
                    if depth < max_depth:
                        await traverse(related_locus, depth + 1)
            
            await traverse(locus_id, 1)
            
            return {
                "source_locus": locus_id,
                "related_loci": related_loci,
                "total_found": len(related_loci),
                "max_depth": max_depth
            }
            
        except Exception as e:
            logger.error(f"Failed to find related loci: {e}")
            raise MetadataIndexingError(f"Related loci search failed: {e}")
