"""
Blueprint vector sync module.

This module provides vector synchronization functionality for blueprints.
"""

from typing import List, Dict, Any, Optional
from .blueprint_model import Blueprint


class BlueprintVectorSyncError(Exception):
    """Exception raised when blueprint vector sync operations fail."""
    pass


class BlueprintVectorSync:
    """Vector sync class for blueprint operations."""
    
    def __init__(self):
        """Initialize the blueprint vector sync system."""
        self.vector_store: Dict[str, Dict[str, Any]] = {}
        self.sync_status: Dict[str, str] = {}  # 'synced', 'pending', 'failed'
        self.last_sync: Dict[str, Any] = {}
    
    async def sync_blueprint_vectors(self, blueprint: Blueprint) -> bool:
        """Synchronize vectors for a blueprint."""
        try:
            blueprint_id = blueprint.id
            
            # Mark as pending
            self.sync_status[blueprint_id] = 'pending'
            
            # Create mock vectors (in a real system, this would call an embedding service)
            vectors = await self._create_vectors(blueprint)
            
            # Store vectors
            self.vector_store[blueprint_id] = {
                'vectors': vectors,
                'metadata': {
                    'name': blueprint.name,
                    'description': blueprint.description,
                    'content_length': len(blueprint.content),
                    'chunk_count': len(vectors)
                },
                'last_updated': blueprint.updated_at
            }
            
            # Mark as synced
            self.sync_status[blueprint_id] = 'synced'
            self.last_sync[blueprint_id] = {
                'timestamp': blueprint.updated_at,
                'status': 'success',
                'vector_count': len(vectors)
            }
            
            return True
            
        except Exception as e:
            if blueprint_id in self.sync_status:
                self.sync_status[blueprint_id] = 'failed'
                self.last_sync[blueprint_id] = {
                    'timestamp': blueprint.updated_at,
                    'status': 'failed',
                    'error': str(e)
                }
            raise BlueprintVectorSyncError(f"Failed to sync blueprint vectors: {str(e)}")
    
    async def get_blueprint_vectors(self, blueprint_id: str) -> Optional[Dict[str, Any]]:
        """Get vectors for a specific blueprint."""
        try:
            return self.vector_store.get(blueprint_id)
        except Exception as e:
            raise BlueprintVectorSyncError(f"Failed to get blueprint vectors: {str(e)}")
    
    async def search_similar_vectors(self, query_vector: List[float], blueprint_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors within a blueprint."""
        try:
            if blueprint_id not in self.vector_store:
                return []
            
            blueprint_vectors = self.vector_store[blueprint_id]['vectors']
            results = []
            
            for i, vector in enumerate(blueprint_vectors):
                similarity = self._cosine_similarity(query_vector, vector)
                results.append({
                    'chunk_index': i,
                    'similarity': similarity,
                    'vector': vector
                })
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            raise BlueprintVectorSyncError(f"Failed to search similar vectors: {str(e)}")
    
    async def update_blueprint_vectors(self, blueprint: Blueprint) -> bool:
        """Update vectors for an existing blueprint."""
        try:
            blueprint_id = blueprint.id
            
            if blueprint_id not in self.vector_store:
                return await self.sync_blueprint_vectors(blueprint)
            
            # Check if update is needed
            stored_metadata = self.vector_store[blueprint_id]['metadata']
            if (stored_metadata['name'] == blueprint.name and 
                stored_metadata['description'] == blueprint.description and
                stored_metadata['content_length'] == len(blueprint.content)):
                return True  # No update needed
            
            # Perform full sync
            return await self.sync_blueprint_vectors(blueprint)
            
        except Exception as e:
            raise BlueprintVectorSyncError(f"Failed to update blueprint vectors: {str(e)}")
    
    async def remove_blueprint_vectors(self, blueprint_id: str) -> bool:
        """Remove vectors for a blueprint."""
        try:
            if blueprint_id in self.vector_store:
                del self.vector_store[blueprint_id]
            
            if blueprint_id in self.sync_status:
                del self.sync_status[blueprint_id]
            
            if blueprint_id in self.last_sync:
                del self.last_sync[blueprint_id]
            
            return True
            
        except Exception as e:
            raise BlueprintVectorSyncError(f"Failed to remove blueprint vectors: {str(e)}")
    
    async def get_sync_status(self, blueprint_id: str) -> Optional[str]:
        """Get the sync status for a blueprint."""
        return self.sync_status.get(blueprint_id)
    
    async def get_sync_history(self, blueprint_id: str) -> Optional[Dict[str, Any]]:
        """Get the sync history for a blueprint."""
        return self.last_sync.get(blueprint_id)
    
    async def get_vector_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            total_blueprints = len(self.vector_store)
            total_vectors = sum(len(data['vectors']) for data in self.vector_store.values())
            
            sync_status_counts = {}
            for status in self.sync_status.values():
                if status not in sync_status_counts:
                    sync_status_counts[status] = 0
                sync_status_counts[status] += 1
            
            return {
                'total_blueprints': total_blueprints,
                'total_vectors': total_vectors,
                'average_vectors_per_blueprint': total_vectors / total_blueprints if total_blueprints > 0 else 0,
                'sync_status_counts': sync_status_counts,
                'vector_store_size': len(str(self.vector_store))
            }
            
        except Exception as e:
            raise BlueprintVectorSyncError(f"Failed to get vector stats: {str(e)}")
    
    async def _create_vectors(self, blueprint: Blueprint) -> List[List[float]]:
        """Create vectors for a blueprint (mock implementation)."""
        # In a real system, this would chunk the content and call an embedding service
        content = blueprint.content
        chunk_size = 1000
        vectors = []
        
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            # Mock vector (1536 dimensions like OpenAI embeddings)
            vector = [0.1] * 1536
            vectors.append(vector)
        
        return vectors
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def has_vectors(self, blueprint_id: str) -> bool:
        """Check if a blueprint has vectors."""
        return blueprint_id in self.vector_store
    
    def is_synced(self, blueprint_id: str) -> bool:
        """Check if a blueprint is synced."""
        return self.sync_status.get(blueprint_id) == 'synced'
    
    def get_pending_syncs(self) -> List[str]:
        """Get list of blueprints with pending syncs."""
        return [bp_id for bp_id, status in self.sync_status.items() if status == 'pending']
    
    def get_failed_syncs(self) -> List[str]:
        """Get list of blueprints with failed syncs."""
        return [bp_id for bp_id, status in self.sync_status.items() if status == 'failed']
