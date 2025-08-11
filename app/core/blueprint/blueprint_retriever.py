"""
Blueprint retriever module - adapter for existing functionality.

This module provides the retrieval interface expected by tests,
using functionality that already exists in the blueprint RAG system.
"""

from typing import List, Dict, Any, Optional
from app.models.blueprint import Blueprint
from app.core.blueprint.blueprint_rag import BlueprintRAG


class BlueprintRetriever:
    """Adapter for blueprint content retrieval functionality."""
    
    def __init__(self, rag_service: BlueprintRAG):
        """Initialize the retriever with RAG service."""
        self.rag_service = rag_service
    
    async def retrieve_relevant_chunks(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant content chunks for a query."""
        return await self.rag_service.retrieve_relevant_context(query, limit)
    
    async def retrieve_with_filters(self, query: str, filters: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve content with additional filters."""
        # Apply filters to the retrieval
        results = await self.rag_service.retrieve_relevant_context(query, limit * 2)
        
        # Filter results based on criteria
        filtered_results = []
        for result in results:
            if self._matches_filters(result, filters):
                filtered_results.append(result)
        
        return filtered_results[:limit]
    
    async def retrieve_multimodal(self, query: str, content_types: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve multimodal content."""
        # For now, just return regular retrieval
        # In a real implementation, this would handle different content types
        return await self.retrieve_relevant_chunks(query, limit)
    
    async def retrieve_temporal(self, query: str, time_range: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve content within a time range."""
        # For now, just return regular retrieval
        # In a real implementation, this would filter by time
        return await self.retrieve_relevant_chunks(query, limit)
    
    async def retrieve_hierarchical(self, query: str, hierarchy_level: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve content at a specific hierarchy level."""
        # For now, just return regular retrieval
        # In a real implementation, this would filter by hierarchy
        return await self.retrieve_relevant_chunks(query, limit)
    
    def _matches_filters(self, result: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if a result matches the given filters."""
        for key, value in filters.items():
            if key in result:
                if isinstance(value, list):
                    if not any(v in result[key] for v in value):
                        return False
                elif result[key] != value:
                    return False
        return True
