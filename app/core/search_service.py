"""
Search service for metadata-based filtering and retrieval.

This module provides search capabilities for TextNodes with rich metadata filtering
including locus type, UUE stage, relationships, and content-based filtering.
"""

import time
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import logging

from app.core.vector_store import VectorStore, SearchResult, VectorStoreError
from app.core.embeddings import EmbeddingService, EmbeddingError
from app.models.text_node import TextNode, LocusType, UUEStage
from app.api.schemas import (
    SearchRequest, SearchResponse, SearchResultItem,
    RelatedLocusSearchRequest, RelatedLocusSearchResponse, RelatedLocusItem
)

logger = logging.getLogger(__name__)


class SearchServiceError(Exception):
    """Base exception for search service operations."""
    pass


class SearchService:
    """Service for performing metadata-based search and retrieval."""
    
    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.index_name = "blueprint-nodes"  # Default index name
    
    async def search_nodes(self, request: SearchRequest) -> SearchResponse:
        """
        Search for TextNodes with metadata filtering.
        
        Args:
            request: Search request with query and filters
            
        Returns:
            SearchResponse with filtered results
        """
        start_time = time.time()
        
        try:
            # Generate embedding for the query
            embedding_start = time.time()
            query_embedding = await self.embedding_service.embed_text(request.query)
            embedding_time = (time.time() - embedding_start) * 1000
            
            # Build metadata filters
            metadata_filters = self._build_metadata_filters(request)
            
            # Perform vector search
            search_start = time.time()
            results = await self.vector_store.search(
                index_name=self.index_name,
                query_vector=query_embedding,
                top_k=request.top_k,
                filter_metadata=metadata_filters
            )
            search_time = (time.time() - search_start) * 1000
            
            # Convert results to response format
            search_results = []
            for result in results:
                search_item = self._convert_search_result(result)
                search_results.append(search_item)
            
            # Apply additional content filtering if specified
            if request.min_chunk_size or request.max_chunk_size:
                search_results = self._filter_by_chunk_size(search_results, request)
            
            return SearchResponse(
                results=search_results,
                total_results=len(search_results),
                query=request.query,
                filters_applied=metadata_filters,
                search_time_ms=search_time,
                embedding_time_ms=embedding_time,
                created_at=datetime.utcnow().isoformat()
            )
            
        except EmbeddingError as e:
            logger.error(f"Embedding generation failed: {e}")
            raise SearchServiceError(f"Failed to generate query embedding: {e}")
        except VectorStoreError as e:
            logger.error(f"Vector store search failed: {e}")
            raise SearchServiceError(f"Vector search failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in search: {e}")
            raise SearchServiceError(f"Search operation failed: {e}")
    
    def _build_metadata_filters(self, request: SearchRequest) -> Dict[str, Any]:
        """Build metadata filters from search request."""
        filters = {}
        
        # Blueprint filtering
        if request.blueprint_id:
            filters["blueprint_id"] = request.blueprint_id
        
        # Locus type filtering
        if request.locus_type:
            filters["locus_type"] = request.locus_type
        
        # UUE stage filtering
        if request.uue_stage:
            filters["uue_stage"] = request.uue_stage
        
        # Relationship filtering
        if request.related_to_locus:
            # This creates a filter for nodes that have relationships to the specified locus
            filters["relationships.target_locus_id"] = request.related_to_locus
            
            if request.relationship_type:
                filters["relationships.relationship_type"] = request.relationship_type
        
        return filters
    
    def _convert_search_result(self, result: SearchResult) -> SearchResultItem:
        """Convert vector store search result to API response format."""
        metadata = result.metadata
        
        return SearchResultItem(
            id=result.id,
            content=result.content,
            score=result.score,
            blueprint_id=metadata.get("blueprint_id", ""),
            locus_id=metadata.get("locus_id", ""),
            locus_type=metadata.get("locus_type", ""),
            uue_stage=metadata.get("uue_stage", ""),
            chunk_index=metadata.get("chunk_index"),
            chunk_total=metadata.get("chunk_total"),
            word_count=metadata.get("word_count", 0),
            relationships=metadata.get("relationships", []),
            created_at=metadata.get("created_at", ""),
            indexed_at=metadata.get("indexed_at", "")
        )
    
    def _filter_by_chunk_size(self, results: List[SearchResultItem], request: SearchRequest) -> List[SearchResultItem]:
        """Apply chunk size filtering to search results."""
        filtered_results = []
        
        for result in results:
            word_count = result.word_count
            
            # Check minimum chunk size
            if request.min_chunk_size and word_count < request.min_chunk_size:
                continue
            
            # Check maximum chunk size
            if request.max_chunk_size and word_count > request.max_chunk_size:
                continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    async def search_by_locus_type(self, locus_type: LocusType, blueprint_id: Optional[str] = None, 
                                   limit: int = 50) -> List[SearchResultItem]:
        """
        Search for nodes by specific locus type.
        
        Args:
            locus_type: Type of locus to search for
            blueprint_id: Optional blueprint ID to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of matching nodes
        """
        try:
            filters = {"locus_type": locus_type.value}
            if blueprint_id:
                filters["blueprint_id"] = blueprint_id
            
            # Use a generic query vector (all zeros) since we're filtering by metadata only
            zero_vector = [0.0] * 1536  # Standard embedding dimension
            
            results = await self.vector_store.search(
                index_name=self.index_name,
                query_vector=zero_vector,
                top_k=limit,
                filter_metadata=filters
            )
            
            return [self._convert_search_result(result) for result in results]
            
        except VectorStoreError as e:
            logger.error(f"Failed to search by locus type: {e}")
            raise SearchServiceError(f"Locus type search failed: {e}")
    
    async def search_by_uue_stage(self, uue_stage: UUEStage, blueprint_id: Optional[str] = None,
                                  limit: int = 50) -> List[SearchResultItem]:
        """
        Search for nodes by UUE stage.
        
        Args:
            uue_stage: UUE stage to search for
            blueprint_id: Optional blueprint ID to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of matching nodes
        """
        try:
            filters = {"uue_stage": uue_stage.value}
            if blueprint_id:
                filters["blueprint_id"] = blueprint_id
            
            # Use a generic query vector (all zeros) since we're filtering by metadata only
            zero_vector = [0.0] * 1536  # Standard embedding dimension
            
            results = await self.vector_store.search(
                index_name=self.index_name,
                query_vector=zero_vector,
                top_k=limit,
                filter_metadata=filters
            )
            
            return [self._convert_search_result(result) for result in results]
            
        except VectorStoreError as e:
            logger.error(f"Failed to search by UUE stage: {e}")
            raise SearchServiceError(f"UUE stage search failed: {e}")
    
    async def find_related_loci(self, request: RelatedLocusSearchRequest) -> RelatedLocusSearchResponse:
        """
        Find loci related to a specific locus through relationships.
        
        Args:
            request: Related locus search request
            
        Returns:
            Related loci with relationship information
        """
        try:
            related_loci = []
            visited = set()
            
            # Start with the source locus
            await self._traverse_relationships(
                locus_id=request.locus_id,
                current_depth=0,
                max_depth=request.max_depth,
                relationship_types=request.relationship_types,
                include_reverse=request.include_reverse,
                visited=visited,
                related_loci=related_loci,
                path=[request.locus_id]
            )
            
            return RelatedLocusSearchResponse(
                source_locus_id=request.locus_id,
                related_loci=related_loci,
                total_related=len(related_loci),
                max_depth_reached=min(request.max_depth, max(item.depth for item in related_loci) if related_loci else 0),
                created_at=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Failed to find related loci: {e}")
            raise SearchServiceError(f"Related loci search failed: {e}")
    
    async def _traverse_relationships(self, locus_id: str, current_depth: int, max_depth: int,
                                      relationship_types: Optional[List[str]], include_reverse: bool,
                                      visited: Set[str], related_loci: List[RelatedLocusItem],
                                      path: List[str]) -> None:
        """
        Recursively traverse relationships to find related loci.
        
        Args:
            locus_id: Current locus ID
            current_depth: Current traversal depth
            max_depth: Maximum depth to traverse
            relationship_types: Filter by relationship types
            include_reverse: Include reverse relationships
            visited: Set of visited loci to avoid cycles
            related_loci: List to accumulate results
            path: Current path from source
        """
        if current_depth >= max_depth or locus_id in visited:
            return
        
        visited.add(locus_id)
        
        # Find nodes with relationships to this locus
        filters = {"relationships.target_locus_id": locus_id}
        if relationship_types:
            filters["relationships.relationship_type"] = {"$in": relationship_types}
        
        try:
            zero_vector = [0.0] * 1536
            results = await self.vector_store.search(
                index_name=self.index_name,
                query_vector=zero_vector,
                top_k=100,  # Get more results for relationship traversal
                filter_metadata=filters
            )
            
            for result in results:
                relationships = result.metadata.get("relationships", [])
                
                for relationship in relationships:
                    target_locus = relationship.get("target_locus_id")
                    rel_type = relationship.get("relationship_type")
                    
                    if target_locus and target_locus != locus_id:
                        # Filter by relationship type if specified
                        if relationship_types and rel_type not in relationship_types:
                            continue
                        
                        # Add to related loci
                        related_item = RelatedLocusItem(
                            locus_id=target_locus,
                            relationship_type=rel_type,
                            relationship_strength=relationship.get("strength", 1.0),
                            depth=current_depth + 1,
                            path=path + [target_locus],
                            locus_type=result.metadata.get("locus_type", ""),
                            blueprint_id=result.metadata.get("blueprint_id", ""),
                            content_preview=result.content[:200] + "..." if len(result.content) > 200 else result.content
                        )
                        related_loci.append(related_item)
                        
                        # Recursively traverse if not at max depth
                        if current_depth + 1 < max_depth:
                            await self._traverse_relationships(
                                target_locus, current_depth + 1, max_depth,
                                relationship_types, include_reverse, visited, related_loci,
                                path + [target_locus]
                            )
        
        except VectorStoreError as e:
            logger.warning(f"Failed to traverse relationships for locus {locus_id}: {e}")
            # Continue traversal even if one step fails
    
    async def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """
        Get search suggestions based on partial query.
        
        Args:
            partial_query: Partial search query
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested search terms
        """
        try:
            # This is a simple implementation - in production, you might want
            # to use a more sophisticated approach like n-gram matching
            suggestions = []
            
            # Search for nodes containing the partial query
            if len(partial_query) >= 3:  # Only search for queries with 3+ characters
                embedding = await self.embedding_service.embed_text(partial_query)
                
                results = await self.vector_store.search(
                    index_name=self.index_name,
                    query_vector=embedding,
                    top_k=limit * 2,  # Get more to filter
                    filter_metadata={}
                )
                
                # Extract unique terms from results
                terms = set()
                for result in results:
                    content_words = result.content.lower().split()
                    for word in content_words:
                        if partial_query.lower() in word and len(word) > len(partial_query):
                            terms.add(word)
                            if len(terms) >= limit:
                                break
                
                suggestions = list(terms)[:limit]
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to get search suggestions: {e}")
            return []  # Return empty list on error
