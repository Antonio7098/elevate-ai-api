"""
Enhanced Vector Store Models for AI API

This module defines models for vector store operations that support hierarchical
blueprint section indexing and knowledge graph integration for improved context assembly.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

from .blueprint_centric import UueStage, DifficultyLevel


class IndexingStrategy(str, Enum):
    """Vector indexing strategies."""
    HIERARCHICAL = "hierarchical"      # Index by section hierarchy
    FLAT = "flat"                      # Flat indexing
    HYBRID = "hybrid"                  # Combined approach
    GRAPH_AWARE = "graph_aware"        # Knowledge graph aware


class VectorMetadata(BaseModel):
    """Metadata for vector embeddings."""
    content_id: str = Field(..., description="Content identifier")
    content_type: str = Field(..., description="Type of content")
    blueprint_id: int = Field(..., description="Blueprint ID")
    section_id: Optional[int] = Field(None, description="Section ID")
    
    # Content properties
    uue_stage: Optional[UueStage] = Field(None, description="UUE stage")
    difficulty: Optional[DifficultyLevel] = Field(None, description="Difficulty level")
    complexity_score: Optional[float] = Field(None, description="Complexity score")
    
    # Hierarchical information
    hierarchy_path: List[str] = Field(default_factory=list, description="Hierarchy path")
    depth: int = Field(default=0, description="Hierarchy depth")
    parent_section_id: Optional[int] = Field(None, description="Parent section ID")
    
    # Knowledge graph information
    graph_node_id: Optional[str] = Field(None, description="Knowledge graph node ID")
    relationship_tags: List[str] = Field(default_factory=list, description="Relationship tags")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    @field_validator('complexity_score')
    @classmethod
    def validate_complexity_score(cls, v):
        if v is not None and (v < 1.0 or v > 10.0):
            raise ValueError('Complexity score must be between 1.0 and 10.0')
        return v
    
    @field_validator('depth')
    @classmethod
    def validate_depth(cls, v):
        if v < 0 or v > 10:
            raise ValueError('Depth must be between 0 and 10')
        return v


class VectorEmbedding(BaseModel):
    """Vector embedding with metadata."""
    id: str = Field(..., description="Embedding ID")
    content: str = Field(..., description="Content text")
    embedding: List[float] = Field(..., description="Vector embedding")
    metadata: VectorMetadata = Field(..., description="Content metadata")
    
    # Vector properties
    dimension: int = Field(..., description="Embedding dimension")
    model: str = Field(..., description="Embedding model used")
    similarity_score: Optional[float] = Field(None, description="Similarity score")
    
    @field_validator('dimension')
    @classmethod
    def validate_dimension(cls, v):
        if v <= 0:
            raise ValueError('Dimension must be positive')
        return v
    
    @field_validator('similarity_score')
    @classmethod
    def validate_similarity_score(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError('Similarity score must be between 0.0 and 1.0')
        return v


class SearchQuery(BaseModel):
    """Vector search query."""
    query_text: str = Field(..., description="Search query text")
    user_id: int = Field(..., description="User ID")
    
    # Search options
    max_results: int = Field(default=20, description="Maximum results to return")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold")
    
    # Filtering options
    blueprint_id: Optional[int] = Field(None, description="Filter by blueprint ID")
    section_id: Optional[int] = Field(None, description="Filter by section ID")
    uue_stage: Optional[UueStage] = Field(None, description="Filter by UUE stage")
    difficulty: Optional[DifficultyLevel] = Field(None, description="Filter by difficulty level")
    
    # Hierarchical search options
    include_hierarchy: bool = Field(default=True, description="Include hierarchical context")
    max_depth: int = Field(default=3, description="Maximum hierarchy depth")
    
    # Knowledge graph options
    include_graph_context: bool = Field(default=True, description="Include knowledge graph context")
    relationship_types: List[str] = Field(default_factory=list, description="Preferred relationship types")
    
    @field_validator('max_results')
    @classmethod
    def validate_max_results(cls, v):
        if v < 1 or v > 100:
            raise ValueError('Max results must be between 1 and 100')
        return v
    
    @field_validator('similarity_threshold')
    @classmethod
    def validate_similarity_threshold(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Similarity threshold must be between 0.0 and 1.0')
        return v


class SearchResult(BaseModel):
    """Vector search result."""
    embedding: VectorEmbedding = Field(..., description="Matched embedding")
    similarity_score: float = Field(..., description="Similarity score")
    rank: int = Field(..., description="Result rank")
    
    # Context information
    hierarchy_context: Optional[List[str]] = Field(None, description="Hierarchical context")
    graph_context: Optional[List[str]] = Field(None, description="Knowledge graph context")
    
    # Relevance information
    relevance_factors: List[str] = Field(default_factory=list, description="Relevance factors")
    confidence_score: float = Field(default=1.0, description="Confidence in result")
    
    @field_validator('similarity_score', 'confidence_score')
    @classmethod
    def validate_score_range(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Score must be between 0.0 and 1.0')
        return v


class SearchResponse(BaseModel):
    """Vector search response."""
    query: SearchQuery = Field(..., description="Original search query")
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    
    # Search metadata
    total_results: int = Field(default=0, description="Total results found")
    search_time_ms: float = Field(default=0.0, description="Search time in milliseconds")
    index_stats: Optional[Dict[str, Any]] = Field(None, description="Index statistics")
    
    # Quality metrics
    average_similarity: float = Field(default=0.0, description="Average similarity score")
    result_diversity: float = Field(default=0.0, description="Result diversity score")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Search timestamp")
    
    def calculate_metrics(self):
        """Calculate search quality metrics."""
        if self.results:
            self.total_results = len(self.results)
            self.average_similarity = sum(r.similarity_score for r in self.results) / len(self.results)
            
            # Calculate diversity based on unique sections and UUE stages
            unique_sections = set(r.embedding.metadata.section_id for r in self.results if r.embedding.metadata.section_id)
            unique_stages = set(r.embedding.metadata.uue_stage for r in self.results if r.embedding.metadata.uue_stage)
            
            section_diversity = len(unique_sections) / max(len(self.results), 1)
            stage_diversity = len(unique_stages) / len(UueStage) if UueStage else 0
            
            self.result_diversity = (section_diversity + stage_diversity) / 2


class IndexingRequest(BaseModel):
    """Request for content indexing."""
    content_items: List[Dict[str, Any]] = Field(..., description="Content items to index")
    blueprint_id: int = Field(..., description="Blueprint ID")
    indexing_strategy: IndexingStrategy = Field(default=IndexingStrategy.HIERARCHICAL, description="Indexing strategy")
    
    # Indexing options
    update_existing: bool = Field(default=False, description="Update existing embeddings")
    include_metadata: bool = Field(default=True, description="Include metadata in index")
    batch_size: int = Field(default=100, description="Batch size for processing")
    
    # Hierarchical options
    build_hierarchy: bool = Field(default=True, description="Build section hierarchy")
    max_hierarchy_depth: int = Field(default=5, description="Maximum hierarchy depth")
    
    # Knowledge graph options
    extract_relationships: bool = Field(default=True, description="Extract knowledge graph relationships")
    relationship_threshold: float = Field(default=0.6, description="Relationship extraction threshold")
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        if v < 1 or v > 1000:
            raise ValueError('Batch size must be between 1 and 1000')
        return v
    
    @field_validator('relationship_threshold')
    @classmethod
    def validate_relationship_threshold(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Relationship threshold must be between 0.0 and 1.0')
        return v


class IndexingResponse(BaseModel):
    """Response from content indexing."""
    request: IndexingRequest = Field(..., description="Original indexing request")
    
    # Indexing results
    success: bool = Field(..., description="Whether indexing was successful")
    indexed_items: int = Field(default=0, description="Number of items indexed")
    updated_items: int = Field(default=0, description="Number of items updated")
    failed_items: int = Field(default=0, description="Number of items that failed")
    
    # Index statistics
    total_embeddings: int = Field(default=0, description="Total embeddings in index")
    index_size_mb: float = Field(default=0.0, description="Index size in MB")
    indexing_time_ms: float = Field(default=0.0, description="Indexing time in milliseconds")
    
    # Quality metrics
    average_embedding_quality: float = Field(default=0.0, description="Average embedding quality")
    hierarchy_completeness: float = Field(default=0.0, description="Hierarchy completeness")
    
    # Errors and warnings
    errors: List[str] = Field(default_factory=list, description="Indexing errors")
    warnings: List[str] = Field(default_factory=list, description="Indexing warnings")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Indexing timestamp")
    
    def calculate_metrics(self):
        """Calculate indexing quality metrics."""
        if self.indexed_items > 0:
            self.average_embedding_quality = (self.indexed_items - self.failed_items) / self.indexed_items
        else:
            self.average_embedding_quality = 0.0


class IndexHealth(BaseModel):
    """Vector index health status."""
    index_id: str = Field(..., description="Index identifier")
    status: str = Field(..., description="Index status")
    
    # Performance metrics
    total_vectors: int = Field(default=0, description="Total vectors in index")
    index_dimension: int = Field(default=0, description="Vector dimension")
    index_size_mb: float = Field(default=0.0, description="Index size in MB")
    
    # Health indicators
    fragmentation_level: float = Field(default=0.0, description="Index fragmentation level")
    query_performance_ms: float = Field(default=0.0, description="Average query performance")
    memory_usage_mb: float = Field(default=0.0, description="Memory usage in MB")
    
    # Timestamps
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    health_check_time: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    
    @field_validator('fragmentation_level')
    @classmethod
    def validate_fragmentation_level(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Fragmentation level must be between 0.0 and 1.0')
        return v

