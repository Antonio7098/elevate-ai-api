"""
TextNode model for representing searchable text chunks in the vector database.

This module defines the data structures used for storing and retrieving
text content with rich metadata for the RAG system.
"""

from pydantic import BaseModel, Field, field_serializer
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum


class LocusType(str, Enum):
    """Types of knowledge loci in the LearningBlueprint."""
    FOUNDATIONAL_CONCEPT = "foundational_concept"
    USE_CASE = "use_case"
    EXPLORATION = "exploration"
    KEY_TERM = "key_term"
    COMMON_MISCONCEPTION = "common_misconception"


class UUEStage(str, Enum):
    """UUE (Understand, Use, Explore) stages for learning progression."""
    UNDERSTAND = "understand"
    USE = "use"
    EXPLORE = "explore"


class TextNode(BaseModel):
    """Represents a searchable text chunk with rich metadata."""
    
    # Core identification
    id: str = Field(..., description="Unique identifier for the text node")
    content: str = Field(..., description="The actual text content")
    
    # Source information
    blueprint_id: str = Field(..., description="ID of the source LearningBlueprint")
    source_text_hash: Optional[str] = Field(None, description="Hash of the original source text")
    
    # Locus information (from LearningBlueprint)
    locus_id: Optional[str] = Field(None, description="ID of the locus in the blueprint")
    locus_type: Optional[LocusType] = Field(None, description="Type of the locus")
    locus_title: Optional[str] = Field(None, description="Title of the locus")
    uue_stage: Optional[UUEStage] = Field(None, description="UUE stage of the locus")
    
    # Relationship information
    pathway_ids: List[str] = Field(default_factory=list, description="IDs of pathways this node participates in")
    related_locus_ids: List[str] = Field(default_factory=list, description="IDs of related loci")
    
    # Content metadata
    chunk_index: Optional[int] = Field(None, description="Index of this chunk within the locus")
    total_chunks: Optional[int] = Field(None, description="Total number of chunks in the locus")
    word_count: Optional[int] = Field(None, description="Number of words in the content")
    
    # Vector information
    embedding_dimension: Optional[int] = Field(None, description="Dimension of the embedding vector")
    embedding_model: Optional[str] = Field(None, description="Model used for embedding")
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the node was created")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the node was last updated")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_serializer('created_at', 'updated_at')
    def serialize_datetime(self, dt: datetime) -> str:
        return dt.isoformat() if dt else None


class TextNodeCreate(BaseModel):
    """Schema for creating a new TextNode."""
    
    content: str = Field(..., description="The actual text content")
    blueprint_id: str = Field(..., description="ID of the source LearningBlueprint")
    source_text_hash: Optional[str] = Field(None, description="Hash of the original source text")
    
    # Locus information
    locus_id: Optional[str] = Field(None, description="ID of the locus in the blueprint")
    locus_type: Optional[LocusType] = Field(None, description="Type of the locus")
    locus_title: Optional[str] = Field(None, description="Title of the locus")
    uue_stage: Optional[UUEStage] = Field(None, description="UUE stage of the locus")
    
    # Relationship information
    pathway_ids: List[str] = Field(default_factory=list, description="IDs of pathways this node participates in")
    related_locus_ids: List[str] = Field(default_factory=list, description="IDs of related loci")
    
    # Content metadata
    chunk_index: Optional[int] = Field(None, description="Index of this chunk within the locus")
    total_chunks: Optional[int] = Field(None, description="Total number of chunks in the locus")
    word_count: Optional[int] = Field(None, description="Number of words in the content")
    
    # Vector information
    embedding_dimension: Optional[int] = Field(None, description="Dimension of the embedding vector")
    embedding_model: Optional[str] = Field(None, description="Model used for embedding")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TextNodeUpdate(BaseModel):
    """Schema for updating an existing TextNode."""
    
    content: Optional[str] = Field(None, description="The actual text content")
    locus_title: Optional[str] = Field(None, description="Title of the locus")
    pathway_ids: Optional[List[str]] = Field(None, description="IDs of pathways this node participates in")
    related_locus_ids: Optional[List[str]] = Field(None, description="IDs of related loci")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class TextNodeSearchResult(BaseModel):
    """Schema for search results containing TextNodes."""
    
    node: TextNode
    score: float = Field(..., description="Similarity score from vector search")
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the result was retrieved")
    
    @field_serializer('retrieved_at')
    def serialize_datetime(self, dt: datetime) -> str:
        return dt.isoformat() if dt else None


class TextNodeBatch(BaseModel):
    """Schema for batch operations on TextNodes."""
    
    nodes: List[TextNode] = Field(..., description="List of text nodes")
    total_count: int = Field(..., description="Total number of nodes in the batch")
    blueprint_id: str = Field(..., description="ID of the source blueprint")


def create_text_node_id(blueprint_id: str, locus_id: str, chunk_index: int) -> str:
    """Create a unique ID for a TextNode."""
    return f"{blueprint_id}:{locus_id}:{chunk_index}"


def calculate_word_count(text: str) -> int:
    """Calculate the word count of a text string."""
    return len(text.split())


def extract_searchable_metadata(node: TextNode) -> Dict[str, Any]:
    """Extract metadata that should be searchable in the vector database."""
    return {
        "blueprint_id": node.blueprint_id,
        "locus_id": node.locus_id,
        "locus_type": node.locus_type.value if node.locus_type else None,
        "locus_title": node.locus_title,
        "uue_stage": node.uue_stage.value if node.uue_stage else None,
        "pathway_ids": node.pathway_ids,
        "related_locus_ids": node.related_locus_ids,
        "chunk_index": node.chunk_index,
        "total_chunks": node.total_chunks,
        "word_count": node.word_count,
        "embedding_model": node.embedding_model,
        **node.metadata
    } 