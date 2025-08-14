"""
Knowledge Graph Models for AI API

This module defines models for knowledge graph relationships, graph traversal,
and integration with the existing RAG system for enhanced context assembly.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union, Set
from datetime import datetime
from enum import Enum

from .blueprint_centric import UueStage, DifficultyLevel


class RelationshipType(str, Enum):
    """Types of relationships between knowledge primitives."""
    PREREQUISITE = "prerequisite"      # Must be learned before
    RELATED = "related"                # Related concepts
    BUILDS_ON = "builds_on"           # Extends or builds upon
    EXTENDS = "extends"                # Further development
    CONTRADICTS = "contradicts"        # Opposing concepts
    COMPONENT_OF = "component_of"     # Part of a larger concept
    EXAMPLE_OF = "example_of"          # Example of a concept
    APPLICATION_OF = "application_of"  # Practical application


class GraphNodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    MASTERY_CRITERION = "mastery_criterion"
    KNOWLEDGE_PRIMITIVE = "knowledge_primitive"
    BLUEPRINT_SECTION = "blueprint_section"
    QUESTION_INSTANCE = "question_instance"
    USER_MASTERY = "user_mastery"


class GraphNode(BaseModel):
    """Node in the knowledge graph."""
    id: str = Field(..., description="Unique node ID")
    node_type: GraphNodeType = Field(..., description="Type of node")
    title: str = Field(..., description="Node title")
    description: Optional[str] = Field(None, description="Node description")
    
    # Content metadata
    content_hash: Optional[str] = Field(None, description="Hash of node content for deduplication")
    difficulty: Optional[DifficultyLevel] = Field(None, description="Difficulty level")
    uue_stage: Optional[UueStage] = Field(None, description="UUE stage")
    
    # Graph properties
    in_degree: int = Field(default=0, description="Number of incoming edges")
    out_degree: int = Field(default=0, description="Number of outgoing edges")
    centrality_score: Optional[float] = Field(None, description="Graph centrality score")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional node metadata")
    
    @field_validator('centrality_score')
    @classmethod
    def validate_centrality_score(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError('Centrality score must be between 0.0 and 1.0')
        return v


class GraphEdge(BaseModel):
    """Edge in the knowledge graph."""
    id: str = Field(..., description="Unique edge ID")
    source_node_id: str = Field(..., description="Source node ID")
    target_node_id: str = Field(..., description="Target node ID")
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    
    # Edge properties
    strength: float = Field(default=1.0, description="Relationship strength (0.0-1.0)")
    confidence: float = Field(default=1.0, description="Confidence in relationship (0.0-1.0)")
    bidirectional: bool = Field(default=False, description="Whether relationship is bidirectional")
    
    # Content
    description: Optional[str] = Field(None, description="Description of the relationship")
    evidence: List[str] = Field(default_factory=list, description="Evidence supporting this relationship")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional edge metadata")
    
    @field_validator('strength', 'confidence')
    @classmethod
    def validate_probability_range(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Value must be between 0.0 and 1.0')
        return v
    
    @field_validator('source_node_id', 'target_node_id')
    @classmethod
    def validate_different_nodes(cls, v, values):
        if 'source_node_id' in values and v == values['source_node_id']:
            raise ValueError('Source and target nodes must be different')
        return v


class KnowledgeGraph(BaseModel):
    """Complete knowledge graph representation."""
    id: str = Field(..., description="Graph ID")
    name: str = Field(..., description="Graph name")
    description: Optional[str] = Field(None, description="Graph description")
    
    # Graph structure
    nodes: List[GraphNode] = Field(default_factory=list, description="Graph nodes")
    edges: List[GraphEdge] = Field(default_factory=list, description="Graph edges")
    
    # Metadata
    blueprint_id: Optional[int] = Field(None, description="Associated blueprint ID")
    user_id: int = Field(..., description="Graph owner")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    # Graph statistics
    total_nodes: int = Field(default=0, description="Total number of nodes")
    total_edges: int = Field(default=0, description="Total number of edges")
    density: float = Field(default=0.0, description="Graph density")
    average_degree: float = Field(default=0.0, description="Average node degree")
    
    def calculate_statistics(self):
        """Calculate graph statistics."""
        self.total_nodes = len(self.nodes)
        self.total_edges = len(self.edges)
        
        if self.total_nodes > 1:
            max_edges = self.total_nodes * (self.total_nodes - 1)
            self.density = self.total_edges / max_edges if max_edges > 0 else 0.0
            self.average_degree = (2 * self.total_edges) / self.total_nodes
        else:
            self.density = 0.0
            self.average_degree = 0.0
    
    def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_edges_for_node(self, node_id: str) -> List[GraphEdge]:
        """Get all edges connected to a node."""
        edges = []
        for edge in self.edges:
            if edge.source_node_id == node_id or edge.target_node_id == node_id:
                edges.append(edge)
        return edges
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighbor node IDs for a given node."""
        neighbors = set()
        for edge in self.edges:
            if edge.source_node_id == node_id:
                neighbors.add(edge.target_node_id)
            if edge.target_node_id == node_id and edge.bidirectional:
                neighbors.add(edge.source_node_id)
        return list(neighbors)


# Graph Traversal Models
class TraversalOptions(BaseModel):
    """Options for graph traversal."""
    max_depth: int = Field(default=3, description="Maximum traversal depth")
    max_nodes: int = Field(default=100, description="Maximum number of nodes to visit")
    relationship_types: List[RelationshipType] = Field(default_factory=list, description="Allowed relationship types")
    min_strength: float = Field(default=0.5, description="Minimum relationship strength")
    min_confidence: float = Field(default=0.7, description="Minimum confidence threshold")
    include_metadata: bool = Field(default=True, description="Include node metadata in results")
    
    @field_validator('max_depth')
    @classmethod
    def validate_max_depth(cls, v):
        if v < 1 or v > 10:
            raise ValueError('Max depth must be between 1 and 10')
        return v
    
    @field_validator('max_nodes')
    @classmethod
    def validate_max_nodes(cls, v):
        if v < 1 or v > 1000:
            raise ValueError('Max nodes must be between 1 and 1000')
        return v


class TraversalNode(BaseModel):
    """Node in a traversal result."""
    node: GraphNode = Field(..., description="Graph node")
    depth: int = Field(..., description="Depth from start node")
    path: List[str] = Field(default_factory=list, description="Path from start to this node")
    relationship_strength: float = Field(default=1.0, description="Cumulative relationship strength")
    relationship_confidence: float = Field(default=1.0, description="Cumulative relationship confidence")


class GraphTraversalResult(BaseModel):
    """Result of a graph traversal."""
    start_node_id: str = Field(..., description="Starting node ID")
    traversal_options: TraversalOptions = Field(..., description="Traversal options used")
    
    # Traversal results
    visited_nodes: List[TraversalNode] = Field(default_factory=list, description="Visited nodes")
    traversal_paths: List[List[str]] = Field(default_factory=list, description="All traversal paths")
    
    # Statistics
    total_nodes_visited: int = Field(default=0, description="Total nodes visited")
    max_depth_reached: int = Field(default=0, description="Maximum depth reached")
    traversal_time_ms: float = Field(default=0.0, description="Traversal time in milliseconds")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Traversal timestamp")
    
    def calculate_statistics(self):
        """Calculate traversal statistics."""
        self.total_nodes_visited = len(self.visited_nodes)
        if self.visited_nodes:
            self.max_depth_reached = max(node.depth for node in self.visited_nodes)
    
    def get_nodes_at_depth(self, depth: int) -> List[TraversalNode]:
        """Get all nodes at a specific depth."""
        return [node for node in self.visited_nodes if node.depth == depth]
    
    def get_strongest_relationships(self, limit: int = 10) -> List[TraversalNode]:
        """Get nodes with strongest cumulative relationship strength."""
        sorted_nodes = sorted(self.visited_nodes, key=lambda x: x.relationship_strength, reverse=True)
        return sorted_nodes[:limit]


# Learning Path Discovery Models
class PathDiscoveryRequest(BaseModel):
    """Request for learning path discovery."""
    start_criterion_id: str = Field(..., description="Starting mastery criterion ID")
    target_criterion_id: str = Field(..., description="Target mastery criterion ID")
    user_id: int = Field(..., description="User ID")
    blueprint_id: int = Field(..., description="Blueprint ID")
    
    # Path constraints
    max_path_length: int = Field(default=10, description="Maximum path length")
    preferred_uue_stages: List[UueStage] = Field(default_factory=list, description="Preferred UUE stages")
    difficulty_preference: Optional[DifficultyLevel] = Field(None, description="Preferred difficulty level")
    include_prerequisites: bool = Field(default=True, description="Include prerequisite relationships")
    
    @field_validator('max_path_length')
    @classmethod
    def validate_max_path_length(cls, v):
        if v < 2 or v > 20:
            raise ValueError('Max path length must be between 2 and 20')
        return v


class LearningPathSegment(BaseModel):
    """Segment of a learning path."""
    criterion_id: str = Field(..., description="Mastery criterion ID")
    uue_stage: UueStage = Field(..., description="UUE stage")
    difficulty: DifficultyLevel = Field(..., description="Difficulty level")
    estimated_time: int = Field(default=0, description="Estimated time in minutes")
    prerequisites: List[str] = Field(default_factory=list, description="Prerequisite criterion IDs")
    
    # Path properties
    order: int = Field(..., description="Order in the path")
    relationship_strength: float = Field(default=1.0, description="Relationship strength to next segment")
    
    @field_validator('estimated_time')
    @classmethod
    def validate_estimated_time(cls, v):
        if v < 0 or v > 480:  # 0 to 8 hours
            raise ValueError('Estimated time must be between 0 and 480 minutes')
        return v


class LearningPathDiscoveryResult(BaseModel):
    """Result of learning path discovery."""
    request: PathDiscoveryRequest = Field(..., description="Original discovery request")
    
    # Discovered paths
    primary_path: List[LearningPathSegment] = Field(default_factory=list, description="Primary learning path")
    alternative_paths: List[List[LearningPathSegment]] = Field(default_factory=list, description="Alternative paths")
    
    # Path analysis
    total_paths_found: int = Field(default=0, description="Total paths found")
    shortest_path_length: int = Field(default=0, description="Length of shortest path")
    longest_path_length: int = Field(default=0, description="Length of longest path")
    
    # Path quality metrics
    average_relationship_strength: float = Field(default=0.0, description="Average relationship strength")
    path_coherence_score: float = Field(default=0.0, description="Path coherence score")
    difficulty_progression_score: float = Field(default=0.0, description="Difficulty progression score")
    
    # Metadata
    discovery_time_ms: float = Field(default=0.0, description="Path discovery time in milliseconds")
    created_at: datetime = Field(default_factory=datetime.now, description="Discovery timestamp")
    
    def calculate_path_statistics(self):
        """Calculate path statistics."""
        all_paths = [self.primary_path] + self.alternative_paths
        self.total_paths_found = len(all_paths)
        
        if all_paths:
            path_lengths = [len(path) for path in all_paths if path]
            if path_lengths:
                self.shortest_path_length = min(path_lengths)
                self.longest_path_length = max(path_lengths)
    
    def get_path_by_length(self, length: int) -> Optional[List[LearningPathSegment]]:
        """Get a path of specific length."""
        all_paths = [self.primary_path] + self.alternative_paths
        for path in all_paths:
            if len(path) == length:
                return path
        return None


# Context Assembly Models
class ContextAssemblyRequest(BaseModel):
    """Request for context assembly using knowledge graph."""
    query: str = Field(..., description="User query")
    user_id: int = Field(..., description="User ID")
    blueprint_id: Optional[int] = Field(None, description="Blueprint ID for context")
    
    # Context options
    max_context_nodes: int = Field(default=20, description="Maximum context nodes")
    include_relationships: bool = Field(default=True, description="Include relationship information")
    include_metadata: bool = Field(default=True, description="Include node metadata")
    context_depth: int = Field(default=2, description="Context assembly depth")
    
    # Filtering options
    relationship_types: List[RelationshipType] = Field(default_factory=list, description="Preferred relationship types")
    min_strength: float = Field(default=0.5, description="Minimum relationship strength")
    uue_stage_filter: Optional[UueStage] = Field(None, description="Filter by UUE stage")
    difficulty_filter: Optional[DifficultyLevel] = Field(None, description="Filter by difficulty level")
    
    @field_validator('max_context_nodes')
    @classmethod
    def validate_max_context_nodes(cls, v):
        if v < 5 or v > 100:
            raise ValueError('Max context nodes must be between 5 and 100')
        return v
    
    @field_validator('context_depth')
    @classmethod
    def validate_context_depth(cls, v):
        if v < 1 or v > 5:
            raise ValueError('Context depth must be between 1 and 5')
        return v


class ContextNode(BaseModel):
    """Node in assembled context."""
    node: GraphNode = Field(..., description="Graph node")
    relevance_score: float = Field(..., description="Relevance to query (0.0-1.0)")
    relationship_path: List[str] = Field(default_factory=list, description="Path to query node")
    relationship_strength: float = Field(default=1.0, description="Cumulative relationship strength")
    
    # Context-specific information
    context_rank: int = Field(default=0, description="Rank in context assembly")
    context_depth: int = Field(default=0, description="Depth in context tree")
    
    @field_validator('relevance_score')
    @classmethod
    def validate_relevance_score(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Relevance score must be between 0.0 and 1.0')
        return v


class ContextAssemblyResult(BaseModel):
    """Result of context assembly."""
    request: ContextAssemblyRequest = Field(..., description="Original assembly request")
    
    # Assembled context
    context_nodes: List[ContextNode] = Field(default_factory=list, description="Assembled context nodes")
    context_edges: List[GraphEdge] = Field(default_factory=list, description="Context relationships")
    
    # Context quality metrics
    context_coverage: float = Field(default=0.0, description="Context coverage score")
    context_relevance: float = Field(default=0.0, description="Average context relevance")
    context_diversity: float = Field(default=0.0, description="Context diversity score")
    
    # Assembly statistics
    total_nodes_considered: int = Field(default=0, description="Total nodes considered")
    assembly_time_ms: float = Field(default=0.0, description="Assembly time in milliseconds")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Assembly timestamp")
    
    def get_top_context_nodes(self, limit: int = 10) -> List[ContextNode]:
        """Get top-ranked context nodes."""
        sorted_nodes = sorted(self.context_nodes, key=lambda x: x.context_rank)
        return sorted_nodes[:limit]
    
    def get_context_by_depth(self, depth: int) -> List[ContextNode]:
        """Get context nodes at a specific depth."""
        return [node for node in self.context_nodes if node.context_depth == depth]
    
    def calculate_context_metrics(self):
        """Calculate context quality metrics."""
        if self.context_nodes:
            self.context_relevance = sum(node.relevance_score for node in self.context_nodes) / len(self.context_nodes)
            
            # Calculate diversity based on unique node types and UUE stages
            node_types = set(node.node.node_type for node in self.context_nodes)
            uue_stages = set(node.node.uue_stage for node in self.context_nodes if node.node.uue_stage)
            
            type_diversity = len(node_types) / len(GraphNodeType)
            stage_diversity = len(uue_stages) / len(UueStage) if UueStage else 0
            
            self.context_diversity = (type_diversity + stage_diversity) / 2

