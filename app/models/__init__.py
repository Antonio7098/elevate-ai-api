"""
AI API Models Package

This package contains all the data models for the AI API, including
blueprint-centric models, mastery tracking, knowledge graph, content generation,
and vector store models.
"""

# Blueprint-Centric Models
from .blueprint_centric import (
    UueStage,
    TrackingIntensity,
    DifficultyLevel,
    AssessmentType,
    BlueprintSection,
    MasteryCriterion,
    KnowledgePrimitive,
    LearningBlueprint,
    MasteryCriterionRelationship,
    QuestionInstance,
    ContentGenerationRequest,
    ContentGenerationResponse,
    SectionTree,
    BlueprintValidationResult
)

# Mastery Tracking Models
from .mastery_tracking import (
    MasteryThreshold,
    LearningStyle,
    ExperienceLevel,
    UserMasteryPreferences,
    SectionMasteryThreshold,
    CriterionMasteryThreshold,
    UserCriterionMastery,
    MasteryCalculationRequest,
    MasteryCalculationResult,
    MasteryPerformanceMetrics,
    LearningPathNode,
    LearningPath
)

# Knowledge Graph Models
from .knowledge_graph import (
    RelationshipType,
    GraphNodeType,
    GraphNode,
    GraphEdge,
    KnowledgeGraph,
    TraversalOptions,
    TraversalNode,
    GraphTraversalResult,
    PathDiscoveryRequest,
    LearningPathSegment,
    LearningPathDiscoveryResult,
    ContextAssemblyRequest,
    ContextNode,
    ContextAssemblyResult
)

# Content Generation Models
from .content_generation import (
    ContentType,
    GenerationStyle,
    QuestionType,
    ContentGenerationRequest as BaseContentGenerationRequest,
    MasteryCriteriaGenerationRequest,
    QuestionGenerationRequest,
    ContentGenerationResponse as BaseContentGenerationResponse,
    GeneratedMasteryCriterion,
    GeneratedQuestion,
    QuestionFamily
)

# Vector Store Models
from .vector_store import (
    IndexingStrategy,
    VectorMetadata,
    VectorEmbedding,
    SearchQuery,
    SearchResult,
    SearchResponse,
    IndexingRequest,
    IndexingResponse,
    IndexHealth
)

# Legacy Models (for backward compatibility)
from .learning_blueprint import (
    LearningBlueprint as LegacyLearningBlueprint,
    KnowledgePrimitives as LegacyKnowledgePrimitives
)

# Re-export commonly used models with aliases
__all__ = [
    # Blueprint-Centric Core
    'UueStage',
    'TrackingIntensity', 
    'DifficultyLevel',
    'AssessmentType',
    'BlueprintSection',
    'MasteryCriterion',
    'KnowledgePrimitive',
    'LearningBlueprint',
    
    # Mastery Tracking
    'MasteryThreshold',
    'LearningStyle',
    'ExperienceLevel',
    'UserMasteryPreferences',
    'UserCriterionMastery',
    'LearningPath',
    
    # Knowledge Graph
    'RelationshipType',
    'KnowledgeGraph',
    'GraphTraversalResult',
    'LearningPathDiscoveryResult',
    'ContextAssemblyResult',
    
    # Content Generation
    'ContentType',
    'GenerationStyle',
    'QuestionType',
    'MasteryCriteriaGenerationRequest',
    'QuestionGenerationRequest',
    'GeneratedMasteryCriterion',
    'GeneratedQuestion',
    'QuestionFamily',
    
    # Vector Store
    'IndexingStrategy',
    'VectorEmbedding',
    'SearchQuery',
    'SearchResponse',
    'IndexingRequest',
    'IndexingResponse',
    
    # Legacy (for compatibility)
    'LegacyLearningBlueprint',
    'LegacyKnowledgePrimitives'
]

