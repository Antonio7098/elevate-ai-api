"""
Premium API schemas for advanced RAG features.
Defines request/response models for premium chat, graph search, and expert selection.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class ExpertType(str, Enum):
    """Types of expert agents available for premium users"""
    EXPLAINER = "explainer"
    ASSESSOR = "assessor"
    CURATOR = "curator"
    PLANNER = "planner"
    RESEARCHER = "researcher"

class PremiumChatRequest(BaseModel):
    """Request model for premium chat endpoint"""
    query: str = Field(..., description="User query")
    user_id: str = Field(..., description="User identifier")
    user_context: Dict[str, Any] = Field(default_factory=dict, description="User context")
    mode: Optional[str] = Field(default="chat", description="Learning mode")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    max_tokens: Optional[int] = Field(default=1000, description="Maximum response tokens")
    complexity: Optional[str] = Field(default="medium", description="Query complexity level")
    optimize_cost: Optional[bool] = Field(default=False, description="Enable cost optimization")
    operation: Optional[str] = Field(default=None, description="Specific operation to perform")

class PremiumChatResponse(BaseModel):
    """Response model for premium chat endpoint"""
    response: str = Field(..., description="AI response")
    experts_used: List[str] = Field(..., description="Expert agents used")
    confidence_score: float = Field(..., description="Confidence score [0..1]")
    timestamp: datetime = Field(..., description="Response timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    model_used: Optional[str] = Field(default=None, description="LLM model used for response")
    estimated_cost: Optional[float] = Field(default=0.0, description="Estimated cost of the response")
    token_count: Optional[int] = Field(default=0, description="Token count of the response")
    optimization_applied: Optional[bool] = Field(default=False, description="Whether cost optimization was applied")

class GraphSearchRequest(BaseModel):
    """Request model for graph search endpoint"""
    query: str = Field(..., description="Search query")
    user_id: str = Field(..., description="User identifier")
    depth: Optional[int] = Field(default=3, description="Graph traversal depth")
    max_results: Optional[int] = Field(default=10, description="Maximum results")

class GraphSearchResponse(BaseModel):
    """Response model for graph search endpoint"""
    results: List[Dict[str, Any]] = Field(..., description="Graph search results")
    query: str = Field(..., description="Original query")
    timestamp: datetime = Field(..., description="Search timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Search metadata")

class ExpertSelection(BaseModel):
    """Model for expert agent selection"""
    experts: List[str] = Field(..., description="Selected expert agents")
    reasoning: str = Field(..., description="Reasoning for expert selection")
    confidence: float = Field(..., description="Selection confidence [0..1]")

class Response(BaseModel):
    """Generic response model for expert agents"""
    content: str = Field(..., description="Response content")
    confidence: float = Field(..., description="Response confidence [0..1]")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")

class UserContext(BaseModel):
    """Model for user context in premium requests"""
    learning_analytics: Dict[str, Any] = Field(default_factory=dict, description="Learning analytics")
    memory_insights: List[Dict[str, Any]] = Field(default_factory=list, description="Memory insights")
    learning_paths: List[Dict[str, Any]] = Field(default_factory=list, description="Learning paths")
    cognitive_profile: Dict[str, Any] = Field(default_factory=dict, description="Cognitive profile")

class LangGraphChatRequest(BaseModel):
    """Request model for LangGraph chat endpoint"""
    query: str = Field(..., description="User query")
    user_id: str = Field(..., description="User identifier")
    user_context: Dict[str, Any] = Field(default_factory=dict, description="User context")
    session_id: Optional[str] = Field(default=None, description="Session identifier")

class LangGraphChatResponse(BaseModel):
    """Response model for LangGraph chat endpoint"""
    response: str = Field(..., description="AI response")
    agents_used: List[str] = Field(..., description="Agents used in workflow")
    workflow_status: str = Field(..., description="Workflow status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Workflow metadata")
    timestamp: datetime = Field(..., description="Response timestamp")

class LearningWorkflowRequest(BaseModel):
    """Request model for learning workflow endpoint"""
    learning_goal: str = Field(..., description="Learning goal")
    user_id: str = Field(..., description="User identifier")
    user_context: Dict[str, Any] = Field(default_factory=dict, description="User context")
    workflow_type: Optional[str] = Field(default="standard", description="Workflow type")

class LearningWorkflowResponse(BaseModel):
    """Response model for learning workflow endpoint"""
    learning_plan: str = Field(..., description="Generated learning plan")
    adapted_plan: Optional[str] = Field(default=None, description="Adapted learning plan")
    progress_evaluation: Optional[str] = Field(default=None, description="Progress evaluation")
    workflow_status: str = Field(..., description="Workflow status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Workflow metadata")
    timestamp: datetime = Field(..., description="Response timestamp")

class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status endpoint"""
    workflow_id: str = Field(..., description="Workflow identifier")
    status: str = Field(..., description="Workflow status")
    logs: List[Dict[str, Any]] = Field(default_factory=list, description="Workflow logs")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    timestamp: datetime = Field(..., description="Response timestamp")

class CAARequest(BaseModel):
    """Request model for Context Assembly Agent"""
    query: str = Field(..., description="User query")
    user_id: str = Field(..., description="User identifier")
    mode: str = Field(..., description="Learning mode (chat/quiz/deep_dive/walk_through/note_editing)")
    session_context: Dict[str, Any] = Field(default_factory=dict, description="Session context")
    hints: List[str] = Field(default_factory=list, description="Context hints")
    token_budget: int = Field(default=3000, description="Token budget")
    latency_budget_ms: int = Field(default=1200, description="Latency budget")

class CAAResponse(BaseModel):
    """Response model for Context Assembly Agent"""
    assembled_context: str = Field(..., description="Assembled context")
    short_context: str = Field(..., description="Short context summary")
    long_context: List[Dict[str, Any]] = Field(..., description="Detailed context chunks")
    knowledge_primitives: List[Dict[str, Any]] = Field(..., description="Knowledge primitives")
    examples: List[Dict[str, Any]] = Field(..., description="Relevant examples")
    tool_outputs: List[Dict[str, Any]] = Field(..., description="Tool outputs")
    sufficiency_score: float = Field(..., description="Sufficiency score [0..1]")
    token_count: int = Field(..., description="Total token count")
    rerank_scores: Dict[str, float] = Field(..., description="Reranking scores")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    cache_key: str = Field(..., description="Cache key for reuse")
    timestamp: datetime = Field(..., description="Timestamp")

class SessionContext(BaseModel):
    """Model for session context in CAA requests"""
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Conversation history")
    current_topic: str = Field(default="", description="Current learning topic")
    user_focus_areas: List[str] = Field(default_factory=list, description="User focus areas")
    learning_objectives: List[str] = Field(default_factory=list, description="Learning objectives")

class AdvancedSearchRequest(BaseModel):
    """Request model for advanced search endpoint"""
    query: str = Field(..., description="Search query")
    user_id: str = Field(..., description="User identifier")
    mode: Optional[str] = Field(default="chat", description="Search mode")
    max_results: Optional[int] = Field(default=20, description="Maximum results")
    strategy_weights: Optional[Dict[str, float]] = Field(default_factory=dict, description="Strategy weights")

class AdvancedSearchResponse(BaseModel):
    """Response model for advanced search endpoint"""
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    strategy_scores: Dict[str, float] = Field(..., description="Strategy scores")
    fusion_quality: float = Field(..., description="Fusion quality score")
    optimization_metrics: Dict[str, float] = Field(..., description="Optimization metrics")
    timestamp: datetime = Field(..., description="Search timestamp")

class MultiModalSearchRequest(BaseModel):
    """Request model for multi-modal search endpoint"""
    text_query: str = Field(..., description="Text query")
    user_id: str = Field(..., description="User identifier")
    image_query: Optional[str] = Field(default=None, description="Base64 encoded image")
    audio_query: Optional[str] = Field(default=None, description="Base64 encoded audio")
    code_query: Optional[str] = Field(default=None, description="Code snippet")
    diagram_query: Optional[str] = Field(default=None, description="Base64 encoded diagram")
    modality_weights: Optional[Dict[str, float]] = Field(default_factory=dict, description="Modality weights")

class MultiModalSearchResponse(BaseModel):
    """Response model for multi-modal search endpoint"""
    text_results: List[Dict[str, Any]] = Field(..., description="Text search results")
    image_results: List[Dict[str, Any]] = Field(..., description="Image search results")
    code_results: List[Dict[str, Any]] = Field(..., description="Code search results")
    diagram_results: List[Dict[str, Any]] = Field(..., description="Diagram search results")
    audio_results: List[Dict[str, Any]] = Field(..., description="Audio search results")
    fusion_scores: Dict[str, float] = Field(..., description="Fusion scores")
    cross_modal_relationships: List[Dict[str, Any]] = Field(..., description="Cross-modal relationships")
    text_response: str = Field(..., description="Text response")
    image_response: Optional[str] = Field(default=None, description="Base64 encoded image response")
    audio_response: Optional[str] = Field(default=None, description="Base64 encoded audio response")
    code_response: Optional[str] = Field(default=None, description="Code response")
    diagram_response: Optional[str] = Field(default=None, description="Base64 encoded diagram response")
    cross_modal_explanations: List[Dict[str, Any]] = Field(..., description="Cross-modal explanations")
    timestamp: datetime = Field(..., description="Search timestamp")

