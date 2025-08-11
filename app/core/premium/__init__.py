"""
Premium module for advanced AI features.
Provides Context Assembly Agent, multi-agent orchestration, and premium capabilities.
"""

from .context_assembly_agent import ContextAssemblyAgent, CAARequest, CAAResponse, CAAState
from .langgraph_setup import PremiumAgentState, LangGraphSetup
from .core_api_client import CoreAPIClient
from .gemini_service import GeminiService
from .memory_system import PremiumMemorySystem
from .agents.routing_agent import PremiumRoutingAgent
from .graph_store import Neo4jGraphStore
from .agents.expert_agents import (
    ExplanationAgent, AssessmentAgent, ContentCuratorAgent,
    LearningPlannerAgent, ResearchAgent
)
from .workflows.learning_workflow import LearningWorkflow, AdaptiveLearningWorkflow
from .modes.mode_aware_assembly import ModeAwareAssembly

__all__ = [
    "ContextAssemblyAgent",
    "CAARequest",
    "CAAResponse", 
    "CAAState",
    "PremiumAgentState",
    "LangGraphSetup",
    "CoreAPIClient",
    "GeminiService",
    "PremiumMemorySystem",
    "PremiumRoutingAgent",
    "Neo4jGraphStore",
    "ExplanationAgent",
    "AssessmentAgent",
    "ContentCuratorAgent",
    "LearningPlannerAgent",
    "ResearchAgent",
    "LearningWorkflow",
    "AdaptiveLearningWorkflow",
    "ModeAwareAssembly"
]


