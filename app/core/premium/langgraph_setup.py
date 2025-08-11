"""
LangGraph integration and setup for premium multi-agent system.
Provides state management, workflow orchestration, and agent coordination.
"""

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, Dict, Any, List
from datetime import datetime
import json

class PremiumAgentState(TypedDict):
    """State management for premium agent workflows"""
    messages: Annotated[List[Dict[str, Any]], "The messages in the conversation"]
    user_query: Annotated[str, "The user's original query"]
    user_id: Annotated[str, "User identifier"]
    user_context: Annotated[Dict[str, Any], "User profile and preferences"]
    selected_agents: Annotated[List[str], "List of agents to involve"]
    context_assembled: Annotated[Dict[str, Any], "Assembled context from all sources"]
    final_response: Annotated[str, "The final response to the user"]
    workflow_status: Annotated[str, "Current workflow status"]
    agent_responses: Annotated[Dict[str, Any], "Responses from each agent"]
    metadata: Annotated[Dict[str, Any], "Additional workflow metadata"]

class LangGraphSetup:
    """LangGraph setup and configuration for premium workflows"""
    
    def __init__(self):
        self.state_persistence = StatePersistence()
        self.monitoring = LangGraphMonitoring()
    
    def create_base_state(self, user_query: str, user_id: str, user_context: Dict[str, Any]) -> PremiumAgentState:
        """Create initial state for premium agent workflows"""
        return PremiumAgentState(
            messages=[],
            user_query=user_query,
            user_id=user_id,
            user_context=user_context,
            selected_agents=[],
            context_assembled={},
            final_response="",
            workflow_status="initialized",
            agent_responses={},
            metadata={
                "start_time": datetime.utcnow().isoformat(),
                "workflow_type": "premium_multi_agent",
                "version": "1.0"
            }
        )
    
    def update_state(self, state: PremiumAgentState, updates: Dict[str, Any]) -> PremiumAgentState:
        """Update state with new information"""
        for key, value in updates.items():
            if key in state:
                state[key] = value
        
        # Update metadata
        state["metadata"]["last_updated"] = datetime.utcnow().isoformat()
        return state
    
    async def persist_state(self, state: PremiumAgentState, workflow_id: str):
        """Persist state for recovery and monitoring"""
        await self.state_persistence.save_state(workflow_id, state)
    
    async def recover_state(self, workflow_id: str) -> PremiumAgentState:
        """Recover state from persistence"""
        return await self.state_persistence.load_state(workflow_id)
    
    def get_workflow_status(self, state: PremiumAgentState) -> str:
        """Get current workflow status"""
        return state.get("workflow_status", "unknown")

class StatePersistence:
    """State persistence for LangGraph workflows"""
    
    def __init__(self):
        self.storage = {}  # In production, use Redis or database
    
    async def save_state(self, workflow_id: str, state: PremiumAgentState):
        """Save state to persistent storage"""
        try:
            # Convert state to JSON-serializable format
            serializable_state = self._make_serializable(state)
            self.storage[workflow_id] = serializable_state
        except Exception as e:
            print(f"Error saving state for workflow {workflow_id}: {e}")
    
    async def load_state(self, workflow_id: str) -> PremiumAgentState:
        """Load state from persistent storage"""
        try:
            if workflow_id in self.storage:
                return self.storage[workflow_id]
            else:
                raise ValueError(f"Workflow {workflow_id} not found")
        except Exception as e:
            print(f"Error loading state for workflow {workflow_id}: {e}")
            return None
    
    def _make_serializable(self, state: PremiumAgentState) -> Dict[str, Any]:
        """Convert state to JSON-serializable format"""
        serializable = {}
        for key, value in state.items():
            if isinstance(value, datetime):
                serializable[key] = value.isoformat()
            elif isinstance(value, (list, dict, str, int, float, bool)):
                serializable[key] = value
            else:
                serializable[key] = str(value)
        return serializable

class LangGraphMonitoring:
    """Monitoring and debugging tools for LangGraph workflows"""
    
    def __init__(self):
        self.workflow_logs = {}
        self.performance_metrics = {}
    
    def log_workflow_event(self, workflow_id: str, event: str, data: Dict[str, Any]):
        """Log workflow events for monitoring"""
        if workflow_id not in self.workflow_logs:
            self.workflow_logs[workflow_id] = []
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "data": data
        }
        self.workflow_logs[workflow_id].append(log_entry)
    
    def get_workflow_logs(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get logs for a specific workflow"""
        return self.workflow_logs.get(workflow_id, [])
    
    def record_performance_metric(self, workflow_id: str, metric: str, value: float):
        """Record performance metrics"""
        if workflow_id not in self.performance_metrics:
            self.performance_metrics[workflow_id] = {}
        
        self.performance_metrics[workflow_id][metric] = value
    
    def get_performance_metrics(self, workflow_id: str) -> Dict[str, float]:
        """Get performance metrics for a workflow"""
        return self.performance_metrics.get(workflow_id, {})
    
    def debug_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Get comprehensive debug information for a workflow"""
        return {
            "workflow_id": workflow_id,
            "logs": self.get_workflow_logs(workflow_id),
            "metrics": self.get_performance_metrics(workflow_id),
            "status": "active"  # In production, get from actual workflow state
        }












