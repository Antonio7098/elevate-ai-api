"""
LangGraph-based routing agent for premium multi-agent system.
Provides sophisticated agent selection and orchestration using LangGraph workflows.
"""

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, Any, List, Optional
from ..langgraph_setup import PremiumAgentState, LangGraphSetup
from ..gemini_service import GeminiService
from ..model_cascader import ModelCascader
from ..core_api_client import CoreAPIClient
from .expert_agents import (
    ExplanationAgent, AssessmentAgent, ContentCuratorAgent,
    LearningPlannerAgent, ResearchAgent
)
from datetime import datetime

class PremiumRoutingAgent:
    """LangGraph-powered routing agent for expert selection and orchestration"""
    
    def __init__(self):
        self.llm = GeminiService()
        self.model_cascader = ModelCascader()
        self.core_api_client = CoreAPIClient()
        self.langgraph_setup = LangGraphSetup()
        
        # Agent registry
        self.agent_registry = {
            'explainer': ExplanationAgent(),
            'assessor': AssessmentAgent(),
            'curator': ContentCuratorAgent(),
            'planner': LearningPlannerAgent(),
            'researcher': ResearchAgent()
        }
        
        # Create LangGraph workflow
        self.workflow = self.create_routing_graph()
    
    def create_routing_graph(self) -> StateGraph:
        """Create LangGraph workflow for agent routing"""
        workflow = StateGraph(PremiumAgentState)
        
        # Add nodes for each step
        workflow.add_node("route_query", self.route_query)
        workflow.add_node("assemble_context", self.assemble_context)
        workflow.add_node("explainer_agent", self.explainer_agent)
        workflow.add_node("assessor_agent", self.assessor_agent)
        workflow.add_node("curator_agent", self.curator_agent)
        workflow.add_node("planner_agent", self.planner_agent)
        workflow.add_node("researcher_agent", self.researcher_agent)
        workflow.add_node("synthesize_response", self.synthesize_response)
        
        # Define edges and conditional routing
        workflow.add_edge("route_query", "assemble_context")
        workflow.add_conditional_edges(
            "assemble_context",
            self.select_agents,
            {
                "explainer": "explainer_agent",
                "assessor": "assessor_agent", 
                "curator": "curator_agent",
                "planner": "planner_agent",
                "researcher": "researcher_agent",
                "multi": "explainer_agent"  # Start with explainer for multi-agent
            }
        )
        
        # Add edges to synthesis
        workflow.add_edge("explainer_agent", "synthesize_response")
        workflow.add_edge("assessor_agent", "synthesize_response")
        workflow.add_edge("curator_agent", "synthesize_response")
        workflow.add_edge("planner_agent", "synthesize_response")
        workflow.add_edge("researcher_agent", "synthesize_response")
        workflow.add_edge("synthesize_response", END)
        
        # Set entrypoint
        workflow.set_entry_point("route_query")
        
        return workflow.compile()
    
    async def route_query(self, state: PremiumAgentState) -> PremiumAgentState:
        """Route user query to determine required agents"""
        try:
            # Analyze query to determine required agents
            routing_prompt = self._create_routing_prompt(state["user_query"], state["user_context"])
            routing_response = await self.llm.generate(routing_prompt)
            
            # Parse routing decision
            selected_agents = self._parse_routing_response(routing_response)
            
            # Get user's learning analytics for expert selection
            user_analytics = await self.core_api_client.get_user_learning_analytics(state["user_id"])
            
            # Adjust expert selection based on user analytics
            adjusted_agents = self._adjust_agents_for_user(selected_agents, user_analytics)
            
            # Update state
            state["selected_agents"] = adjusted_agents
            state["workflow_status"] = "routed"
            state["metadata"]["routing_decision"] = routing_response
            state["metadata"]["selected_agents"] = adjusted_agents
            
            return state
            
        except Exception as e:
            print(f"Error in route_query: {e}")
            state["selected_agents"] = ["explainer"]
            state["workflow_status"] = "error"
            state["metadata"]["error"] = str(e)
            return state
    
    async def assemble_context(self, state: PremiumAgentState) -> PremiumAgentState:
        """Assemble user context and learning history"""
        try:
            user_id = state["user_id"]
            
            # Get user's learning context
            user_memory = await self.core_api_client.get_user_memory(user_id)
            learning_analytics = await self.core_api_client.get_user_learning_analytics(user_id)
            
            # Assemble context
            context = {
                "user_memory": user_memory,
                "learning_analytics": learning_analytics,
                "current_session": state.get("user_context", {})
            }
            
            state["assembled_context"] = context
            state["workflow_status"] = "context_assembled"
            
            return state
            
        except Exception as e:
            print(f"Error in assemble_context: {e}")
            state["assembled_context"] = {}
            state["workflow_status"] = "error"
            state["metadata"]["error"] = str(e)
            return state
    
    def select_agents(self, state: PremiumAgentState) -> str:
        """Select which agent to execute next"""
        selected_agents = state.get("selected_agents", [])
        
        if len(selected_agents) == 1:
            return selected_agents[0]
        elif len(selected_agents) > 1:
            return "multi"
        else:
            return "explainer"
    
    async def explainer_agent(self, state: PremiumAgentState) -> PremiumAgentState:
        """Execute explainer agent"""
        try:
            agent = self.agent_registry["explainer"]
            response = await agent.process_explanation_request(state)
            
            state["agent_responses"]["explainer"] = response
            state["workflow_status"] = "explainer_completed"
            
            return state
            
        except Exception as e:
            print(f"Error in explainer_agent: {e}")
            state["agent_responses"]["explainer"] = {"error": str(e)}
            state["workflow_status"] = "error"
            return state
    
    async def assessor_agent(self, state: PremiumAgentState) -> PremiumAgentState:
        """Execute assessor agent"""
        try:
            agent = self.agent_registry["assessor"]
            response = await agent.process_assessment_request(state)
            
            state["agent_responses"]["assessor"] = response
            state["workflow_status"] = "assessor_completed"
            
            return state
            
        except Exception as e:
            print(f"Error in assessor_agent: {e}")
            state["agent_responses"]["assessor"] = {"error": str(e)}
            state["workflow_status"] = "error"
            return state
    
    async def curator_agent(self, state: PremiumAgentState) -> PremiumAgentState:
        """Execute curator agent"""
        try:
            agent = self.agent_registry["curator"]
            response = await agent.process_curation_request(state)
            
            state["agent_responses"]["curator"] = response
            state["workflow_status"] = "curator_completed"
            
            return state
            
        except Exception as e:
            print(f"Error in curator_agent: {e}")
            state["agent_responses"]["curator"] = {"error": str(e)}
            state["workflow_status"] = "error"
            return state
    
    async def planner_agent(self, state: PremiumAgentState) -> PremiumAgentState:
        """Execute planner agent"""
        try:
            agent = self.agent_registry["planner"]
            response = await agent.process_planning_request(state)
            
            state["agent_responses"]["planner"] = response
            state["workflow_status"] = "planner_completed"
            
            return state
            
        except Exception as e:
            print(f"Error in planner_agent: {e}")
            state["agent_responses"]["planner"] = {"error": str(e)}
            state["workflow_status"] = "error"
            return state
    
    async def researcher_agent(self, state: PremiumAgentState) -> PremiumAgentState:
        """Execute researcher agent"""
        try:
            agent = self.agent_registry["researcher"]
            response = await agent.process_research_request(state)
            
            state["agent_responses"]["researcher"] = response
            state["workflow_status"] = "researcher_completed"
            
            return state
            
        except Exception as e:
            print(f"Error in researcher_agent: {e}")
            state["agent_responses"]["researcher"] = {"error": str(e)}
            state["workflow_status"] = "error"
            return state
    
    async def synthesize_response(self, state: PremiumAgentState) -> PremiumAgentState:
        """Synthesize responses from all agents"""
        try:
            # Get the final response using model cascading
            query = state["user_query"]
            user_tier = state.get("user_context", {}).get("user_tier", "standard")
            
            # Use model cascader for final synthesis
            cascaded_response = await self.model_cascader.select_and_execute(
                query=query,
                user_tier=user_tier,
                min_confidence=0.7
            )
            
            # Create synthesis prompt
            synthesis_prompt = self._create_synthesis_prompt(query, state["agent_responses"])
            
            # Generate final response using the cascaded model
            final_response = await self.llm.generate(synthesis_prompt)
            
            state["final_response"] = final_response
            state["workflow_status"] = "completed"
            state["metadata"]["model_used"] = cascaded_response.model_used
            state["metadata"]["confidence_score"] = cascaded_response.confidence
            state["metadata"]["estimated_cost"] = cascaded_response.cost_estimate
            state["metadata"]["token_count"] = len(final_response) // 4  # Rough estimation
            
            return state
            
        except Exception as e:
            print(f"Error in synthesize_response: {e}")
            state["final_response"] = "I apologize, but I encountered an error synthesizing the response."
            state["workflow_status"] = "error"
            state["metadata"]["error"] = str(e)
            return state
    
    def _create_routing_prompt(self, query: str, user_context: Dict[str, Any]) -> str:
        """Create prompt for routing decision"""
        return f"""
        Analyze the following user query and determine which expert agents should handle it:
        
        Query: {query}
        User Context: {user_context}
        
        Available Agents:
        - explainer: For explanations, clarifications, and step-by-step guidance
        - assessor: For assessments, quizzes, and knowledge testing
        - curator: For content curation, recommendations, and resource finding
        - planner: For learning path planning and study strategies
        - researcher: For in-depth research and complex problem solving
        
        Return a JSON response with:
        {{
            "agents": ["list", "of", "agent", "names"],
            "reasoning": "explanation of why these agents were chosen",
            "priority": "high/medium/low"
        }}
        """
    
    def _parse_routing_response(self, response: str) -> List[str]:
        """Parse routing response to extract agent names"""
        try:
            # Simple parsing - in production, use proper JSON parsing
            agents = []
            if "explainer" in response.lower():
                agents.append("explainer")
            if "assessor" in response.lower():
                agents.append("assessor")
            if "curator" in response.lower():
                agents.append("curator")
            if "planner" in response.lower():
                agents.append("planner")
            if "researcher" in response.lower():
                agents.append("researcher")
            
            return agents if agents else ["explainer"]
        except Exception as e:
            print(f"Error parsing routing response: {e}")
            return ["explainer"]
    
    def _adjust_agents_for_user(self, agents: List[str], user_analytics: Dict[str, Any]) -> List[str]:
        """Adjust agent selection based on user analytics"""
        try:
            learning_efficiency = user_analytics.get("learningEfficiency", 0.5)
            mastery_level = user_analytics.get("masteryLevel", "BEGINNER")
            
            # High efficiency users get more complex agent combinations
            if learning_efficiency > 0.8 and len(agents) == 1:
                if "explainer" in agents:
                    agents.append("researcher")
                elif "assessor" in agents:
                    agents.append("planner")
            
            # Advanced users get researcher for complex queries
            if mastery_level == "ADVANCED" and "explainer" in agents:
                agents.append("researcher")
            
            return agents
            
        except Exception as e:
            print(f"Error adjusting agents for user: {e}")
            return agents
    
    def _create_synthesis_prompt(self, query: str, agent_responses: Dict[str, Any]) -> str:
        """Create prompt for synthesizing multi-agent responses"""
        prompt = f"""
        Synthesize the following expert responses into a coherent answer:
        
        Query: {query}
        
        Expert Responses:
        """
        
        for agent_name, response in agent_responses.items():
            content = response.get("content", "No response")
            prompt += f"\n{agent_name.upper()}: {content}"
        
        prompt += "\n\nProvide a synthesized response that combines the best insights from all experts."
        return prompt
    
    async def execute_workflow(self, user_query: str, user_id: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete LangGraph workflow"""
        try:
            # Create initial state
            initial_state = self.langgraph_setup.create_base_state(user_query, user_id, user_context)
            
            # Execute workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            return {
                "response": final_state["final_response"],
                "agents_used": final_state["selected_agents"],
                "workflow_status": final_state["workflow_status"],
                "model_used": final_state["metadata"].get("model_used", "gemini-1.5-flash"),
                "confidence_score": final_state["metadata"].get("confidence_score", 0.8),
                "estimated_cost": final_state["metadata"].get("estimated_cost", 0.0),
                "token_count": final_state["metadata"].get("token_count", 0),
                "metadata": final_state["metadata"]
            }
            
        except Exception as e:
            print(f"Error executing workflow: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your request.",
                "agents_used": [],
                "workflow_status": "error",
                "model_used": "gemini-1.5-flash",
                "confidence_score": 0.0,
                "estimated_cost": 0.0,
                "token_count": 0,
                "metadata": {"error": str(e)}
            }
