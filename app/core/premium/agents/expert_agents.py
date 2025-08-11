"""
LangGraph-enhanced expert agents with Core API integration.
Provides specialized agents with tools and Core API data access.
"""

from langchain.tools import tool
from typing import Dict, Any, List, Optional
from ..langgraph_setup import PremiumAgentState
from ..gemini_service import GeminiService
from ..core_api_client import CoreAPIClient

class ExplanationAgent:
    """LangGraph-enhanced explanation agent with Core API integration"""
    
    def __init__(self):
        self.llm = GeminiService()
        self.core_api_client = CoreAPIClient()
        self.tools = [
            self.generate_diagram,
            self.create_interactive_simulation,
            self.find_analogies,
            self.generate_code_examples,
            self.get_user_learning_context,
            self.get_knowledge_primitives,
            self.create_learning_path_step
        ]
    
    @tool
    async def generate_diagram(self, concept: str, user_id: str) -> str:
        """Generate visual diagram for concept explanation using user's learning context"""
        try:
            # Get user's learning analytics from Core API
            analytics = await self.core_api_client.get_user_learning_analytics(user_id)
            learning_efficiency = analytics.get("learningEfficiency", 0.5)
            
            return f"Diagram for {concept} tailored to user's learning efficiency: {learning_efficiency}"
        except Exception as e:
            return f"Diagram for {concept} (fallback mode)"
    
    @tool
    async def create_interactive_simulation(self, concept: str, user_id: str) -> str:
        """Create interactive simulation for complex concepts based on user's cognitive profile"""
        try:
            # Get user's cognitive profile from Core API
            user_memory = await self.core_api_client.get_user_memory(user_id)
            cognitive_approach = user_memory.get("cognitiveApproach", "BALANCED")
            
            return f"Simulation for {concept} optimized for {cognitive_approach} approach"
        except Exception as e:
            return f"Simulation for {concept} (fallback mode)"
    
    @tool
    async def find_analogies(self, concept: str, user_id: str) -> str:
        """Find relevant analogies based on user's learning strengths"""
        try:
            # Get user's learning strengths from Core API
            user_memory = await self.core_api_client.get_user_memory(user_id)
            learning_strengths = user_memory.get("learningStrengths", ["visual"])
            
            return f"Analogies for {concept} leveraging strengths: {learning_strengths}"
        except Exception as e:
            return f"Analogies for {concept} (fallback mode)"
    
    @tool
    async def generate_code_examples(self, concept: str, user_id: str) -> str:
        """Generate code examples for programming concepts"""
        return f"Code examples for {concept} with step-by-step explanations"
    
    async def get_user_learning_context(self, user_id: str) -> dict:
        """Get comprehensive user learning context from Core API"""
        try:
            analytics = await self.core_api_client.get_user_learning_analytics(user_id)
            memory_insights = await self.core_api_client.get_user_memory_insights(user_id)
            learning_paths = await self.core_api_client.get_user_learning_paths(user_id)
            
            return {
                "analytics": analytics,
                "insights": memory_insights,
                "learning_paths": learning_paths
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def get_knowledge_primitives(self, concept: str, user_id: str) -> list:
        """Get knowledge primitives with premium fields from Core API"""
        try:
            primitives = await self.core_api_client.get_knowledge_primitives(
                user_id=user_id,
                concept=concept,
                include_premium_fields=True  # complexityScore, isCoreConcept, etc.
            )
            return primitives
        except Exception as e:
            return []
    
    async def create_learning_path_step(self, primitive_id: str, user_id: str) -> dict:
        """Create a learning path step using Core API"""
        try:
            step = await self.core_api_client.create_learning_path_step(
                user_id=user_id,
                primitive_id=primitive_id
            )
            return step
        except Exception as e:
            return {"error": str(e)}
    
    async def process_explanation_request(self, state: PremiumAgentState) -> Dict[str, Any]:
        """Process explanation request using LangGraph tools and Core API data"""
        try:
            # Get user context from Core API
            user_context = await self.get_user_learning_context(state["user_id"])
            state["user_context"].update(user_context)
            
            # Create explanation prompt with user context
            prompt = self._create_explanation_prompt(state)
            response = await self.llm.generate(prompt)
            
            return {
                "content": response,
                "confidence": 0.8,
                "agent_type": "explainer",
                "tools_used": ["user_learning_context", "knowledge_primitives"],
                "metadata": {
                    "user_context": user_context,
                    "explanation_style": "step_by_step"
                }
            }
        except Exception as e:
            return {
                "content": "I apologize, but I encountered an error providing an explanation.",
                "confidence": 0.0,
                "agent_type": "explainer",
                "error": str(e)
            }
    
    def _create_explanation_prompt(self, state: PremiumAgentState) -> str:
        """Create explanation prompt with user context"""
        user_context = state.get("user_context", {})
        query = state.get("user_query", "")
        
        return f"""
        Provide a clear, step-by-step explanation for: {query}
        
        User Context:
        - Learning Style: {user_context.get('learningStyle', 'VISUAL')}
        - Cognitive Approach: {user_context.get('cognitiveApproach', 'BALANCED')}
        - Preferred Explanation Style: {user_context.get('preferredExplanationStyle', 'STEP_BY_STEP')}
        
        Please provide an explanation that is:
        1. Tailored to the user's learning style
        2. Clear and easy to follow
        3. Engaging and educational
        4. Based on the latest research and best practices
        """

class AssessmentAgent:
    """LangGraph-enhanced assessment agent with adaptive question generation"""
    
    def __init__(self):
        self.llm = GeminiService()
        self.core_api_client = CoreAPIClient()
    
    async def process_assessment_request(self, state: PremiumAgentState) -> Dict[str, Any]:
        """Process assessment request with user context"""
        try:
            # Get user's learning analytics for adaptive assessment
            analytics = await self.core_api_client.get_user_learning_analytics(state["user_id"])
            mastery_level = analytics.get("masteryLevel", "BEGINNER")
            
            # Create adaptive assessment
            assessment_prompt = self._create_assessment_prompt(state, mastery_level)
            response = await self.llm.generate(assessment_prompt)
            
            return {
                "content": response,
                "confidence": 0.7,
                "agent_type": "assessor",
                "metadata": {
                    "mastery_level": mastery_level,
                    "assessment_type": "adaptive",
                    "difficulty_level": "auto_determined"
                }
            }
        except Exception as e:
            return {
                "content": "I apologize, but I encountered an error creating an assessment.",
                "confidence": 0.0,
                "agent_type": "assessor",
                "error": str(e)
            }
    
    def _create_assessment_prompt(self, state: PremiumAgentState, mastery_level: str) -> str:
        """Create assessment prompt based on user mastery level"""
        query = state.get("user_query", "")
        
        return f"""
        Create an assessment for: {query}
        
        User Mastery Level: {mastery_level}
        
        Please create:
        1. Multiple choice questions appropriate for {mastery_level} level
        2. Short answer questions to test understanding
        3. Practical exercises if applicable
        4. Clear explanations for correct answers
        
        Make the assessment engaging and educational.
        """

class ContentCuratorAgent:
    """LangGraph-enhanced curator agent with resource discovery"""
    
    def __init__(self):
        self.llm = GeminiService()
        self.core_api_client = CoreAPIClient()
    
    async def process_curation_request(self, state: PremiumAgentState) -> Dict[str, Any]:
        """Process content curation request"""
        try:
            # Get user's learning preferences
            user_memory = await self.core_api_client.get_user_memory(state["user_id"])
            learning_style = user_memory.get("learningStyle", "VISUAL")
            
            # Create curation response
            curation_prompt = self._create_curation_prompt(state, learning_style)
            response = await self.llm.generate(curation_prompt)
            
            return {
                "content": response,
                "confidence": 0.75,
                "agent_type": "curator",
                "metadata": {
                    "learning_style": learning_style,
                    "curation_style": "personalized",
                    "resource_types": "mixed"
                }
            }
        except Exception as e:
            return {
                "content": "I apologize, but I encountered an error curating content.",
                "confidence": 0.0,
                "agent_type": "curator",
                "error": str(e)
            }
    
    def _create_curation_prompt(self, state: PremiumAgentState, learning_style: str) -> str:
        """Create curation prompt based on learning style"""
        query = state.get("user_query", "")
        
        return f"""
        Curate learning resources for: {query}
        
        User Learning Style: {learning_style}
        
        Please recommend:
        1. High-quality learning resources
        2. Interactive materials if applicable
        3. Practice exercises
        4. Additional reading materials
        5. Tools and applications
        
        Focus on resources that match the user's {learning_style} learning style.
        """

class LearningPlannerAgent:
    """LangGraph-enhanced planner agent with learning path optimization"""
    
    def __init__(self):
        self.llm = GeminiService()
        self.core_api_client = CoreAPIClient()
    
    async def process_planning_request(self, state: PremiumAgentState) -> Dict[str, Any]:
        """Process learning planning request"""
        try:
            # Get user's learning analytics and paths
            analytics = await self.core_api_client.get_user_learning_analytics(state["user_id"])
            learning_paths = await self.core_api_client.get_user_learning_paths(state["user_id"])
            
            # Create learning plan
            planning_prompt = self._create_planning_prompt(state, analytics, learning_paths)
            response = await self.llm.generate(planning_prompt)
            
            return {
                "content": response,
                "confidence": 0.8,
                "agent_type": "planner",
                "metadata": {
                    "learning_efficiency": analytics.get("learningEfficiency", 0.5),
                    "existing_paths": len(learning_paths),
                    "planning_style": "adaptive",
                    "timeframe": "flexible"
                }
            }
        except Exception as e:
            return {
                "content": "I apologize, but I encountered an error creating a learning plan.",
                "confidence": 0.0,
                "agent_type": "planner",
                "error": str(e)
            }
    
    def _create_planning_prompt(self, state: PremiumAgentState, analytics: Dict, learning_paths: List) -> str:
        """Create planning prompt with user analytics"""
        query = state.get("user_query", "")
        efficiency = analytics.get("learningEfficiency", 0.5)
        
        return f"""
        Create a learning plan for: {query}
        
        User Learning Efficiency: {efficiency}
        Existing Learning Paths: {len(learning_paths)}
        
        Please create:
        1. A structured learning path
        2. Milestones and checkpoints
        3. Time estimates for each step
        4. Practice recommendations
        5. Progress tracking methods
        
        Optimize for the user's learning efficiency and existing paths.
        """

class ResearchAgent:
    """LangGraph-enhanced research agent with academic search capabilities"""
    
    def __init__(self):
        self.llm = GeminiService()
        self.core_api_client = CoreAPIClient()
    
    async def process_research_request(self, state: PremiumAgentState) -> Dict[str, Any]:
        """Process research request with in-depth analysis"""
        try:
            # Get user's research preferences
            user_memory = await self.core_api_client.get_user_memory(state["user_id"])
            cognitive_approach = user_memory.get("cognitiveApproach", "BALANCED")
            
            # Create research response
            research_prompt = self._create_research_prompt(state, cognitive_approach)
            response = await self.llm.generate(research_prompt)
            
            return {
                "content": response,
                "confidence": 0.85,
                "agent_type": "researcher",
                "metadata": {
                    "cognitive_approach": cognitive_approach,
                    "research_depth": "comprehensive",
                    "analysis_style": "detailed"
                }
            }
        except Exception as e:
            return {
                "content": "I apologize, but I encountered an error conducting research.",
                "confidence": 0.0,
                "agent_type": "researcher",
                "error": str(e)
            }
    
    def _create_research_prompt(self, state: PremiumAgentState, cognitive_approach: str) -> str:
        """Create research prompt with cognitive approach"""
        query = state.get("user_query", "")
        
        return f"""
        Conduct comprehensive research on: {query}
        
        User Cognitive Approach: {cognitive_approach}
        
        Please provide:
        1. In-depth analysis of the topic
        2. Current research and developments
        3. Different perspectives and approaches
        4. Practical applications and implications
        5. Future directions and trends
        
        Tailor the research depth to the user's {cognitive_approach} cognitive approach.
        """
