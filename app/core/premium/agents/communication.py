"""
Advanced agent communication protocols for LangGraph multi-agent system.
Provides agent-to-agent communication, coordination, and conflict resolution.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio

@dataclass
class Task:
    """Task definition for agent coordination"""
    id: str
    description: str
    priority: str
    assigned_agents: List[str]
    status: str
    created_at: datetime

@dataclass
class Context:
    """Context for agent handoffs"""
    task_id: str
    from_agent: str
    to_agent: str
    data: Dict[str, Any]
    handoff_reason: str

@dataclass
class CoordinatedResponse:
    """Response from coordinated agents"""
    primary_response: str
    supporting_responses: Dict[str, str]
    consensus_score: float
    conflicts_resolved: List[str]

@dataclass
class ParallelResults:
    """Results from parallel agent execution"""
    results: Dict[str, Any]
    execution_time: float
    success_rate: float

class MessageQueue:
    """Message queue for agent communication"""
    
    def __init__(self):
        self.messages = {}
        self.subscribers = {}
    
    async def publish(self, topic: str, message: Dict[str, Any]):
        """Publish message to topic"""
        if topic not in self.messages:
            self.messages[topic] = []
        
        message["timestamp"] = datetime.utcnow().isoformat()
        self.messages[topic].append(message)
        
        # Notify subscribers
        if topic in self.subscribers:
            for subscriber in self.subscribers[topic]:
                await subscriber(message)
    
    async def subscribe(self, topic: str, callback):
        """Subscribe to topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
    
    async def get_messages(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages for topic"""
        return self.messages.get(topic, [])[-limit:]

class CoordinationEngine:
    """Engine for coordinating multiple agents"""
    
    def __init__(self):
        self.active_tasks = {}
        self.agent_states = {}
        self.conflict_resolution_strategies = {
            "majority": self._resolve_by_majority,
            "consensus": self._resolve_by_consensus,
            "expert_opinion": self._resolve_by_expert_opinion
        }
    
    async def coordinate_agents(self, task: Task, agents: List[str]) -> CoordinatedResponse:
        """Coordinate multiple agents using LangGraph"""
        try:
            # Initialize task
            self.active_tasks[task.id] = task
            
            # Execute agents in parallel
            agent_tasks = []
            for agent_name in agents:
                task = asyncio.create_task(self._execute_agent(agent_name, task))
                agent_tasks.append(task)
            
            # Wait for all agents to complete
            results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            # Process results
            agent_responses = {}
            for i, result in enumerate(results):
                agent_name = agents[i]
                if isinstance(result, Exception):
                    agent_responses[agent_name] = {"error": str(result)}
                else:
                    agent_responses[agent_name] = result
            
            # Resolve conflicts and synthesize response
            coordinated_response = await self._synthesize_responses(agent_responses, task)
            
            return coordinated_response
            
        except Exception as e:
            print(f"Error coordinating agents: {e}")
            return CoordinatedResponse(
                primary_response="Error in agent coordination",
                supporting_responses={},
                consensus_score=0.0,
                conflicts_resolved=[]
            )
    
    async def handle_agent_handoff(self, from_agent: str, to_agent: str, context: Context):
        """Handle smooth transitions between agents"""
        try:
            # Log handoff
            print(f"Handoff from {from_agent} to {to_agent}: {context.handoff_reason}")
            
            # Transfer context data
            if to_agent in self.agent_states:
                self.agent_states[to_agent].update(context.data)
            else:
                self.agent_states[to_agent] = context.data
            
            # Notify receiving agent
            await self._notify_agent_handoff(to_agent, context)
            
        except Exception as e:
            print(f"Error in agent handoff: {e}")
    
    async def parallel_agent_execution(self, agents: List[str], task: Task) -> ParallelResults:
        """Execute multiple agents in parallel when appropriate"""
        try:
            start_time = datetime.utcnow()
            
            # Create parallel execution tasks
            execution_tasks = []
            for agent_name in agents:
                task_obj = asyncio.create_task(self._execute_agent(agent_name, task))
                execution_tasks.append(task_obj)
            
            # Execute in parallel
            results = await asyncio.gather(*execution_tasks, return_exceptions=True)
            
            # Process results
            successful_results = {}
            errors = 0
            
            for i, result in enumerate(results):
                agent_name = agents[i]
                if isinstance(result, Exception):
                    errors += 1
                    successful_results[agent_name] = {"error": str(result)}
                else:
                    successful_results[agent_name] = result
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            success_rate = (len(agents) - errors) / len(agents)
            
            return ParallelResults(
                results=successful_results,
                execution_time=execution_time,
                success_rate=success_rate
            )
            
        except Exception as e:
            print(f"Error in parallel execution: {e}")
            return ParallelResults(
                results={},
                execution_time=0.0,
                success_rate=0.0
            )
    
    async def _execute_agent(self, agent_name: str, task: Task) -> Dict[str, Any]:
        """Execute a single agent"""
        try:
            # Simulate agent execution
            await asyncio.sleep(0.1)  # Simulate processing time
            
            return {
                "agent": agent_name,
                "task_id": task.id,
                "result": f"Result from {agent_name} for {task.description}",
                "confidence": 0.8,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "agent": agent_name,
                "task_id": task.id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _synthesize_responses(self, agent_responses: Dict[str, Any], task: Task) -> CoordinatedResponse:
        """Synthesize responses from multiple agents"""
        try:
            # Check for conflicts
            conflicts = self._identify_conflicts(agent_responses)
            
            # Resolve conflicts
            resolved_conflicts = []
            for conflict in conflicts:
                resolution = await self._resolve_conflict(conflict, agent_responses)
                resolved_conflicts.append(resolution)
            
            # Synthesize final response
            primary_response = self._create_primary_response(agent_responses)
            supporting_responses = self._extract_supporting_responses(agent_responses)
            
            # Calculate consensus score
            consensus_score = self._calculate_consensus_score(agent_responses)
            
            return CoordinatedResponse(
                primary_response=primary_response,
                supporting_responses=supporting_responses,
                consensus_score=consensus_score,
                conflicts_resolved=resolved_conflicts
            )
            
        except Exception as e:
            print(f"Error synthesizing responses: {e}")
            return CoordinatedResponse(
                primary_response="Error in response synthesis",
                supporting_responses={},
                consensus_score=0.0,
                conflicts_resolved=[]
            )
    
    def _identify_conflicts(self, agent_responses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify conflicts between agent responses"""
        conflicts = []
        
        # Simple conflict detection - in production, use more sophisticated analysis
        responses = list(agent_responses.values())
        if len(responses) > 1:
            # Check for contradictory responses
            for i, response1 in enumerate(responses):
                for j, response2 in enumerate(responses[i+1:], i+1):
                    if self._are_contradictory(response1, response2):
                        conflicts.append({
                            "agent1": list(agent_responses.keys())[i],
                            "agent2": list(agent_responses.keys())[j],
                            "conflict_type": "contradiction"
                        })
        
        return conflicts
    
    def _are_contradictory(self, response1: Dict[str, Any], response2: Dict[str, Any]) -> bool:
        """Check if two responses are contradictory"""
        # Simple contradiction detection
        content1 = response1.get("result", "").lower()
        content2 = response2.get("result", "").lower()
        
        # Check for opposite keywords
        positive_words = ["correct", "right", "yes", "true", "good"]
        negative_words = ["incorrect", "wrong", "no", "false", "bad"]
        
        has_positive1 = any(word in content1 for word in positive_words)
        has_negative2 = any(word in content2 for word in negative_words)
        has_positive2 = any(word in content2 for word in positive_words)
        has_negative1 = any(word in content1 for word in negative_words)
        
        return (has_positive1 and has_negative2) or (has_positive2 and has_negative1)
    
    async def _resolve_conflict(self, conflict: Dict[str, Any], agent_responses: Dict[str, Any]) -> str:
        """Resolve a conflict between agents"""
        try:
            # Use consensus strategy for conflict resolution
            strategy = "consensus"
            resolver = self.conflict_resolution_strategies.get(strategy, self._resolve_by_majority)
            
            resolution = await resolver(conflict, agent_responses)
            return f"Resolved {conflict['conflict_type']} between {conflict['agent1']} and {conflict['agent2']}: {resolution}"
            
        except Exception as e:
            return f"Failed to resolve conflict: {str(e)}"
    
    async def _resolve_by_majority(self, conflict: Dict[str, Any], agent_responses: Dict[str, Any]) -> str:
        """Resolve conflict by majority vote"""
        # Simple majority resolution
        return "Majority consensus applied"
    
    async def _resolve_by_consensus(self, conflict: Dict[str, Any], agent_responses: Dict[str, Any]) -> str:
        """Resolve conflict by building consensus"""
        # Consensus building
        return "Consensus building applied"
    
    async def _resolve_by_expert_opinion(self, conflict: Dict[str, Any], agent_responses: Dict[str, Any]) -> str:
        """Resolve conflict by expert opinion"""
        # Expert opinion resolution
        return "Expert opinion applied"
    
    def _create_primary_response(self, agent_responses: Dict[str, Any]) -> str:
        """Create primary response from agent responses"""
        if not agent_responses:
            return "No agent responses available"
        
        # Combine responses
        combined = []
        for agent_name, response in agent_responses.items():
            if "result" in response:
                combined.append(f"{agent_name}: {response['result']}")
        
        return " | ".join(combined) if combined else "No valid responses"
    
    def _extract_supporting_responses(self, agent_responses: Dict[str, Any]) -> Dict[str, str]:
        """Extract supporting responses from agents"""
        supporting = {}
        for agent_name, response in agent_responses.items():
            if "result" in response:
                supporting[agent_name] = response["result"]
        
        return supporting
    
    def _calculate_consensus_score(self, agent_responses: Dict[str, Any]) -> float:
        """Calculate consensus score among agents"""
        if not agent_responses:
            return 0.0
        
        # Calculate average confidence
        confidences = []
        for response in agent_responses.values():
            confidence = response.get("confidence", 0.0)
            confidences.append(confidence)
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    async def _notify_agent_handoff(self, agent_name: str, context: Context):
        """Notify agent of handoff"""
        # In production, this would notify the actual agent
        print(f"Notified {agent_name} of handoff with context: {context.handoff_reason}")

class AgentCommunicationProtocol:
    """Protocol for agent-to-agent communication"""
    
    def __init__(self):
        self.message_queue = MessageQueue()
        self.coordination_engine = CoordinationEngine()
    
    async def coordinate_agents(self, task: Task, agents: List[str]) -> CoordinatedResponse:
        """Coordinate multiple agents using LangGraph"""
        return await self.coordination_engine.coordinate_agents(task, agents)
    
    async def handle_agent_handoff(self, from_agent: str, to_agent: str, context: Context):
        """Handle smooth transitions between agents"""
        await self.coordination_engine.handle_agent_handoff(from_agent, to_agent, context)
    
    async def parallel_agent_execution(self, agents: List[str], task: Task) -> ParallelResults:
        """Execute multiple agents in parallel when appropriate"""
        return await self.coordination_engine.parallel_agent_execution(agents, task)
    
    async def broadcast_message(self, topic: str, message: Dict[str, Any]):
        """Broadcast message to all agents"""
        await self.message_queue.publish(topic, message)
    
    async def subscribe_to_messages(self, topic: str, callback):
        """Subscribe to agent messages"""
        await self.message_queue.subscribe(topic, callback)
