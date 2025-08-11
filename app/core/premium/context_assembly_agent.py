"""
Context Assembly Agent (CAA) - Core Pipeline Implementation.
Provides sophisticated 10-stage context assembly pipeline for premium users.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
# from ..langgraph_setup import PremiumAgentState  # Commented out as not used in CAA
from .core_api_client import CoreAPIClient
from .gemini_service import GeminiService

@dataclass
class CAARequest:
    """Request for Context Assembly Agent"""
    query: str
    user_id: str
    mode: str  # chat, quiz, deep_dive, walk_through, note_editing
    session_context: Dict[str, Any]
    hints: List[str]
    token_budget: int = 3000
    latency_budget_ms: int = 1200

@dataclass
class CAAState:
    """State for Context Assembly Agent pipeline"""
    request: CAARequest
    normalized_query: str = ""
    augmented_queries: List[str] = None
    retrieved_chunks: List[Dict[str, Any]] = None
    graph_results: List[Dict[str, Any]] = None
    reranked_chunks: List[Dict[str, Any]] = None
    sufficiency_score: float = 0.0
    condensed_context: str = ""
    tool_outputs: List[Dict[str, Any]] = None
    final_context: str = ""
    user_context: Dict[str, Any] = None
    cache_key: str = ""
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.augmented_queries is None:
            self.augmented_queries = []
        if self.retrieved_chunks is None:
            self.retrieved_chunks = []
        if self.graph_results is None:
            self.graph_results = []
        if self.reranked_chunks is None:
            self.reranked_chunks = []
        if self.tool_outputs is None:
            self.tool_outputs = []
        if self.user_context is None:
            self.user_context = {}
        if self.metrics is None:
            self.metrics = {}

@dataclass
class CAAResponse:
    """Response from Context Assembly Agent"""
    assembled_context: str
    short_context: str
    long_context: List[Dict[str, Any]]
    knowledge_primitives: List[Dict[str, Any]]
    examples: List[Dict[str, Any]]
    tool_outputs: List[Dict[str, Any]]
    sufficiency_score: float
    token_count: int
    rerank_scores: Dict[str, float]
    warnings: List[str]
    cache_key: str
    timestamp: datetime
    
    @classmethod
    def from_state(cls, state: CAAState) -> 'CAAResponse':
        """Create response from CAA state"""
        # Extract examples from tool outputs and context
        examples = []
        for tool_output in state.tool_outputs:
            if tool_output.get("tool") == "example_generator":
                examples.append({
                    "type": "practical_example",
                    "content": tool_output.get("output", ""),
                    "source": "example_generator"
                })
        
        # Extract examples from context chunks
        for chunk in state.reranked_chunks:
            content = chunk.get("content", "")
            if any(word in content.lower() for word in ["example", "instance", "case", "scenario"]):
                examples.append({
                    "type": "context_example",
                    "content": content,
                    "source": "context_chunk"
                })
        
        return cls(
            assembled_context=state.final_context,
            short_context=state.condensed_context,
            long_context=state.reranked_chunks,
            knowledge_primitives=state.graph_results,
            examples=examples,
            tool_outputs=state.tool_outputs,
            sufficiency_score=state.sufficiency_score,
            token_count=len(state.final_context.split()),
            rerank_scores=state.metrics.get("rerank_scores", {}),
            warnings=state.metrics.get("warnings", []),
            cache_key=state.cache_key,
            timestamp=datetime.utcnow()
        )

class ContextAssemblyAgent:
    """Context Assembly Agent with 10-stage pipeline"""
    
    def __init__(self):
        self.pipeline_stages = [
            self.input_normalization,
            self.query_augmentation,
            self.coarse_retrieval,
            self.graph_traversal,
            self.cross_encoder_rerank,
            self.sufficiency_check,
            self.context_condensation,
            self.tool_enrichment,
            self.final_assembly,
            self.cache_and_metrics
        ]
        self.core_api_client = CoreAPIClient()
        self.llm = GeminiService()
        self.hybrid_retriever = HybridRetriever()
        self.cross_encoder = CrossEncoderReranker()
        self.sufficiency_classifier = SufficiencyClassifier()
    
    async def assemble_context(self, request: CAARequest) -> CAAResponse:
        """Execute the full 10-stage context assembly pipeline with Core API data"""
        try:
            state = CAAState(request)
            
            # Enrich state with Core API user data
            user_analytics = await self.core_api_client.get_user_learning_analytics(request.user_id)
            memory_insights = await self.core_api_client.get_user_memory_insights(request.user_id)
            learning_paths = await self.core_api_client.get_user_learning_paths(request.user_id)
            
            state.user_context.update({
                "analytics": user_analytics,
                "insights": memory_insights,
                "learning_paths": learning_paths
            })
            
            # Execute pipeline stages
            for stage in self.pipeline_stages:
                state = await stage(state)
            
            return CAAResponse.from_state(state)
            
        except Exception as e:
            print(f"Error in context assembly: {e}")
            # Return fallback response
            return CAAResponse(
                assembled_context="Error assembling context",
                short_context="Error",
                long_context=[],
                knowledge_primitives=[],
                examples=[],
                tool_outputs=[],
                sufficiency_score=0.0,
                token_count=0,
                rerank_scores={},
                warnings=[f"Error: {str(e)}"],
                cache_key="",
                timestamp=datetime.utcnow()
            )
    
    async def input_normalization(self, state: CAAState) -> CAAState:
        """Normalize input using Core API user memory data"""
        try:
            user_memory = await self.core_api_client.get_user_memory(state.request.user_id)
            
            # Normalize query based on user's cognitive approach and learning preferences
            normalization_prompt = f"""
            Normalize the following query based on user preferences:
            
            Query: {state.request.query}
            User Memory: {user_memory}
            
            Please normalize the query to:
            1. Match the user's cognitive approach ({user_memory.get('cognitiveApproach', 'BALANCED')})
            2. Use their preferred explanation style ({user_memory.get('preferredExplanationStyle', 'STEP_BY_STEP')})
            3. Adapt to their learning style ({user_memory.get('learningStyle', 'VISUAL')})
            
            Return only the normalized query, not an explanation.
            """
            
            normalized = await self.llm.generate(normalization_prompt)
            # For testing, use the original query if normalization fails
            if "mock" in normalized.lower() or len(normalized) > 100:
                state.normalized_query = state.request.query
            else:
                state.normalized_query = normalized
            
            return state
            
        except Exception as e:
            print(f"Error in input normalization: {e}")
            state.normalized_query = state.request.query
            return state
    
    async def query_augmentation(self, state: CAAState) -> CAAState:
        """Augment query using Core API learning analytics"""
        try:
            analytics = state.user_context.get("analytics", {})
            
            # Augment query based on user's learning efficiency and focus areas
            augmentation_prompt = f"""
            Augment the following query based on user learning analytics:
            
            Query: {state.normalized_query}
            Learning Analytics: {analytics}
            
            Generate 3 augmented queries that:
            1. Address the user's focus areas: {analytics.get('focusAreas', [])}
            2. Match their learning efficiency: {analytics.get('learningEfficiency', 0.5)}
            3. Consider their mastery level: {analytics.get('masteryLevel', 'BEGINNER')}
            """
            
            augmented = await self.llm.generate(augmentation_prompt)
            state.augmented_queries = [state.normalized_query] + augmented.split('\n')[:2]
            
            return state
            
        except Exception as e:
            print(f"Error in query augmentation: {e}")
            state.augmented_queries = [state.normalized_query]
            return state
    
    async def coarse_retrieval(self, state: CAAState) -> CAAState:
        """Perform coarse retrieval using hybrid search"""
        try:
            # Use hybrid retriever for initial retrieval
            retrieved_chunks = await self.hybrid_retriever.retrieve(
                queries=state.augmented_queries,
                user_id=state.request.user_id,
                limit=20
            )
            
            state.retrieved_chunks = retrieved_chunks
            return state
            
        except Exception as e:
            print(f"Error in coarse retrieval: {e}")
            state.retrieved_chunks = []
            return state
    
    async def graph_traversal(self, state: CAAState) -> CAAState:
        """Traverse knowledge graph for related concepts"""
        try:
            # Get knowledge primitives from Core API
            primitives = await self.core_api_client.get_knowledge_primitives(
                blueprint_id="mock-blueprint",
                include_premium_fields=True
            )
            
            state.graph_results = primitives
            return state
            
        except Exception as e:
            print(f"Error in graph traversal: {e}")
            state.graph_results = []
            return state
    
    async def cross_encoder_rerank(self, state: CAAState) -> CAAState:
        """Rerank chunks using cross-encoder"""
        try:
            # Rerank chunks for relevance and suitability
            reranked_chunks = await self.cross_encoder.rerank_chunks(
                chunks=state.retrieved_chunks,
                query=state.normalized_query,
                mode=state.request.mode
            )
            
            state.reranked_chunks = reranked_chunks
            return state
            
        except Exception as e:
            print(f"Error in cross-encoder rerank: {e}")
            state.reranked_chunks = state.retrieved_chunks
            return state
    
    async def sufficiency_check(self, state: CAAState) -> CAAState:
        """Check if assembled context is sufficient"""
        try:
            # Check sufficiency of current context
            sufficiency_result = await self.sufficiency_classifier.check_sufficiency(
                context=state.reranked_chunks,
                query=state.normalized_query
            )
            
            state.sufficiency_score = sufficiency_result.get("score", 0.5)
            
            # If insufficient, expand context
            if sufficiency_result.get("score", 0.5) < 0.7:
                expanded_chunks = await self.sufficiency_classifier.iterative_expansion(
                    context=state.reranked_chunks,
                    query=state.normalized_query
                )
                state.reranked_chunks.extend(expanded_chunks)
            
            return state
            
        except Exception as e:
            print(f"Error in sufficiency check: {e}")
            state.sufficiency_score = 0.5
            return state
    
    async def context_condensation(self, state: CAAState) -> CAAState:
        """Condense context while preserving important information"""
        try:
            # Compress context to fit token budget
            compressed_context = await self.compress_context(
                chunks=state.reranked_chunks,
                target_tokens=state.request.token_budget // 2
            )
            
            state.condensed_context = compressed_context
            return state
            
        except Exception as e:
            print(f"Error in context condensation: {e}")
            state.condensed_context = "Context condensation failed"
            return state
    
    async def tool_enrichment(self, state: CAAState) -> CAAState:
        """Enrich context with tool outputs"""
        try:
            # Select and execute tools based on query
            selected_tools = await self.select_tools(state.normalized_query)
            tool_outputs = await self.execute_tools(selected_tools, state.normalized_query)
            
            state.tool_outputs = tool_outputs
            return state
            
        except Exception as e:
            print(f"Error in tool enrichment: {e}")
            state.tool_outputs = []
            return state
    
    async def final_assembly(self, state: CAAState) -> CAAState:
        """Assemble final context from all components"""
        try:
            # Combine condensed context, tool outputs, and knowledge primitives
            final_context = await self.assemble_final_context(
                condensed_context=state.condensed_context,
                tool_outputs=state.tool_outputs,
                knowledge_primitives=state.graph_results,
                mode=state.request.mode
            )
            
            state.final_context = final_context
            return state
            
        except Exception as e:
            print(f"Error in final assembly: {e}")
            state.final_context = state.condensed_context
            return state
    
    async def cache_and_metrics(self, state: CAAState) -> CAAState:
        """Cache results and collect metrics"""
        try:
            # Generate cache key
            state.cache_key = self.generate_cache_key(state.request)
            
            # Collect metrics
            state.metrics = {
                "pipeline_stages": len(self.pipeline_stages),
                "retrieved_chunks": len(state.retrieved_chunks),
                "reranked_chunks": len(state.reranked_chunks),
                "sufficiency_score": state.sufficiency_score,
                "tool_outputs": len(state.tool_outputs),
                "final_context_length": len(state.final_context)
            }
            
            return state
            
        except Exception as e:
            print(f"Error in cache and metrics: {e}")
            state.cache_key = ""
            state.metrics = {"error": str(e)}
            return state
    
    async def compress_context(self, chunks: List[Dict[str, Any]], target_tokens: int) -> str:
        """Compress context to target token count"""
        try:
            # Simple compression - in production, use more sophisticated methods
            combined_text = " ".join([chunk.get("content", "") for chunk in chunks])
            
            if len(combined_text.split()) <= target_tokens:
                return combined_text
            
            # Truncate to target tokens
            words = combined_text.split()
            return " ".join(words[:target_tokens])
            
        except Exception as e:
            print(f"Error compressing context: {e}")
            return "Context compression failed"
    
    async def select_tools(self, query: str) -> List[str]:
        """Select appropriate tools based on query"""
        # Enhanced tool selection with more comprehensive detection
        tools = []
        query_lower = query.lower()
        
        # Calculator tool - mathematical operations
        if any(word in query_lower for word in ["calculate", "math", "equation", "formula", "compute", "solve", "multiply", "divide", "add", "subtract"]):
            tools.append("calculator")
        
        # Code executor tool - programming and code
        if any(word in query_lower for word in ["code", "program", "script", "algorithm", "function", "class", "method", "variable", "loop", "condition", "syntax"]):
            tools.append("code_executor")
        
        # Web search tool - information lookup
        if any(word in query_lower for word in ["search", "find", "lookup", "research", "information", "data", "statistics", "latest", "current", "recent"]):
            tools.append("web_search")
        
        # Diagram generator tool - visual explanations
        if any(word in query_lower for word in ["diagram", "visual", "chart", "graph", "illustration", "draw", "show", "picture", "image"]):
            tools.append("diagram_generator")
        
        # Example generator tool - practical examples
        if any(word in query_lower for word in ["example", "instance", "case", "scenario", "demonstrate", "show how", "practical", "real-world"]):
            tools.append("example_generator")
        
        # For learning queries, always include example generator
        if any(word in query_lower for word in ["explain", "teach", "learn", "understand", "concept", "topic", "subject"]) and "example_generator" not in tools:
            tools.append("example_generator")
        
        return tools
    
    async def execute_tools(self, tools: List[str], query: str) -> List[Dict[str, Any]]:
        """Execute selected tools"""
        outputs = []
        
        for tool in tools:
            try:
                if tool == "calculator":
                    outputs.append({"tool": "calculator", "output": "Mock calculation result for mathematical operations"})
                elif tool == "code_executor":
                    outputs.append({"tool": "code_executor", "output": "Mock code execution result with syntax highlighting"})
                elif tool == "web_search":
                    outputs.append({"tool": "web_search", "output": "Mock search result with relevant information"})
                elif tool == "diagram_generator":
                    outputs.append({"tool": "diagram_generator", "output": "Mock diagram showing visual representation"})
                elif tool == "example_generator":
                    outputs.append({"tool": "example_generator", "output": "Mock practical example demonstrating the concept"})
                else:
                    outputs.append({"tool": tool, "output": f"Mock output for {tool}"})
            except Exception as e:
                outputs.append({"tool": tool, "error": str(e)})
        
        return outputs
    
    async def assemble_final_context(self, condensed_context: str, tool_outputs: List[Dict[str, Any]], 
                                   knowledge_primitives: List[Dict[str, Any]], mode: str) -> str:
        """Assemble final context from all components"""
        try:
            # Combine all components
            components = [condensed_context]
            
            # Add tool outputs
            for output in tool_outputs:
                if "output" in output:
                    components.append(f"Tool ({output['tool']}): {output['output']}")
            
            # Add knowledge primitives
            for primitive in knowledge_primitives:
                if "description" in primitive:
                    components.append(f"Knowledge: {primitive['description']}")
            
            return "\n\n".join(components)
            
        except Exception as e:
            print(f"Error assembling final context: {e}")
            return condensed_context
    
    def generate_cache_key(self, request: CAARequest) -> str:
        """Generate cache key for request"""
        import hashlib
        key_string = f"{request.query}:{request.user_id}:{request.mode}"
        return hashlib.md5(key_string.encode()).hexdigest()

class HybridRetriever:
    """Hybrid retriever for coarse retrieval"""
    
    async def retrieve(self, queries: List[str], user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve chunks using hybrid search"""
        # Mock implementation - in production, use actual vector + keyword search
        chunks = []
        for i, query in enumerate(queries):
            chunks.append({
                "id": f"chunk_{i}",
                "content": f"Mock content for query: {query}",
                "score": 0.8 - (i * 0.1),
                "source": "mock_source"
            })
        return chunks[:limit]

class CrossEncoderReranker:
    """Cross-encoder reranker for relevance scoring"""
    
    async def rerank_chunks(self, chunks: List[Dict[str, Any]], query: str, mode: str) -> List[Dict[str, Any]]:
        """Rerank chunks using cross-encoder"""
        # Mock implementation - in production, use actual cross-encoder model
        reranked = []
        for i, chunk in enumerate(chunks):
            # Simulate reranking scores
            relevance_score = 0.9 - (i * 0.05)
            mode_score = 0.8 if mode == "chat" else 0.7
            final_score = (relevance_score + mode_score) / 2
            
            chunk["rerank_score"] = final_score
            reranked.append(chunk)
        
        # Sort by rerank score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked

class SufficiencyClassifier:
    """Sufficiency classifier for context evaluation"""
    
    async def check_sufficiency(self, context: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Check if context is sufficient to answer query"""
        # Mock implementation - in production, use actual sufficiency model
        return {
            "score": 0.75,
            "is_sufficient": True,
            "missing_aspects": []
        }
    
    async def iterative_expansion(self, context: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Expand context if insufficient"""
        # Mock implementation - in production, use bridging retrieval
        return [
            {
                "id": "expanded_chunk",
                "content": f"Additional context for: {query}",
                "score": 0.6,
                "source": "expansion"
            }
        ]
