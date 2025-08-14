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
from .model_cascader import ModelCascader, CostTracker

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
        self.core_api_client = CoreAPIClient()
        self.gemini_service = GeminiService()
        self.model_cascader = ModelCascader()
        self.cost_tracker = CostTracker()
        self.hybrid_retriever = HybridRetriever()
        self.cross_encoder = CrossEncoderReranker()
        self.sufficiency_classifier = SufficiencyClassifier()
    
    async def assemble_context(self, request: CAARequest) -> CAAResponse:
        """Execute 10-stage context assembly pipeline"""
        try:
            # Initialize state
            state = CAAState(
                request=request
            )
            
            # Stage 1: Input Normalization (No LLM needed)
            state = await self.normalize_input(state)
            
            # Stage 2: Query Augmentation (Use Flash Lite - routing task)
            state = await self.augment_query(state)
            
            # Stage 3: Coarse Retrieval (No LLM needed - vector search)
            state = await self.coarse_retrieval(state)
            
            # Stage 4: Graph Traversal (No LLM needed - graph algorithms)
            state = await self.graph_traversal(state)
            
            # Stage 5: Cross-Encoder Reranking (Use Flash Lite - classification task)
            state = await self.cross_encoder_reranking(state)
            
            # Stage 6: Sufficiency Checking (Use Flash Lite - classification task)
            state = await self.sufficiency_checking(state)
            
            # Stage 7: Context Condensation (Use Flash Lite - compression task)
            state = await self.compress_context(state, state.request.token_budget)
            
            # Stage 8: Tool Enrichment (Use Flash Lite - routing task)
            state = await self.tool_enrichment(state)
            
            # Stage 9: Final Assembly (Use Flash Lite - summarization task)
            state = await self.assemble_final_context(state)
            
            # Stage 10: Cache & Metrics (No LLM needed)
            state = await self.cache_and_metrics(state)
            
            # Generate response
            return CAAResponse.from_state(state)
            
        except Exception as e:
            print(f"Error in context assembly: {e}")
            raise e

    async def normalize_input(self, state: CAAState) -> CAAState:
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
            
            normalized = await self.gemini_service.generate(normalization_prompt, "flash_lite")
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

    async def augment_query(self, state: CAAState) -> CAAState:
        """Stage 2: Query Augmentation using cost-effective model"""
        try:
            # Use Flash Lite for query augmentation (routing task)
            model = await self.model_cascader.select_context_assembly_model(
                context_size=len(state.request.query),
                complexity="simple",
                task_type="routing"
            )
            
            prompt = f"""
            Augment this query with context from the user's session:
            Query: {state.request.query}
            Session Context: {state.request.session_context}
            Hints: {state.request.hints}
            
            Provide 3-5 augmented queries for better retrieval.
            """
            
            augmented_queries = await self.gemini_service.generate(prompt, model)
            state.augmented_queries = [augmented_queries]  # Simplified for now
            
            # Track cost
            cost_estimate = await self.gemini_service.get_cost_estimate(
                model, len(prompt), len(augmented_queries)
            )
            state.metrics["query_augmentation_cost"] = cost_estimate
            
            return state
            
        except Exception as e:
            print(f"Error in query augmentation: {e}")
            state.augmented_queries = [state.request.query]  # Fallback to original
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
    
    async def cross_encoder_reranking(self, state: CAAState) -> CAAState:
        """Stage 5: Cross-Encoder Reranking using cost-effective model"""
        try:
            # Use Flash Lite for reranking (classification task)
            model = await self.model_cascader.select_context_assembly_model(
                context_size=len(str(state.retrieved_chunks)),
                complexity="simple",
                task_type="classification"
            )
            
            # Rerank chunks using cross-encoder
            reranked_chunks = await self.cross_encoder.rerank_chunks(
                state.retrieved_chunks, state.normalized_query, state.request.mode
            )
            
            # Use Flash Lite for additional relevance scoring
            for chunk in reranked_chunks:
                relevance_prompt = f"""
                Rate the relevance of this content to the query (0-1):
                Query: {state.normalized_query}
                Content: {chunk.get('content', '')[:500]}
                
                Return only a number between 0 and 1.
                """
                
                relevance_score = await self.gemini_service.generate(relevance_prompt, model)
                try:
                    chunk["llm_relevance_score"] = float(relevance_score.strip())
                except:
                    chunk["llm_relevance_score"] = chunk.get("rerank_score", 0.5)
            
            state.reranked_chunks = reranked_chunks
            
            # Track cost
            total_cost = 0
            for chunk in reranked_chunks:
                cost_estimate = await self.gemini_service.get_cost_estimate(
                    model, len(relevance_prompt), len(relevance_score)
                )
                total_cost += cost_estimate.get("total_cost", 0)
            
            state.metrics["reranking_cost"] = total_cost
            
            return state
            
        except Exception as e:
            print(f"Error in cross-encoder reranking: {e}")
            return state

    async def sufficiency_checking(self, state: CAAState) -> CAAState:
        """Stage 6: Sufficiency Checking using cost-effective model"""
        try:
            # Use Flash Lite for sufficiency checking (classification task)
            model = await self.model_cascader.select_context_assembly_model(
                context_size=len(str(state.reranked_chunks)),
                complexity="simple",
                task_type="classification"
            )
            
            # Check if context is sufficient
            sufficiency_result = await self.sufficiency_classifier.check_sufficiency(
                state.reranked_chunks, state.normalized_query
            )
            
            # Use Flash Lite for additional sufficiency analysis
            sufficiency_prompt = f"""
            Analyze if this context can answer the query:
            Query: {state.normalized_query}
            Context: {str(state.reranked_chunks)[:1000]}
            
            Return: SUFFICIENT or INSUFFICIENT
            """
            
            sufficiency_analysis = await self.gemini_service.generate(sufficiency_prompt, model)
            state.sufficiency_score = 0.8 if "SUFFICIENT" in sufficiency_analysis.upper() else 0.3
            
            # Track cost
            cost_estimate = await self.gemini_service.get_cost_estimate(
                model, len(sufficiency_prompt), len(sufficiency_analysis)
            )
            state.metrics["sufficiency_checking_cost"] = cost_estimate
            
            return state
            
        except Exception as e:
            print(f"Error in sufficiency checking: {e}")
            state.sufficiency_score = 0.5  # Default score
            return state

    async def compress_context(self, state: CAAState, target_tokens: int = None) -> CAAState:
        """Stage 7: Context Condensation using cost-effective model"""
        try:
            if target_tokens is None:
                target_tokens = state.request.token_budget // 2
            
            # Use Flash Lite for context compression (compression task)
            model = await self.model_cascader.select_context_assembly_model(
                context_size=len(str(state.reranked_chunks)),
                complexity="medium",
                task_type="compression"
            )
            
            # Compress context while preserving key information
            compression_prompt = f"""
            Compress this context to approximately {target_tokens} tokens while preserving:
            - Key facts and information
            - Source citations
            - Important relationships
            
            Context: {str(state.reranked_chunks)[:2000]}
            """
            
            condensed_context = await self.gemini_service.generate(compression_prompt, model)
            state.condensed_context = condensed_context
            
            # Track cost
            cost_estimate = await self.gemini_service.get_cost_estimate(
                model, len(compression_prompt), len(condensed_context)
            )
            state.metrics["context_compression_cost"] = cost_estimate
            
            return state
            
        except Exception as e:
            print(f"Error in context compression: {e}")
            state.condensed_context = "Context compression failed"
            return state
    
    async def tool_enrichment(self, state: CAAState) -> CAAState:
        """Stage 8: Tool Enrichment using cost-effective model"""
        try:
            # Select tools (No LLM needed - keyword detection)
            tools = await self.select_tools(state.normalized_query)
            
            # Execute tools (No LLM needed - tool execution)
            tool_outputs = await self.execute_tools(tools, state.normalized_query)
            
            # Use Flash Lite for tool output enhancement (routing task)
            model = await self.model_cascader.select_context_assembly_model(
                context_size=len(str(tool_outputs)),
                complexity="simple",
                task_type="routing"
            )
            
            # Enhance tool outputs with context
            enhanced_outputs = []
            for output in tool_outputs:
                enhancement_prompt = f"""
                Enhance this tool output with context relevance:
                Query: {state.normalized_query}
                Tool: {output.get('tool', '')}
                Output: {output.get('output', '')}
                
                Provide a brief enhancement explaining relevance.
                """
                
                enhancement = await self.gemini_service.generate(enhancement_prompt, model)
                enhanced_outputs.append({
                    **output,
                    "enhancement": enhancement
                })
            
            state.tool_outputs = enhanced_outputs
            
            # Track cost
            total_cost = 0
            for output in enhanced_outputs:
                cost_estimate = await self.gemini_service.get_cost_estimate(
                    model, len(enhancement_prompt), len(enhancement)
                )
                total_cost += cost_estimate.get("total_cost", 0)
            
            state.metrics["tool_enrichment_cost"] = total_cost
            
            return state
            
        except Exception as e:
            print(f"Error in tool enrichment: {e}")
            state.tool_outputs = []
            return state
    
    async def assemble_final_context(self, state: CAAState) -> CAAState:
        """Stage 9: Final Assembly using cost-effective model"""
        try:
            # Use Flash Lite for final assembly (summarization task)
            model = await self.model_cascader.select_context_assembly_model(
                context_size=len(state.condensed_context) + len(str(state.tool_outputs)),
                complexity="medium",
                task_type="summarization"
            )
            
            # Assemble final context
            assembly_prompt = f"""
            Assemble a comprehensive context from these components:
            
            Condensed Context: {state.condensed_context}
            Tool Outputs: {str(state.tool_outputs)}
            Knowledge Primitives: {str(state.graph_results)}
            
            Create a well-structured, coherent context that answers the query:
            {state.normalized_query}
            """
            
            final_context = await self.gemini_service.generate(assembly_prompt, model)
            state.final_context = final_context
            
            # Track cost
            cost_estimate = await self.gemini_service.get_cost_estimate(
                model, len(assembly_prompt), len(final_context)
            )
            state.metrics["final_assembly_cost"] = cost_estimate
            
            return state
            
        except Exception as e:
            print(f"Error in final assembly: {e}")
            state.final_context = state.condensed_context
            return state
    
    async def cache_and_metrics(self, state: CAAState) -> CAAState:
        """Stage 10: Cache & Metrics (No LLM needed)"""
        try:
            # Generate cache key
            state.cache_key = self.generate_cache_key(CAARequest(
                query=state.request.query,
                user_id=state.request.user_id,
                mode=state.request.mode,
                session_context=state.request.session_context,
                hints=state.request.hints,
                token_budget=state.request.token_budget,
                latency_budget_ms=state.request.latency_budget_ms
            ))
            
            # Calculate total cost
            total_cost = sum([
                state.metrics.get("query_augmentation_cost", {}).get("total_cost", 0),
                state.metrics.get("reranking_cost", 0),
                state.metrics.get("sufficiency_checking_cost", {}).get("total_cost", 0),
                state.metrics.get("context_compression_cost", {}).get("total_cost", 0),
                state.metrics.get("tool_enrichment_cost", 0),
                state.metrics.get("final_assembly_cost", {}).get("total_cost", 0)
            ])
            
            state.metrics["total_cost"] = total_cost
            state.metrics["cost_breakdown"] = {
                "query_augmentation": state.metrics.get("query_augmentation_cost", {}).get("total_cost", 0),
                "reranking": state.metrics.get("reranking_cost", 0),
                "sufficiency_checking": state.metrics.get("sufficiency_checking_cost", {}).get("total_cost", 0),
                "context_compression": state.metrics.get("context_compression_cost", {}).get("total_cost", 0),
                "tool_enrichment": state.metrics.get("tool_enrichment_cost", 0),
                "final_assembly": state.metrics.get("final_assembly_cost", {}).get("total_cost", 0)
            }
            
            return state
            
        except Exception as e:
            print(f"Error in cache and metrics: {e}")
            return state
    
    async def compress_chunks(self, chunks: List[Dict[str, Any]], target_tokens: int) -> str:
        """Compress context chunks to target token count"""
        try:
            # Simple compression - in production, use more sophisticated methods
            combined_text = " ".join([chunk.get("content", "") for chunk in chunks])
            
            if len(combined_text.split()) <= target_tokens:
                return combined_text
            
            # Truncate to target tokens
            words = combined_text.split()
            return " ".join(words[:target_tokens])
            
        except Exception as e:
            print(f"Error compressing chunks: {e}")
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
        """Execute selected tools using real implementations"""
        from .tools import ToolsIntegrationService, ToolExecutionRequest
        
        outputs = []
        tools_service = ToolsIntegrationService()
        
        # Create tool execution requests
        tool_requests = []
        for tool in tools:
            # Determine tool parameters based on query analysis
            parameters = self._determine_tool_parameters(tool, query)
            
            request = ToolExecutionRequest(
                tool_name=tool,
                parameters=parameters,
                user_id="user_123",  # Would come from actual user context
                context={"query": query, "mode": "premium"}
            )
            tool_requests.append(request)
        
        # Execute tools concurrently
        try:
            results = await tools_service.execute_multiple_tools(tool_requests)
            
            for result in results:
                if result.success:
                    outputs.append({
                        "tool": result.tool_name,
                        "output": self._format_tool_output(result),
                        "execution_time": result.execution_time,
                        "metadata": result.metadata
                    })
                else:
                    outputs.append({
                        "tool": result.tool_name,
                        "error": result.error,
                        "execution_time": result.execution_time
                    })
        except Exception as e:
            # Fallback to mock responses if real tools fail
            for tool in tools:
                outputs.append({
                    "tool": tool,
                    "output": f"Real tool execution failed, using fallback: {str(e)}",
                    "fallback": True
                })
        
        return outputs
    
    def _determine_tool_parameters(self, tool: str, query: str) -> Dict[str, Any]:
        """Determine appropriate parameters for tool execution based on query"""
        parameters = {}
        
        if tool == "calculator":
            # Extract mathematical expressions from query
            import re
            math_patterns = [
                r'(\d+[\+\-\*/]\d+)',  # Basic arithmetic
                r'(sin|cos|tan|log|sqrt)\s*\([^)]+\)',  # Functions
                r'(\d+\s*[a-zA-Z]+\s*to\s*[a-zA-Z]+)',  # Unit conversions
            ]
            
            for pattern in math_patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    parameters["expression"] = matches[0]
                    parameters["operation"] = "evaluate"
                    break
            
            if not parameters:
                parameters = {"expression": "2 + 2", "operation": "evaluate"}
        
        elif tool == "code_executor":
            # Extract code snippets from query
            code_patterns = [
                r'```(\w+)\n(.*?)\n```',  # Code blocks
                r'def\s+\w+\s*\([^)]*\):',  # Function definitions
                r'print\s*\([^)]*\)',  # Print statements
            ]
            
            for pattern in code_patterns:
                matches = re.findall(pattern, query, re.DOTALL)
                if matches:
                    if len(matches[0]) == 2:  # Code block with language
                        parameters["language"] = matches[0][0]
                        parameters["code"] = matches[0][1]
                    else:  # Single code snippet
                        parameters["code"] = matches[0]
                        parameters["language"] = "python"
                    break
            
            if not parameters:
                parameters = {"code": "print('Hello, World!')", "language": "python"}
        
        elif tool == "web_search":
            # Extract search queries
            if "weather" in query.lower():
                parameters = {"query": "current weather", "search_type": "basic"}
            elif "news" in query.lower():
                parameters = {"query": "latest news", "search_type": "basic"}
            elif "stock" in query.lower() or "price" in query.lower():
                parameters = {"query": "stock market prices", "search_type": "basic"}
            else:
                # Use the query itself for search
                parameters = {"query": query[:100], "search_type": "basic"}
        
        elif tool == "diagram_generator":
            # Determine diagram type and create sample data
            if "flow" in query.lower() or "process" in query.lower():
                parameters = {
                    "diagram_type": "flowchart",
                    "data": {
                        "nodes": [
                            {"id": "start", "label": "Start", "type": "start"},
                            {"id": "process", "label": "Process", "type": "default"},
                            {"id": "end", "label": "End", "type": "end"}
                        ],
                        "edges": [
                            {"from": "start", "to": "process"},
                            {"from": "process", "to": "end"}
                        ]
                    }
                }
            elif "mind" in query.lower() or "concept" in query.lower():
                parameters = {
                    "diagram_type": "mindmap",
                    "data": {
                        "nodes": [
                            {"id": "root", "label": "Main Concept", "type": "root"},
                            {"id": "branch1", "label": "Branch 1", "type": "branch"},
                            {"id": "branch2", "label": "Branch 2", "type": "branch"}
                        ]
                    }
                }
            else:
                parameters = {
                    "diagram_type": "flowchart",
                    "data": {"nodes": [], "edges": []}
                }
        
        elif tool == "example_generator":
            # Extract concept for example generation
            parameters = {
                "concept": query[:50],  # Use first 50 chars as concept
                "example_type": "code",
                "user_level": "intermediate",
                "learning_style": "visual"
            }
        
        return parameters
    
    def _format_tool_output(self, result) -> str:
        """Format tool output for display"""
        if result.tool_name == "calculator":
            if hasattr(result.result, 'result'):
                return f"Calculation: {result.result.result}"
            return str(result.result)
        
        elif result.tool_name == "code_executor":
            if hasattr(result.result, 'output'):
                return f"Code Output: {result.result.output}"
            return str(result.result)
        
        elif result.tool_name == "web_search":
            if hasattr(result.result, 'results') and result.result.results:
                first_result = result.result.results[0]
                return f"Search Result: {first_result.title} - {first_result.content[:100]}..."
            return str(result.result)
        
        elif result.tool_name == "diagram_generator":
            if hasattr(result.result, 'content'):
                return f"Diagram Generated: {result.result.diagram_type} ({result.result.format})"
            return str(result.result)
        
        elif result.tool_name == "example_generator":
            if hasattr(result.result, 'examples') and result.result.examples:
                return f"Examples Generated: {len(result.result.examples)} examples for {result.result.concept}"
            return str(result.result)
        
        return str(result.result)
    
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
