"""
CAA Integration service for premium users.
Enhances Context Assembly Agent with advanced RAG features.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .context_assembly_agent import ContextAssemblyAgent, CAARequest, CAAResponse
from .rag_fusion import RAGFusionService, FusedResults
from .search_optimization import SearchOptimizer, OptimizedResults
from .multimodal_rag import MultiModalRAG, MultiModalQuery, MultiModalResults
from .long_context_llm import LongContextLLM
from .core_api_client import CoreAPIClient

@dataclass
class EnhancedContext:
    """Enhanced context with advanced RAG features"""
    assembled_context: str
    rag_fusion_results: FusedResults
    optimized_results: OptimizedResults
    multimodal_results: Optional[MultiModalResults] = None
    long_context_response: Optional[str] = None
    enhancement_metrics: Dict[str, float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.enhancement_metrics is None:
            self.enhancement_metrics = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class OptimizedContext:
    """Optimized context with performance metrics"""
    context: EnhancedContext
    optimization_score: float
    performance_metrics: Dict[str, float]
    user_profile: Dict[str, Any]
    timestamp: datetime

class CAAIntegration:
    """CAA Integration service for premium users"""
    
    def __init__(self):
        self.caa_service = ContextAssemblyAgent()
        self.rag_fusion = RAGFusionService()
        self.search_optimizer = SearchOptimizer()
        self.multimodal_rag = MultiModalRAG()
        self.long_context_llm = LongContextLLM()
        self.core_api_client = CoreAPIClient()
    
    async def enhanced_context_assembly(self, query: str, user_id: str, mode: str = "chat") -> EnhancedContext:
        """Use CAA with enhanced RAG features for premium context assembly"""
        try:
            # Step 1: Standard CAA assembly
            caa_request = CAARequest(
                query=query,
                user_id=user_id,
                mode=mode,
                session_context={},
                hints=[],
                token_budget=5000,
                latency_budget_ms=2000
            )
            
            caa_response = await self.caa_service.assemble_context(caa_request)
            
            # Step 2: RAG-Fusion enhancement
            rag_fusion_results = await self.rag_fusion.multi_retrieve(query, user_id)
            
            # Step 3: Search optimization
            optimized_results = await self.search_optimizer.optimize_search_results(
                results=rag_fusion_results.fused_chunks,
                query=query,
                context={"user_id": user_id, "mode": mode}
            )
            
            # Step 4: Multi-modal enhancement (if applicable)
            multimodal_results = None
            if self._should_use_multimodal(query):
                multimodal_query = MultiModalQuery(text_query=query)
                multimodal_results = await self.multimodal_rag.retrieve_multimodal(multimodal_query)
            
            # Step 5: Long-context enhancement (if needed)
            long_context_response = None
            if len(caa_response.assembled_context) > 100000:  # Large context
                long_context_response = await self.long_context_llm.generate_with_full_context(
                    context=caa_response.assembled_context,
                    query=query
                )
            
            # Calculate enhancement metrics
            enhancement_metrics = {
                "caa_quality": caa_response.sufficiency_score,
                "rag_fusion_quality": rag_fusion_results.fusion_quality,
                "optimization_score": sum(optimized_results.optimization_metrics.values()) / len(optimized_results.optimization_metrics) if optimized_results.optimization_metrics else 0.0,
                "multimodal_score": sum(multimodal_results.fusion_scores.values()) / len(multimodal_results.fusion_scores) if multimodal_results and multimodal_results.fusion_scores else 0.0,
                "long_context_used": long_context_response is not None
            }
            
            return EnhancedContext(
                assembled_context=caa_response.assembled_context,
                rag_fusion_results=rag_fusion_results,
                optimized_results=optimized_results,
                multimodal_results=multimodal_results,
                long_context_response=long_context_response,
                enhancement_metrics=enhancement_metrics
            )
            
        except Exception as e:
            print(f"Error in enhanced context assembly: {e}")
            return EnhancedContext(
                assembled_context=f"Error assembling context: {str(e)}",
                rag_fusion_results=FusedResults(
                    fused_chunks=[],
                    strategy_scores={},
                    fusion_quality=0.0,
                    user_context={},
                    timestamp=datetime.utcnow()
                ),
                optimized_results=OptimizedResults(
                    results=[],
                    optimization_metrics={},
                    user_preferences={},
                    timestamp=datetime.utcnow()
                )
            )
    
    async def optimize_caa_pipeline(self, context: EnhancedContext, user_profile: Dict[str, Any]) -> OptimizedContext:
        """Optimize CAA output using advanced RAG features"""
        try:
            # Get user preferences from Core API
            user_id = user_profile.get("user_id", "")
            user_memory = await self.core_api_client.get_user_memory(user_id)
            analytics = await self.core_api_client.get_user_learning_analytics(user_id)
            
            # Calculate optimization score based on user profile
            optimization_score = 0.0
            
            # Consider learning efficiency
            efficiency = analytics.get("learningEfficiency", 0.5)
            optimization_score += efficiency * 0.3
            
            # Consider cognitive approach
            cognitive_approach = user_memory.get("cognitiveApproach", "BALANCED")
            if cognitive_approach == "TOP_DOWN" and context.enhancement_metrics.get("rag_fusion_quality", 0) > 0.8:
                optimization_score += 0.2
            elif cognitive_approach == "BOTTOM_UP" and context.enhancement_metrics.get("caa_quality", 0) > 0.8:
                optimization_score += 0.2
            
            # Consider multimodal usage
            if context.multimodal_results:
                optimization_score += 0.15
            
            # Consider long context usage
            if context.long_context_response:
                optimization_score += 0.15
            
            # Calculate performance metrics
            performance_metrics = {
                "context_assembly_time": 0.5,  # Mock value
                "rag_fusion_time": 0.3,        # Mock value
                "optimization_time": 0.2,       # Mock value
                "total_enhancement_time": 1.0,  # Mock value
                "memory_usage": 0.7,            # Mock value
                "cpu_usage": 0.6               # Mock value
            }
            
            return OptimizedContext(
                context=context,
                optimization_score=min(optimization_score, 1.0),
                performance_metrics=performance_metrics,
                user_profile=user_profile,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            print(f"Error optimizing CAA pipeline: {e}")
            return OptimizedContext(
                context=context,
                optimization_score=0.0,
                performance_metrics={},
                user_profile=user_profile,
                timestamp=datetime.utcnow()
            )
    
    def _should_use_multimodal(self, query: str) -> bool:
        """Determine if multi-modal retrieval should be used"""
        # Check for multi-modal keywords
        multimodal_keywords = [
            "image", "picture", "diagram", "chart", "graph", "visual",
            "audio", "sound", "voice", "speech",
            "code", "programming", "implementation", "example",
            "video", "animation", "interactive"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in multimodal_keywords)
    
    async def get_enhancement_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get recommendations for CAA enhancement based on user profile"""
        try:
            user_memory = await self.core_api_client.get_user_memory(user_id)
            analytics = await self.core_api_client.get_user_learning_analytics(user_id)
            
            recommendations = []
            
            # Check learning efficiency
            efficiency = analytics.get("learningEfficiency", 0.5)
            if efficiency < 0.6:
                recommendations.append({
                    "type": "optimization",
                    "title": "Enable Advanced RAG Features",
                    "description": "Your learning efficiency is below optimal. Enable advanced RAG features for better context assembly.",
                    "priority": "high"
                })
            
            # Check cognitive approach
            cognitive_approach = user_memory.get("cognitiveApproach", "BALANCED")
            if cognitive_approach == "TOP_DOWN":
                recommendations.append({
                    "type": "feature",
                    "title": "Enable Graph-Based Retrieval",
                    "description": "Your top-down learning approach would benefit from graph-based knowledge retrieval.",
                    "priority": "medium"
                })
            
            # Check learning style
            learning_style = user_memory.get("learningStyle", "VISUAL")
            if learning_style == "VISUAL":
                recommendations.append({
                    "type": "feature",
                    "title": "Enable Multi-Modal Retrieval",
                    "description": "Your visual learning style would benefit from image and diagram retrieval.",
                    "priority": "medium"
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting enhancement recommendations: {e}")
            return []
    
    async def monitor_caa_performance(self, user_id: str) -> Dict[str, Any]:
        """Monitor CAA performance for a user"""
        try:
            # Get user analytics
            analytics = await self.core_api_client.get_user_learning_analytics(user_id)
            
            # Mock performance metrics
            performance_metrics = {
                "context_assembly_success_rate": 0.95,
                "average_assembly_time": 1.2,  # seconds
                "rag_fusion_quality": 0.87,
                "optimization_effectiveness": 0.82,
                "user_satisfaction": analytics.get("learningEfficiency", 0.5),
                "context_relevance": 0.89,
                "context_completeness": 0.91
            }
            
            return {
                "user_id": user_id,
                "performance_metrics": performance_metrics,
                "recommendations": await self.get_enhancement_recommendations(user_id),
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            print(f"Error monitoring CAA performance: {e}")
            return {
                "user_id": user_id,
                "performance_metrics": {},
                "recommendations": [],
                "timestamp": datetime.utcnow()
            }











