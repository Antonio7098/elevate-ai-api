"""
Premium API endpoints for advanced RAG features.
Provides premium chat, graph search, and advanced context assembly endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

from .schemas import (
    PremiumChatRequest, PremiumChatResponse,
    GraphSearchRequest, GraphSearchResponse,
    ExpertSelection, Response,
    LangGraphChatRequest, LangGraphChatResponse,
    LearningWorkflowRequest, LearningWorkflowResponse,
    WorkflowStatusResponse,
    CAARequest, CAAResponse, SessionContext,
    AdvancedSearchRequest, AdvancedSearchResponse,
    MultiModalSearchRequest, MultiModalSearchResponse
)
from .middleware import PremiumUserMiddleware
from ...core.premium.agents.routing_agent import PremiumRoutingAgent
from ...core.premium.graph_store import Neo4jGraphStore
from ...core.premium.workflows.learning_workflow import LearningWorkflow, AdaptiveLearningWorkflow
from ...core.premium.langgraph_setup import LangGraphSetup
from ...core.premium.context_assembly_agent import ContextAssemblyAgent
from ...core.premium.rag_fusion import RAGFusionService
from ...core.premium.search_optimization import SearchOptimizer
from ...core.premium.multimodal_rag import MultiModalRAG, MultiModalQuery
from ...core.premium.caa_integration import CAAIntegration
from ...core.premium.model_cascader import ModelCascader
from ...core.premium.token_optimizer import TokenOptimizer
from ...core.premium.privacy import PrivacyPreservingAnalytics
from ...core.premium.load_balancer import PremiumLoadBalancer
from ...core.premium.monitoring import PremiumMonitoringSystem

# Premium API router with authentication
premium_router = APIRouter(prefix="/premium", tags=["premium"])

# Initialize premium services
premium_middleware = PremiumUserMiddleware()
routing_agent = PremiumRoutingAgent()
graph_store = Neo4jGraphStore()
learning_workflow = LearningWorkflow()
adaptive_workflow = AdaptiveLearningWorkflow()
langgraph_setup = LangGraphSetup()
context_assembly_agent = ContextAssemblyAgent()
rag_fusion_service = RAGFusionService()
search_optimizer = SearchOptimizer()
multimodal_rag = MultiModalRAG()
caa_integration = CAAIntegration()
model_cascader = ModelCascader()
token_optimizer = TokenOptimizer()
privacy_analytics = PrivacyPreservingAnalytics()
load_balancer = PremiumLoadBalancer()
premium_monitoring = PremiumMonitoringSystem()

# In-memory budget store (stub)
_user_budget_store = {}

@premium_router.post("/chat/advanced")
async def premium_chat_endpoint(request: PremiumChatRequest):
    """Premium chat with advanced RAG and multi-agent orchestration"""
    # Validate premium access
    if not await premium_middleware.validate_premium_access(request.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    
    # Execute full LangGraph workflow for premium chat
    workflow_result = await routing_agent.execute_workflow(
        user_query=request.query,
        user_id=request.user_id,
        user_context=request.user_context
    )

    # Get real model information from the workflow
    model_used = workflow_result.get("model_used", "gemini-1.5-flash")
    response_text = workflow_result.get("response", "")
    
    # Calculate estimated metrics
    estimated_cost = workflow_result.get("estimated_cost", len(response_text) * 0.00001)
    token_count = workflow_result.get("token_count", len(response_text) // 4)
    
    # Determine if optimization was applied
    optimization_applied = request.optimize_cost and estimated_cost > 0.001
    
    return PremiumChatResponse(
        response=response_text,
        experts_used=workflow_result.get("agents_used", []),
        confidence_score=workflow_result.get("confidence_score", 0.8),
        timestamp=datetime.utcnow(),
        model_used=model_used,
        estimated_cost=estimated_cost,
        token_count=token_count,
        optimization_applied=optimization_applied
    )

@premium_router.post("/chat/graph-search")
async def graph_search_endpoint(request: GraphSearchRequest):
    """Graph-based search for premium users"""
    # Validate premium access
    if not await premium_middleware.validate_premium_access(request.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    
    # Perform graph-based search
    graph_results = await graph_store.query_graph(
        query=request.query,
        user_id=request.user_id
    )
    
    return GraphSearchResponse(
        results=graph_results,
        query=request.query,
        timestamp=datetime.utcnow()
    )

@premium_router.get("/health")
async def premium_health_check():
    """Health check for premium API services"""
    return {
        "status": "healthy",
        "services": {
            "routing_agent": "active",
            "graph_store": "active",
            "premium_middleware": "active",
            "learning_workflow": "active",
            "adaptive_workflow": "active",
            "langgraph_setup": "active"
        },
        "timestamp": datetime.utcnow()
    }

@premium_router.post("/chat/langgraph")
async def langgraph_chat_endpoint(request: LangGraphChatRequest):
    """Premium chat using LangGraph multi-agent orchestration"""
    # Validate premium access
    if not await premium_middleware.validate_premium_access(request.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    
    # Execute LangGraph workflow
    workflow_result = await routing_agent.execute_workflow(
        user_query=request.query,
        user_id=request.user_id,
        user_context=request.user_context
    )
    
    return LangGraphChatResponse(
        response=workflow_result["response"],
        agents_used=workflow_result["agents_used"],
        workflow_status=workflow_result["workflow_status"],
        metadata=workflow_result["metadata"],
        timestamp=datetime.utcnow()
    )

@premium_router.post("/learning/workflow")
async def learning_workflow_endpoint(request: LearningWorkflowRequest):
    """Execute complex learning workflows"""
    # Validate premium access
    if not await premium_middleware.validate_premium_access(request.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    
    # Execute learning workflow
    workflow_result = await learning_workflow.execute_workflow(
        user_query=request.learning_goal,
        user_id=request.user_id,
        user_context=request.user_context
    )
    
    return LearningWorkflowResponse(
        learning_plan=workflow_result.get("learning_plan", ""),
        adapted_plan=workflow_result.get("adapted_plan", ""),
        progress_evaluation=workflow_result.get("progress_evaluation", ""),
        workflow_status=workflow_result.get("workflow_status", ""),
        metadata=workflow_result.get("metadata", {}),
        timestamp=datetime.utcnow()
    )

@premium_router.post("/context/assemble")
async def context_assembly_endpoint(request: CAARequest):
    """Context Assembly Agent endpoint for premium users"""
    # Validate premium access
    if not await premium_middleware.validate_premium_access(request.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    
    try:
        # Execute context assembly pipeline
        caa_request = CAARequest(
            query=request.query,
            user_id=request.user_id,
            mode=request.mode,
            session_context=request.session_context,
            hints=request.hints,
            token_budget=request.token_budget,
            latency_budget_ms=request.latency_budget_ms
        )
        
        response = await context_assembly_agent.assemble_context(caa_request)
        
        return CAAResponse(
            assembled_context=response.assembled_context,
            short_context=response.short_context,
            long_context=response.long_context,
            knowledge_primitives=response.knowledge_primitives,
            examples=response.examples,
            tool_outputs=response.tool_outputs,
            sufficiency_score=response.sufficiency_score,
            token_count=response.token_count,
            rerank_scores=response.rerank_scores,
            warnings=response.warnings,
            cache_key=response.cache_key,
            timestamp=response.timestamp
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context assembly failed: {str(e)}"
        )

@premium_router.get("/workflow/status/{workflow_id}")
async def workflow_status_endpoint(workflow_id: str):
    """Get status of running LangGraph workflows"""
    # Get workflow status
    debug_info = langgraph_setup.monitoring.debug_workflow(workflow_id)
    
    return WorkflowStatusResponse(
        workflow_id=workflow_id,
        status=debug_info.get("status", "unknown"),
        logs=debug_info.get("logs", []),
        metrics=debug_info.get("metrics", {}),
        timestamp=datetime.utcnow()
    )

@premium_router.post("/workflow/adaptive")
async def adaptive_workflow_endpoint(request: LearningWorkflowRequest):
    """Execute adaptive learning workflows"""
    # Validate premium access
    if not await premium_middleware.validate_premium_access(request.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    
    # Execute adaptive workflow
    workflow_result = await adaptive_workflow.execute_workflow(
        user_query=request.learning_goal,
        user_id=request.user_id,
        user_context=request.user_context
    )
    
    return LearningWorkflowResponse(
        learning_plan=workflow_result.get("learning_plan", ""),
        adapted_plan=workflow_result.get("adapted_plan", ""),
        progress_evaluation=workflow_result.get("progress_evaluation", ""),
        workflow_status=workflow_result.get("workflow_status", ""),
        metadata=workflow_result.get("metadata", {}),
        timestamp=datetime.utcnow()
    )

@premium_router.post("/search/advanced")
async def advanced_search_endpoint(request: AdvancedSearchRequest):
    """Advanced search with multiple retrieval strategies for premium users"""
    # Validate premium access
    if not await premium_middleware.validate_premium_access(request.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    
    try:
        # Perform RAG-Fusion search
        fused_results = await rag_fusion_service.multi_retrieve(
            query=request.query,
            user_id=request.user_id
        )
        
        # Optimize search results
        optimized_results = await search_optimizer.optimize_search_results(
            results=fused_results.fused_chunks,
            query=request.query,
            context={"user_id": request.user_id, "mode": request.mode}
        )
        
        return AdvancedSearchResponse(
            results=[{
                "content": result.content,
                "score": result.score,
                "source": result.source,
                "metadata": result.metadata,
                "optimization_scores": result.optimization_scores
            } for result in optimized_results.results],
            strategy_scores=fused_results.strategy_scores,
            fusion_quality=fused_results.fusion_quality,
            optimization_metrics=optimized_results.optimization_metrics,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Advanced search failed: {str(e)}"
        )

@premium_router.post("/search/multimodal")
async def multimodal_search_endpoint(request: MultiModalSearchRequest):
    """Multi-modal search for premium users"""
    # Validate premium access
    if not await premium_middleware.validate_premium_access(request.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    
    try:
        # Create multi-modal query
        multimodal_query = MultiModalQuery(
            text_query=request.text_query,
            image_query=request.image_query,
            audio_query=request.audio_query,
            code_query=request.code_query,
            diagram_query=request.diagram_query,
            modality_weights=request.modality_weights
        )
        
        # Perform multi-modal retrieval
        multimodal_results = await multimodal_rag.retrieve_multimodal(multimodal_query)
        
        # Generate multi-modal response
        multimodal_response = await multimodal_rag.generate_multimodal_response(multimodal_results)
        
        return MultiModalSearchResponse(
            text_results=[{
                "content": result.content,
                "score": result.score,
                "source": result.source,
                "metadata": result.metadata
            } for result in multimodal_results.text_results],
            image_results=[{
                "content": result.content,
                "score": result.score,
                "source": result.source,
                "metadata": result.metadata
            } for result in multimodal_results.image_results],
            code_results=[{
                "content": result.content,
                "score": result.score,
                "source": result.source,
                "metadata": result.metadata
            } for result in multimodal_results.code_results],
            diagram_results=[{
                "content": result.content,
                "score": result.score,
                "source": result.source,
                "metadata": result.metadata
            } for result in multimodal_results.diagram_results],
            audio_results=[{
                "content": result.content,
                "score": result.score,
                "source": result.source,
                "metadata": result.metadata
            } for result in multimodal_results.audio_results],
            fusion_scores=multimodal_results.fusion_scores,
            cross_modal_relationships=multimodal_results.cross_modal_relationships,
            text_response=multimodal_response.text_response,
            image_response=multimodal_response.image_response,
            audio_response=multimodal_response.audio_response,
            code_response=multimodal_response.code_response,
            diagram_response=multimodal_response.diagram_response,
            cross_modal_explanations=multimodal_response.cross_modal_explanations,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-modal search failed: {str(e)}"
        )

@premium_router.post("/search/graph")
async def graph_search_endpoint(request: GraphSearchRequest):
    """Graph-based search with relationship traversal for premium users"""
    # Validate premium access
    if not await premium_middleware.validate_premium_access(request.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    
    try:
        # Perform graph-based search with enhanced features
        graph_results = await graph_store.query_graph(
            query=request.query,
            user_id=request.user_id
        )
        
        # Enhance with RAG-Fusion for better results
        fused_results = await rag_fusion_service.adaptive_fusion(
            query=request.query,
            user_id=request.user_id
        )
        
        return GraphSearchResponse(
            results=graph_results,
            query=request.query,
            adaptive_strategy=fused_results.strategy_used,
            adaptation_reason=fused_results.adaptation_reason,
            performance_metrics=fused_results.performance_metrics,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph search failed: {str(e)}"
        )


@premium_router.post("/optimize/tokens")
async def optimize_tokens_endpoint(content: str, max_tokens: int = 1000):
    """Optimize token usage for given content."""
    try:
        optimized = await token_optimizer.optimize_context_window(content, max_tokens)
        return {
            "optimized_content": optimized.content,
            "original_tokens": optimized.original_tokens,
            "optimized_tokens": optimized.optimized_tokens,
            "compression_ratio": optimized.compression_ratio,
            "quality_score": optimized.quality_score,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token optimization failed: {str(e)}")


@premium_router.post("/chat/cascade")
async def chat_cascade_endpoint(request: PremiumChatRequest):
    """Chat with model cascading and early exit optimization."""
    try:
        # Early exit when possible
        cascaded = await model_cascader.early_exit_optimization(request.query)
        if cascaded.confidence < 0.6 and request.mode != "quiz":
            # escalate via full cascade
            cascaded = await model_cascader.select_and_execute(request.query, user_tier="premium")
        
        # Calculate estimated metrics
        response_text = cascaded.content
        estimated_cost = cascaded.cost_estimate or (len(response_text) * 0.00001)
        token_count = len(response_text) // 4
        
        return {
            "response": response_text,
            "model_used": cascaded.model_used or "gemini-1.5-flash",
            "confidence": cascaded.confidence,
            "cost_estimate": estimated_cost,
            "escalations": cascaded.escalations,
            "timestamp": cascaded.timestamp,
            "estimated_cost": estimated_cost,
            "token_count": token_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cascade chat failed: {str(e)}")


@premium_router.post("/privacy/analyze")
async def privacy_analyze_endpoint(user_data: Dict[str, Any]):
    """Run privacy-preserving analytics."""
    try:
        insights = await privacy_analytics.analyze_with_privacy(user_data)
        return {
            "insights": insights.insights,
            "epsilon": insights.epsilon,
            "timestamp": insights.timestamp,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Privacy analysis failed: {str(e)}")


@premium_router.post("/load/distribute")
async def load_distribute_endpoint(request_data: Dict[str, Any]):
    """Distribute load across backends (stub)."""
    try:
        routed = await load_balancer.distribute_load(request_data)
        return routed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load distribution failed: {str(e)}")


@premium_router.get("/cost/analytics")
async def cost_analytics(user_id: Optional[str] = None, time_range: str = "24h"):
    """Get detailed cost analytics for premium operations."""
    try:
        report = await premium_monitoring.generate_performance_report(time_range=time_range)
        budget = _user_budget_store.get(user_id, None) if user_id else None
        return {
            "overview": {
                "time_range": report.time_range,
                "total_operations": report.total_operations,
                "avg_latency_ms": report.avg_latency_ms,
                "total_cost": report.total_cost,
                "avg_quality_score": report.avg_quality_score,
                "cache_hit_rate": report.cache_hit_rate,
                "cost_efficiency_score": report.cost_efficiency_score,
            },
            "alerts": report.alerts,
            "recommendations": report.recommendations,
            "budget": budget,
            "timestamp": report.timestamp,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cost analytics failed: {str(e)}")


@premium_router.get("/cost/optimization")
async def cost_optimization(user_id: Optional[str] = None, time_range: str = "24h"):
    """Get cost optimization recommendations."""
    try:
        report = await premium_monitoring.generate_performance_report(time_range=time_range)
        return {
            "recommendations": report.recommendations,
            "time_range": report.time_range,
            "timestamp": report.timestamp,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cost optimization retrieval failed: {str(e)}")


class Budget(BaseModel):
    user_id: str
    monthly_budget_usd: float
    alert_threshold_pct: float = 0.8


@premium_router.post("/cost/budget")
async def set_cost_budget(budget: Budget):
    """Set and manage cost budgets for premium users."""
    try:
        _user_budget_store[budget.user_id] = {
            "monthly_budget_usd": budget.monthly_budget_usd,
            "alert_threshold_pct": budget.alert_threshold_pct,
            "updated_at": datetime.utcnow().isoformat(),
        }
        return {
            "status": "ok",
            "budget": _user_budget_store[budget.user_id],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Setting budget failed: {str(e)}")

