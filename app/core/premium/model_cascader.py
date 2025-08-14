"""
Model cascading and early-exit system for premium cost optimization.
Provides intelligent model selection and escalation based on confidence and user tier.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

from .gemini_service import GeminiService
from ..llm_service import llm_service


@dataclass
class CascadedResponse:
    content: str
    model_used: str
    confidence: float
    cost_estimate: float
    escalations: int
    timestamp: datetime
    metadata: Dict[str, Any]


class ConfidenceChecker:
    """Confidence checker based on response quality metrics."""

    def estimate_confidence(self, text: str) -> float:
        if not text:
            return 0.0
        
        # Analyze response quality
        length_score = min(len(text) / 500.0, 1.0)
        structure_bonus = 0.1 if any(k in text.lower() for k in ["1.", "2.", "step", "conclusion", "therefore", "thus"]) else 0.0
        coherence_bonus = 0.1 if len(text.split('.')) > 2 else 0.0
        
        return max(0.0, min(1.0, 0.4 * length_score + structure_bonus + coherence_bonus + 0.3))


class CostTracker:
    """Cost tracker for different models and providers."""

    MODEL_COST = {
        # Google Gemini models (per 1K tokens)
        "gemini-1.5-flash": 0.0005,
        "gemini-1.5-pro": 0.0025,
        "gemini-2.0-pro": 0.0040,
        "gemini-2.5-flash": 0.000075,
        "gemini-2.5-flash-lite": 0.000025,  # NEW: Perfect for routing/summarization
        "gemini-2.5-pro": 0.0040,
        # OpenRouter models (per 1K tokens)
        "z-ai/glm-4.5-air:free": 0.0001,  # Free tier
        "z-ai/glm-4.5-air": 0.0005,       # Paid tier
    }

    def estimate_cost(self, model: str, chars: int) -> float:
        rate = self.MODEL_COST.get(model, 0.001)
        # Rough proportional cost; in production use actual tokens
        return rate * max(1, chars // 1000)

    def get_cost_efficient_model(self, task_type: str) -> str:
        """Get most cost-efficient model for specific task types"""
        if task_type in ["routing", "summarization", "classification", "compression"]:
            return "gemini-2.5-flash-lite"  # Best for context assembly tasks
        elif task_type in ["simple_generation", "basic_qa"]:
            return "gemini-2.5-flash"  # Good balance for simple tasks
        elif task_type in ["complex_reasoning", "large_context"]:
            return "gemini-1.5-pro"  # Only for complex tasks
        else:
            return "gemini-2.5-flash-lite"  # Default to most cost-effective


class ModelCascader:
    """Intelligent model selection with fallback to OpenRouter."""

    def __init__(self):
        # Primary models (Google Gemini) - START WITH MOST COST-EFFECTIVE
        self.primary_models = [
            "gemini-2.5-flash-lite",  # NEW: Most cost-effective for routing/summarization
            "gemini-2.5-flash",       # Good balance
            "gemini-1.5-flash",       # Fallback
            "gemini-1.5-pro"          # Only for complex reasoning
        ]
        # Fallback models (OpenRouter) - EVEN CHEAPER
        self.fallback_models = ["z-ai/glm-4.5-air:free", "z-ai/glm-4.5-air"]
        
        self.gemini_service = GeminiService()
        self.confidence_checker = ConfidenceChecker()
        self.cost_tracker = CostTracker()

    async def select_and_execute(self, query: str, user_tier: str = "standard", min_confidence: float = 0.65, task_type: str = "general") -> CascadedResponse:
        """Execute with cascading; premium users can escalate further."""
        max_primary_models = 3 if user_tier in ["premium", "enterprise"] else 2
        escalations = 0
        last_response: Optional[str] = None
        last_model = self.primary_models[0]
        confidence = 0.0

        # Start with most cost-effective model for the task
        if task_type in ["routing", "summarization", "classification", "compression"]:
            start_model = "gemini-2.5-flash-lite"
        else:
            start_model = self.cost_tracker.get_cost_efficient_model(task_type)

        # Try primary models (Google Gemini) - starting with most cost-effective
        for idx, model in enumerate(self.primary_models[:max_primary_models]):
            try:
                last_model = model
                prompt = f"[model={model}] Answer the following query succinctly and accurately. Query: {query}"
                last_response = await self.gemini_service.generate(prompt, model)
                confidence = self.confidence_checker.estimate_confidence(last_response)
                
                if confidence >= min_confidence:
                    break
                escalations += 1
                
            except Exception as e:
                print(f"Primary model {model} failed: {e}")
                escalations += 1
                continue

        # If primary models didn't achieve confidence, try fallback (OpenRouter)
        if confidence < min_confidence:
            for model in self.fallback_models:
                try:
                    last_model = model
                    prompt = f"[fallback={model}] Answer the following query: {query}"
                    last_response = await self.gemini_service.generate(prompt, model)
                    confidence = self.confidence_checker.estimate_confidence(last_response)
                    
                    if confidence >= min_confidence:
                        break
                    escalations += 1
                    
                except Exception as e:
                    print(f"Fallback model {model} failed: {e}")
                    escalations += 1
                    continue

        # Calculate estimated cost
        estimated_cost = self.cost_tracker.estimate_cost(last_model, len(last_response or ""))
        
        return CascadedResponse(
            content=last_response or "Failed to generate response",
            model_used=last_model,
            confidence=confidence,
            cost_estimate=estimated_cost,
            escalations=escalations,
            timestamp=datetime.utcnow(),
            metadata={"user_tier": user_tier, "min_confidence": min_confidence, "task_type": task_type}
        )

    async def early_exit_optimization(self, query: str) -> CascadedResponse:
        """Return a cost-optimized response for simple queries."""
        # Heuristic: short factual queries get fast model only
        if len(query) < 80 and any(w in query.lower() for w in ["what", "when", "where", "who", "define", "meaning"]):
            try:
                prompt = f"[model=gemini-1.5-flash] Provide a concise factual answer. Query: {query}"
                response = await self.gemini_service.generate(prompt, "gemini-1.5-flash")
                conf = self.confidence_checker.estimate_confidence(response)
                cost = self.cost_tracker.estimate_cost("gemini-1.5-flash", len(response))
                
                return CascadedResponse(
                    content=response,
                    model_used="gemini-1.5-flash",
                    confidence=conf,
                    cost_estimate=cost,
                    escalations=0,
                    timestamp=datetime.utcnow(),
                    metadata={"optimization": "early_exit", "reason": "simple_query"}
                )
            except Exception as e:
                # Fallback to OpenRouter for simple queries
                try:
                    response = await llm_service.call_openrouter_ai(
                        f"Provide a concise factual answer: {query}",
                        model="z-ai/glm-4.5-air:free",
                        operation="early_exit_fallback"
                    )
                    conf = self.confidence_checker.estimate_confidence(response)
                    cost = self.cost_tracker.estimate_cost("z-ai/glm-4.5-air:free", len(response))
                    
                    return CascadedResponse(
                        content=response,
                        model_used="z-ai/glm-4.5-air:free",
                        confidence=conf,
                        cost_estimate=cost,
                        escalations=0,
                        timestamp=datetime.utcnow(),
                        metadata={"optimization": "early_exit", "reason": "simple_query", "fallback": True}
                    )
                except Exception as fallback_error:
                    raise Exception(f"Both primary and fallback models failed for early exit: {e}, {fallback_error}")

        # For complex queries, use normal cascading
        return await self.select_and_execute(query, user_tier="standard")

    async def select_context_assembly_model(self, context_size: int, complexity: str, task_type: str) -> str:
        """Select optimal model for context assembly tasks"""
        try:
            # For context assembly, prioritize cost efficiency
            if task_type in ["routing", "summarization", "classification", "compression"]:
                if context_size < 10000:  # Small contexts
                    return "gemini-2.5-flash-lite"  # Most cost-effective
                elif context_size < 100000:  # Medium contexts
                    return "gemini-2.5-flash"  # Good balance
                else:  # Large contexts
                    return "gemini-1.5-pro"  # Only when necessary
            
            # For other tasks, use standard selection
            elif complexity == "simple":
                return "gemini-2.5-flash-lite"
            elif complexity == "medium":
                return "gemini-2.5-flash"
            else:
                return "gemini-1.5-pro"
                
        except Exception as e:
            print(f"Error selecting context assembly model: {e}")
            return "gemini-2.5-flash-lite"  # Default to most cost-effective

