"""
Cost Optimization Service for Premium Context Assembly.
Provides cost analysis, optimization recommendations, and budget management.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio

@dataclass
class CostBreakdown:
    """Detailed cost breakdown for context assembly"""
    stage: str
    model_used: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    cost_per_1k_tokens: float
    optimization_potential: float
    recommended_model: str

@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation"""
    stage: str
    current_cost: float
    optimized_cost: float
    savings_percentage: float
    recommended_model: str
    reasoning: str
    implementation_effort: str

@dataclass
class CostAnalysis:
    """Complete cost analysis for context assembly"""
    total_cost: float
    cost_breakdown: List[CostBreakdown]
    optimization_recommendations: List[OptimizationRecommendation]
    total_savings_potential: float
    cost_efficiency_score: float
    timestamp: datetime

class CostOptimizationService:
    """Service for optimizing costs in premium context assembly"""
    
    def __init__(self):
        # Cost targets for different user tiers
        self.cost_targets = {
            "basic": 0.01,      # $0.01 per request
            "premium": 0.05,    # $0.05 per request
            "enterprise": 0.10  # $0.10 per request
        }
        
        # Model efficiency ratings for different task types
        self.task_model_efficiency = {
            "routing": {
                "gemini-2.5-flash-lite": 0.95,    # Excellent for routing
                "gemini-2.5-flash": 0.85,         # Good for routing
                "gemini-1.5-flash": 0.70,         # Acceptable
                "gemini-1.5-pro": 0.60            # Overkill
            },
            "summarization": {
                "gemini-2.5-flash-lite": 0.90,    # Excellent for summarization
                "gemini-2.5-flash": 0.80,         # Good for summarization
                "gemini-1.5-flash": 0.65,         # Acceptable
                "gemini-1.5-pro": 0.55            # Overkill
            },
            "classification": {
                "gemini-2.5-flash-lite": 0.85,    # Good for classification
                "gemini-2.5-flash": 0.90,         # Excellent for classification
                "gemini-1.5-flash": 0.75,         # Good
                "gemini-1.5-pro": 0.70            # Overkill
            },
            "compression": {
                "gemini-2.5-flash-lite": 0.80,    # Good for compression
                "gemini-2.5-flash": 0.90,         # Excellent for compression
                "gemini-1.5-flash": 0.70,         # Acceptable
                "gemini-1.5-pro": 0.65            # Overkill
            }
        }

    async def analyze_costs(self, context_assembly_metrics: Dict[str, Any]) -> CostAnalysis:
        """Analyze costs from context assembly and provide optimization recommendations"""
        try:
            cost_breakdown = []
            optimization_recommendations = []
            total_cost = 0
            
            # Analyze each stage
            for stage, cost_data in context_assembly_metrics.items():
                if "cost" in stage and isinstance(cost_data, dict):
                    breakdown = await self._analyze_stage_cost(stage, cost_data)
                    cost_breakdown.append(breakdown)
                    total_cost += breakdown.total_cost
                    
                    # Generate optimization recommendation
                    recommendation = await self._generate_optimization_recommendation(breakdown)
                    optimization_recommendations.append(recommendation)
            
            # Calculate total savings potential
            total_savings = sum(rec.savings_percentage * rec.current_cost for rec in optimization_recommendations)
            
            # Calculate cost efficiency score (0-100)
            cost_efficiency_score = max(0, 100 - (total_cost * 1000))  # Scale appropriately
            
            return CostAnalysis(
                total_cost=total_cost,
                cost_breakdown=cost_breakdown,
                optimization_recommendations=optimization_recommendations,
                total_savings_potential=total_savings,
                cost_efficiency_score=cost_efficiency_score,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            print(f"Error analyzing costs: {e}")
            raise e

    async def _analyze_stage_cost(self, stage: str, cost_data: Dict[str, Any]) -> CostBreakdown:
        """Analyze cost for a specific stage"""
        try:
            # Extract cost information
            model_used = cost_data.get("model", "unknown")
            input_tokens = cost_data.get("input_tokens", 0)
            output_tokens = cost_data.get("output_tokens", 0)
            input_cost = cost_data.get("input_cost", 0.0)
            output_cost = cost_data.get("output_cost", 0.0)
            total_cost = cost_data.get("total_cost", 0.0)
            
            # Determine task type from stage name
            task_type = await self._determine_task_type(stage)
            
            # Calculate optimization potential
            optimization_potential = await self._calculate_optimization_potential(
                task_type, model_used, total_cost
            )
            
            # Get recommended model for this task
            recommended_model = await self._get_recommended_model(task_type, input_tokens)
            
            return CostBreakdown(
                stage=stage,
                model_used=model_used,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
                cost_per_1k_tokens=cost_data.get("cost_per_1k_tokens", 0.0),
                optimization_potential=optimization_potential,
                recommended_model=recommended_model
            )
            
        except Exception as e:
            print(f"Error analyzing stage cost: {e}")
            raise e

    async def _determine_task_type(self, stage: str) -> str:
        """Determine task type from stage name"""
        stage_lower = stage.lower()
        
        if "augmentation" in stage_lower or "routing" in stage_lower:
            return "routing"
        elif "summarization" in stage_lower or "assembly" in stage_lower:
            return "summarization"
        elif "classification" in stage_lower or "checking" in stage_lower:
            return "classification"
        elif "compression" in stage_lower or "condensation" in stage_lower:
            return "compression"
        else:
            return "general"

    async def _calculate_optimization_potential(self, task_type: str, current_model: str, current_cost: float) -> float:
        """Calculate optimization potential for a task"""
        try:
            if task_type not in self.task_model_efficiency:
                return 0.0
            
            # Get efficiency of current model
            current_efficiency = self.task_model_efficiency[task_type].get(current_model, 0.5)
            
            # Find most efficient model for this task
            best_efficiency = max(self.task_model_efficiency[task_type].values())
            
            # Calculate optimization potential
            if current_efficiency > 0:
                optimization_potential = (best_efficiency - current_efficiency) / best_efficiency
                return min(optimization_potential, 1.0)  # Cap at 100%
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error calculating optimization potential: {e}")
            return 0.0

    async def _get_recommended_model(self, task_type: str, context_size: int) -> str:
        """Get recommended model for a specific task and context size"""
        try:
            if task_type in self.task_model_efficiency:
                # Get models sorted by efficiency for this task
                sorted_models = sorted(
                    self.task_model_efficiency[task_type].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Consider context size for model selection
                if context_size > 100000:  # Large context
                    # Prefer models with larger context windows
                    for model, efficiency in sorted_models:
                        if "pro" in model.lower():
                            return model
                
                # Return most efficient model for the task
                return sorted_models[0][0]
            else:
                return "gemini-2.5-flash-lite"  # Default to most cost-effective
                
        except Exception as e:
            print(f"Error getting recommended model: {e}")
            return "gemini-2.5-flash-lite"

    async def _generate_optimization_recommendation(self, breakdown: CostBreakdown) -> OptimizationRecommendation:
        """Generate optimization recommendation for a cost breakdown"""
        try:
            # Calculate potential savings
            if breakdown.optimization_potential > 0:
                # Estimate cost with recommended model
                current_cost_per_token = breakdown.cost_per_1k_tokens / 1000
                recommended_cost_per_token = await self._get_model_cost_per_token(breakdown.recommended_model)
                
                if recommended_cost_per_token < current_cost_per_token:
                    total_tokens = breakdown.input_tokens + breakdown.output_tokens
                    optimized_cost = total_tokens * recommended_cost_per_token
                    savings = breakdown.total_cost - optimized_cost
                    savings_percentage = savings / breakdown.total_cost if breakdown.total_cost > 0 else 0
                else:
                    optimized_cost = breakdown.total_cost
                    savings_percentage = 0
            else:
                optimized_cost = breakdown.total_cost
                savings_percentage = 0
            
            # Determine implementation effort
            if breakdown.model_used == breakdown.recommended_model:
                implementation_effort = "none"
            elif "flash-lite" in breakdown.recommended_model:
                implementation_effort = "low"
            elif "flash" in breakdown.recommended_model:
                implementation_effort = "medium"
            else:
                implementation_effort = "high"
            
            # Generate reasoning
            reasoning = self._generate_reasoning(breakdown, savings_percentage)
            
            return OptimizationRecommendation(
                stage=breakdown.stage,
                current_cost=breakdown.total_cost,
                optimized_cost=optimized_cost,
                savings_percentage=savings_percentage,
                recommended_model=breakdown.recommended_model,
                reasoning=reasoning,
                implementation_effort=implementation_effort
            )
            
        except Exception as e:
            print(f"Error generating optimization recommendation: {e}")
            raise e

    async def _get_model_cost_per_token(self, model: str) -> float:
        """Get cost per token for a specific model"""
        # Current Gemini pricing (as of 2024)
        pricing = {
            "gemini-2.5-flash-lite": 0.000025 + 0.0001,  # input + output
            "gemini-2.5-flash": 0.000075 + 0.0003,       # input + output
            "gemini-1.5-flash": 0.0005 + 0.0003,         # input + output
            "gemini-1.5-pro": 0.0035 + 0.0105,           # input + output
        }
        
        return pricing.get(model, pricing["gemini-2.5-flash-lite"]) / 1000

    def _generate_reasoning(self, breakdown: CostBreakdown, savings_percentage: float) -> str:
        """Generate reasoning for optimization recommendation"""
        if savings_percentage == 0:
            return f"Current model {breakdown.model_used} is already optimal for this task."
        
        if "flash-lite" in breakdown.recommended_model:
            return f"Switch to {breakdown.recommended_model} for {savings_percentage:.1%} cost savings. Flash Lite is specifically designed for {self._determine_task_type_sync(breakdown.stage)} tasks."
        elif "flash" in breakdown.recommended_model:
            return f"Switch to {breakdown.recommended_model} for {savings_percentage:.1%} cost savings. Flash provides good balance of cost and performance."
        else:
            return f"Consider {breakdown.recommended_model} only if higher quality is required. Current model provides good cost efficiency."

    def _determine_task_type_sync(self, stage: str) -> str:
        """Synchronous version of task type determination for reasoning"""
        stage_lower = stage.lower()
        
        if "augmentation" in stage_lower or "routing" in stage_lower:
            return "routing"
        elif "summarization" in stage_lower or "assembly" in stage_lower:
            return "summarization"
        elif "classification" in stage_lower or "checking" in stage_lower:
            return "classification"
        elif "compression" in stage_lower or "condensation" in stage_lower:
            return "compression"
        else:
            return "general"

    async def get_cost_summary(self, analysis: CostAnalysis) -> Dict[str, Any]:
        """Get cost summary for reporting"""
        try:
            return {
                "total_cost": analysis.total_cost,
                "total_savings_potential": analysis.total_savings_potential,
                "cost_efficiency_score": analysis.cost_efficiency_score,
                "optimization_priority": sorted(
                    analysis.optimization_recommendations,
                    key=lambda x: x.savings_percentage,
                    reverse=True
                )[:3],  # Top 3 optimizations
                "stage_breakdown": {
                    breakdown.stage: {
                        "cost": breakdown.total_cost,
                        "model": breakdown.model_used,
                        "optimization_potential": breakdown.optimization_potential
                    }
                    for breakdown in analysis.cost_breakdown
                }
            }
            
        except Exception as e:
            print(f"Error generating cost summary: {e}")
            raise e

    async def validate_budget_compliance(self, analysis: CostAnalysis, user_tier: str) -> Dict[str, Any]:
        """Validate if costs comply with user tier budget"""
        try:
            target_cost = self.cost_targets.get(user_tier, 0.10)
            is_compliant = analysis.total_cost <= target_cost
            budget_remaining = max(0, target_cost - analysis.total_cost)
            
            return {
                "is_compliant": is_compliant,
                "target_cost": target_cost,
                "actual_cost": analysis.total_cost,
                "budget_remaining": budget_remaining,
                "compliance_percentage": (analysis.total_cost / target_cost) * 100 if target_cost > 0 else 0,
                "recommendations": [
                    rec for rec in analysis.optimization_recommendations 
                    if rec.savings_percentage > 0.1  # Only significant savings
                ]
            }
            
        except Exception as e:
            print(f"Error validating budget compliance: {e}")
            raise e
