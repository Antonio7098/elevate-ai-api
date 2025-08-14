"""
Integration tests for cost optimization service with real LLM calls.
Tests the actual cost optimization system using real Gemini API responses.
"""

import pytest
import asyncio
import os
from datetime import datetime
from unittest.mock import Mock, AsyncMock

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Continue without dotenv if not available

from app.core.premium.cost_optimization_service import (
    CostOptimizationService, CostAnalysis, CostBreakdown, OptimizationRecommendation
)
from app.core.premium.context_assembly_agent import ContextAssemblyAgent, CAARequest
from app.core.premium.gemini_service import GeminiService

class TestCostOptimizationRealLLM:
    """Integration tests using real LLM calls"""
    
    def setup_method(self):
        """Set up test fixtures with real services"""
        # Check if we have the required API keys
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set - skipping real LLM tests")
        
        self.cost_optimization = CostOptimizationService()
        self.gemini_service = GeminiService()
        
        # Create a mock core API client to prevent connection errors
        from unittest.mock import patch
        from app.core.premium.core_api_client import CoreAPIClient
        
        # Mock the core API client methods
        self.mock_core_api = Mock(spec=CoreAPIClient)
        self.mock_core_api.get_user_memory = AsyncMock(return_value={
            "cognitiveApproach": "BALANCED",
            "preferredExplanationStyle": "STEP_BY_STEP",
            "learningStyle": "VISUAL"
        })
        self.mock_core_api.get_knowledge_primitives = AsyncMock(return_value=[])
        
        # Create context assembly agent with mocked dependencies
        with patch('app.core.premium.context_assembly_agent.CoreAPIClient') as mock_class:
            mock_class.return_value = self.mock_core_api
            self.context_assembly_agent = ContextAssemblyAgent()
            
            # Mock additional dependencies
            self.context_assembly_agent.hybrid_retriever = Mock()
            self.context_assembly_agent.hybrid_retriever.retrieve = AsyncMock(return_value=[
                {"content": "Sample content 1", "source": "test", "relevance": 0.9},
                {"content": "Sample content 2", "source": "test", "relevance": 0.8}
            ])
            
            self.context_assembly_agent.graph_store = Mock()
            self.context_assembly_agent.graph_store.traverse_graph = AsyncMock(return_value=[])
            
            self.context_assembly_agent.tool_executor = Mock()
            self.context_assembly_agent.tool_executor.execute_tool = AsyncMock(return_value={
                "output": "Sample tool output",
                "tool": "test_tool"
            })
            
            # Mock the cross-encoder reranker
            self.context_assembly_agent.cross_encoder = Mock()
            self.context_assembly_agent.cross_encoder.rerank_chunks = AsyncMock(return_value=[
                {"content": "Reranked content 1", "source": "test", "rerank_score": 0.9},
                {"content": "Reranked content 2", "source": "test", "rerank_score": 0.8}
            ])
            
            # Mock the sufficiency classifier
            self.context_assembly_agent.sufficiency_classifier = Mock()
            self.context_assembly_agent.sufficiency_classifier.check_sufficiency = AsyncMock(return_value={
                "score": 0.75,
                "is_sufficient": True,
                "missing_aspects": []
            })
            
            # Mock tool selection and execution methods
            self.context_assembly_agent.select_tools = AsyncMock(return_value=["test_tool"])
            self.context_assembly_agent.execute_tools = AsyncMock(return_value=[
                {"tool": "test_tool", "output": "Sample tool output"}
            ])
        
        # Test data for real LLM calls
        self.test_queries = [
            "Explain machine learning concepts with examples",
            "What are the best practices for API design?",
            "How does blockchain technology work?",
            "Explain quantum computing in simple terms"
        ]

    @pytest.mark.asyncio
    async def test_real_llm_cost_analysis(self):
        """Test cost analysis with real LLM responses"""
        print("\n🧪 Testing real LLM cost analysis...")
        
        # Create real context assembly request
        caa_request = CAARequest(
            query="Explain machine learning concepts with examples",
            user_id="test_user_123",
            mode="deep_dive",
            session_context={
                "previous_queries": ["What is AI?"],
                "user_preferences": {"detail_level": "comprehensive"}
            },
            hints=["focus on practical applications"],
            token_budget=4000,
            latency_budget_ms=5000
        )
        
        try:
            # Execute real context assembly
            print("   📡 Executing real context assembly...")
            response = await self.context_assembly_agent.assemble_context(caa_request)
            
            # Analyze costs from real response
            print("   💰 Analyzing costs from real response...")
            cost_analysis = await self.cost_optimization.analyze_costs(response.tool_outputs)
            
            # Validate the analysis
            assert isinstance(cost_analysis, CostAnalysis)
            assert cost_analysis.total_cost >= 0
            assert len(cost_analysis.cost_breakdown) > 0
            assert len(cost_analysis.optimization_recommendations) > 0
            
            print(f"   ✅ Real cost analysis successful:")
            print(f"      - Total cost: ${cost_analysis.total_cost:.6f}")
            print(f"      - Cost breakdown: {len(cost_analysis.cost_breakdown)} stages")
            print(f"      - Optimization recommendations: {len(cost_analysis.optimization_recommendations)}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Real LLM test failed: {e}")
            pytest.fail(f"Real LLM integration test failed: {e}")

    @pytest.mark.asyncio
    async def test_real_model_cost_comparison(self):
        """Test actual cost differences between different models"""
        print("\n🧪 Testing real model cost comparison...")
        
        test_prompt = "Explain the concept of artificial intelligence in one sentence."
        
        try:
            # Test with different models and measure actual costs
            models_to_test = [
                "gemini-2.5-flash-lite",
                "gemini-2.5-flash", 
                "gemini-1.5-flash",
                "gemini-1.5-pro"
            ]
            
            model_costs = {}
            
            for model in models_to_test:
                print(f"   📡 Testing {model}...")
                
                # Generate response with real model
                response = await self.gemini_service.generate(test_prompt, model)
                
                # Calculate actual cost
                input_tokens = len(test_prompt) // 4  # Rough estimation
                output_tokens = len(response) // 4
                
                cost_estimate = await self.gemini_service.get_cost_estimate(
                    model, input_tokens, output_tokens
                )
                
                model_costs[model] = {
                    "response": response,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_cost": cost_estimate.get("total_cost", 0),
                    "cost_per_1k_tokens": cost_estimate.get("cost_per_1k_tokens", 0)
                }
                
                print(f"      - Cost: ${cost_estimate.get('total_cost', 0):.6f}")
                print(f"      - Tokens: {input_tokens} + {output_tokens} = {input_tokens + output_tokens}")
            
            # Validate cost hierarchy
            flash_lite_cost = model_costs["gemini-2.5-flash-lite"]["total_cost"]
            flash_cost = model_costs["gemini-2.5-flash"]["total_cost"]
            pro_cost = model_costs["gemini-1.5-pro"]["total_cost"]
            
            # Flash Lite should be cheaper than Flash
            assert flash_lite_cost <= flash_cost, f"Flash Lite (${flash_lite_cost:.6f}) should be cheaper than Flash (${flash_cost:.6f})"
            
            # Flash should be cheaper than Pro
            assert flash_cost <= pro_cost, f"Flash (${flash_cost:.6f}) should be cheaper than Pro (${pro_cost:.6f})"
            
            print(f"   ✅ Cost hierarchy validated:")
            print(f"      - Flash Lite: ${flash_lite_cost:.6f}")
            print(f"      - Flash: ${flash_cost:.6f}")
            print(f"      - Pro: ${pro_cost:.6f}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Real model cost comparison failed: {e}")
            pytest.fail(f"Real model cost comparison failed: {e}")

    @pytest.mark.asyncio
    async def test_real_optimization_recommendations(self):
        """Test optimization recommendations with real LLM usage patterns"""
        print("\n🧪 Testing real optimization recommendations...")
        
        try:
            # Simulate expensive usage pattern (using Pro for simple tasks)
            expensive_metrics = {
                "simple_routing_cost": {
                    "model": "gemini-1.5-pro",  # Expensive model for simple task
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "input_cost": 0.0035,
                    "output_cost": 0.00525,
                    "total_cost": 0.00875,
                    "cost_per_1k_tokens": 0.00875
                }
            }
            
            # Analyze costs and get recommendations
            analysis = await self.cost_optimization.analyze_costs(expensive_metrics)
            
            # Should recommend Flash Lite for routing tasks
            assert len(analysis.optimization_recommendations) > 0
            
            routing_recommendation = None
            for rec in analysis.optimization_recommendations:
                if "routing" in rec.stage:
                    routing_recommendation = rec
                    break
            
            assert routing_recommendation is not None, "Should have recommendation for routing task"
            assert "flash-lite" in routing_recommendation.recommended_model.lower(), "Should recommend Flash Lite for routing"
            
            # Get the corresponding cost breakdown to show current model
            routing_breakdown = None
            for breakdown in analysis.cost_breakdown:
                if "routing" in breakdown.stage:
                    routing_breakdown = breakdown
                    break
            
            print(f"   ✅ Optimization recommendations validated:")
            if routing_breakdown:
                print(f"      - Current model: {routing_breakdown.model_used}")
            print(f"      - Recommended model: {routing_recommendation.recommended_model}")
            print(f"      - Potential savings: {routing_recommendation.savings_percentage:.1%}")
            print(f"      - Reasoning: {routing_recommendation.reasoning}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Real optimization recommendations failed: {e}")
            pytest.fail(f"Real optimization recommendations failed: {e}")

    @pytest.mark.asyncio
    async def test_real_budget_compliance(self):
        """Test budget compliance with real cost data"""
        print("\n🧪 Testing real budget compliance...")
        
        try:
            # Create realistic cost metrics
            realistic_metrics = {
                "query_augmentation_cost": {
                    "model": "gemini-2.5-flash-lite",
                    "input_tokens": 800,
                    "output_tokens": 400,
                    "input_cost": 0.00002,
                    "output_cost": 0.00004,
                    "total_cost": 0.00006,
                    "cost_per_1k_tokens": 0.00006
                },
                "content_generation_cost": {
                    "model": "gemini-2.5-flash",
                    "input_tokens": 1200,
                    "output_tokens": 800,
                    "input_cost": 0.00009,
                    "output_cost": 0.00024,
                    "total_cost": 0.00033,
                    "cost_per_1k_tokens": 0.00033
                }
            }
            
            # Test different user tiers
            tiers_to_test = ["basic", "premium", "enterprise"]
            
            for tier in tiers_to_test:
                print(f"   💳 Testing {tier} tier...")
                
                compliance = await self.cost_optimization.validate_budget_compliance(
                    await self.cost_optimization.analyze_costs(realistic_metrics), 
                    tier
                )
                
                assert "is_compliant" in compliance
                assert "target_cost" in compliance
                assert "actual_cost" in compliance
                assert "budget_remaining" in compliance
                
                print(f"      - Compliant: {compliance['is_compliant']}")
                print(f"      - Target: ${compliance['target_cost']:.4f}")
                print(f"      - Actual: ${compliance['actual_cost']:.6f}")
                print(f"      - Remaining: ${compliance['budget_remaining']:.6f}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Real budget compliance failed: {e}")
            pytest.fail(f"Real budget compliance failed: {e}")

    @pytest.mark.asyncio
    async def test_real_cost_efficiency_scoring(self):
        """Test cost efficiency scoring with real usage patterns"""
        print("\n🧪 Testing real cost efficiency scoring...")
        
        try:
            # Test different cost scenarios
            scenarios = {
                "low_cost": {
                    "efficient_cost": {
                        "model": "gemini-2.5-flash-lite",
                        "input_tokens": 500,
                        "output_tokens": 250,
                        "input_cost": 0.0000125,
                        "output_cost": 0.000025,
                        "total_cost": 0.0000375,
                        "cost_per_1k_tokens": 0.0000375
                    }
                },
                "high_cost": {
                    "inefficient_cost": {
                        "model": "gemini-1.5-pro",
                        "input_tokens": 2000,
                        "output_tokens": 1000,
                        "input_cost": 0.007,
                        "output_cost": 0.0105,
                        "total_cost": 0.0175,
                        "cost_per_1k_tokens": 0.0175
                    }
                }
            }
            
            scores = {}
            
            for scenario_name, scenario_data in scenarios.items():
                print(f"   📊 Testing {scenario_name} scenario...")
                
                analysis = await self.cost_optimization.analyze_costs(scenario_data)
                scores[scenario_name] = analysis.cost_efficiency_score
                
                print(f"      - Total cost: ${analysis.total_cost:.6f}")
                print(f"      - Efficiency score: {analysis.cost_efficiency_score:.2f}")
            
            # Low cost should have higher efficiency score
            assert scores["low_cost"] > scores["high_cost"], f"Low cost scenario should have higher efficiency score: {scores['low_cost']} vs {scores['high_cost']}"
            
            print(f"   ✅ Cost efficiency scoring validated:")
            print(f"      - Low cost score: {scores['low_cost']:.2f}")
            print(f"      - High cost score: {scores['high_cost']:.2f}")
            print(f"      - Score difference: {scores['low_cost'] - scores['high_cost']:.2f}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Real cost efficiency scoring failed: {e}")
            pytest.fail(f"Real cost efficiency scoring failed: {e}")

    @pytest.mark.asyncio
    async def test_real_end_to_end_workflow(self):
        """Test complete end-to-end cost optimization workflow"""
        print("\n🧪 Testing real end-to-end workflow...")
        
        try:
            # Step 1: Create realistic context assembly request
            caa_request = CAARequest(
                query="Explain the benefits of microservices architecture",
                user_id="test_user_456",
                mode="chat",
                session_context={
                    "previous_queries": ["What is software architecture?"],
                    "user_preferences": {"detail_level": "moderate"}
                },
                hints=[],
                token_budget=3000,
                latency_budget_ms=3000
            )
            
            print("   🔄 Step 1: Context assembly...")
            response = await self.context_assembly_agent.assemble_context(caa_request)
            
            print("   🔄 Step 2: Cost analysis...")
            cost_analysis = await self.cost_optimization.analyze_costs(response.tool_outputs)
            
            print("   🔄 Step 3: Optimization recommendations...")
            recommendations = cost_analysis.optimization_recommendations
            
            print("   🔄 Step 4: Budget compliance check...")
            compliance = await self.cost_optimization.validate_budget_compliance(
                cost_analysis, "premium"
            )
            
            # Validate the complete workflow
            assert len(response.assembled_context) > 0, "Should have assembled context"
            assert cost_analysis.total_cost >= 0, "Should have valid cost analysis"
            assert len(recommendations) > 0, "Should have optimization recommendations"
            assert "is_compliant" in compliance, "Should have budget compliance data"
            
            print(f"   ✅ End-to-end workflow successful:")
            print(f"      - Context length: {len(response.assembled_context)} chars")
            print(f"      - Total cost: ${cost_analysis.total_cost:.6f}")
            print(f"      - Recommendations: {len(recommendations)}")
            print(f"      - Budget compliant: {compliance['is_compliant']}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ End-to-end workflow failed: {e}")
            pytest.fail(f"End-to-end workflow failed: {e}")

class TestCostOptimizationPerformance:
    """Performance tests with real LLM calls"""
    
    def setup_method(self):
        """Set up performance test fixtures"""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set - skipping performance tests")
        
        self.cost_optimization = CostOptimizationService()
        self.gemini_service = GeminiService()

    @pytest.mark.asyncio
    async def test_cost_analysis_performance(self):
        """Test performance of cost analysis with real data"""
        print("\n⚡ Testing cost analysis performance...")
        
        try:
            # Create large dataset for performance testing
            large_metrics = {}
            for i in range(10):
                large_metrics[f"stage_{i}_cost"] = {
                    "model": "gemini-2.5-flash-lite" if i % 2 == 0 else "gemini-1.5-pro",
                    "input_tokens": 1000 + (i * 100),
                    "output_tokens": 500 + (i * 50),
                    "input_cost": 0.0001 + (i * 0.00001),
                    "output_cost": 0.0002 + (i * 0.00002),
                    "total_cost": 0.0003 + (i * 0.00003),
                    "cost_per_1k_tokens": 0.0003 + (i * 0.00003)
                }
            
            # Measure performance
            import time
            start_time = time.time()
            
            analysis = await self.cost_optimization.analyze_costs(large_metrics)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should complete within reasonable time
            assert execution_time < 5.0, f"Cost analysis took too long: {execution_time:.2f}s"
            assert len(analysis.cost_breakdown) == 10, "Should process all stages"
            
            print(f"   ✅ Performance test passed:")
            print(f"      - Execution time: {execution_time:.3f}s")
            print(f"      - Stages processed: {len(analysis.cost_breakdown)}")
            print(f"      - Total cost: ${analysis.total_cost:.6f}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Performance test failed: {e}")
            pytest.fail(f"Performance test failed: {e}")

if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])
