#!/usr/bin/env python3
"""
Test module for Premium Agents with REAL API calls.
Tests content curation, explanation, context assembly, and routing agents.
"""

import asyncio
import os
import time
from typing import Dict, Any, List

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded")
except ImportError:
    print("⚠️  python-dotenv not available")

from app.core.premium.agents.expert_agents import ContentCuratorAgent, ExplanationAgent
from app.core.premium.agents.routing_agent import PremiumRoutingAgent
from app.core.premium.context_assembly_agent import ContextAssemblyAgent
from app.services.llm_service import create_llm_service


class PremiumAgentsTester:
    """Test suite for premium agents with real API calls."""
    
    def __init__(self):
        self.test_results = []
        self.test_content = """# Introduction to Neural Networks

Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information.

## Basic Structure

A neural network has three main layers:
1. Input layer - receives the input data
2. Hidden layers - process the information
3. Output layer - produces the final result

## Training Process

The network learns through a process called backpropagation, where it adjusts weights based on the difference between predicted and actual outputs."""
        
    async def test_content_curator_agent(self):
        """Test content curator agent with real API calls."""
        print("\n🔍 Testing Content Curator Agent")
        print("-" * 50)
        
        try:
            # Initialize agent
            curator_agent = ContentCuratorAgent()
            
            # Test content curation
            print("   🎨 Testing content curation...")
            start_time = time.time()
            curated_content = await curator_agent.curate_content(
                content=self.test_content,
                instruction="Make this more engaging and add examples",
                content_type="educational"
            )
            end_time = time.time()
            
            print(f"   ✅ Content curation successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📝 Curated content: {len(curated_content)} characters")
            
            # Test content enhancement
            print("   ✨ Testing content enhancement...")
            start_time = time.time()
            enhanced_content = await curator_agent.enhance_content(
                content=self.test_content,
                enhancement_type="clarity",
                target_audience="beginners"
            )
            end_time = time.time()
            
            print(f"   ✅ Content enhancement successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📝 Enhanced content: {len(enhanced_content)} characters")
            
            return True
            
        except Exception as e:
            print(f"❌ Content curator agent test failed: {e}")
            return False
    
    async def test_explanation_agent(self):
        """Test explanation agent with real API calls."""
        print("\n🔍 Testing Explanation Agent")
        print("-" * 50)
        
        try:
            # Initialize agent
            explanation_agent = ExplanationAgent()
            
            # Test concept explanation
            print("   📚 Testing concept explanation...")
            start_time = time.time()
            explanation = await explanation_agent.explain_concept(
                concept="backpropagation",
                context="neural network training",
                difficulty_level="intermediate"
            )
            end_time = time.time()
            
            print(f"   ✅ Concept explanation successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📝 Explanation: {len(explanation)} characters")
            
            # Test step-by-step breakdown
            print("   🔢 Testing step-by-step breakdown...")
            start_time = time.time()
            breakdown = await explanation_agent.break_down_steps(
                process="neural network training",
                steps_count=5
            )
            end_time = time.time()
            
            print(f"   ✅ Step breakdown successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📝 Breakdown: {len(breakdown)} characters")
            
            return True
            
        except Exception as e:
            print(f"❌ Explanation agent test failed: {e}")
            return False
    
    async def test_routing_agent(self):
        """Test routing agent with real API calls."""
        print("\n🔍 Testing Routing Agent")
        print("-" * 50)
        
        try:
            # Initialize agent
            routing_agent = PremiumRoutingAgent()
            
            # Test request routing
            print("   🚦 Testing request routing...")
            start_time = time.time()
            route_decision = await routing_agent.route_request(
                request_type="content_analysis",
                complexity="high",
                user_tier="premium"
            )
            end_time = time.time()
            
            print(f"   ✅ Request routing successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   🎯 Route decision: {route_decision}")
            
            # Test agent selection
            print("   🤖 Testing agent selection...")
            start_time = time.time()
            selected_agents = await routing_agent.select_agents(
                task_type="content_analysis",
                complexity_level="medium",
                available_agents=["expert", "context_builder", "analyzer"]
            )
            end_time = time.time()
            
            print(f"   ✅ Agent selection successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   🤖 Selected agents: {selected_agents}")
            
            return True
            
        except Exception as e:
            print(f"❌ Routing agent test failed: {e}")
            return False
    
    async def test_context_assembly_agent(self):
        """Test context assembly agent with real API calls."""
        print("\n🔍 Testing Context Assembly Agent")
        print("-" * 50)
        
        try:
            # Initialize agent
            llm_service = create_llm_service(provider="gemini")
            context_agent = ContextAssemblyAgent(llm_service)
            
            # Test context building
            print("   🧩 Testing context building...")
            start_time = time.time()
            context = await context_agent.build_context(
                query="Explain neural network training",
                max_context_length=1000,
                include_related_concepts=True
            )
            end_time = time.time()
            
            print(f"   ✅ Context building successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📏 Context length: {len(context)} characters")
            
            # Test context enrichment
            print("   🌟 Testing context enrichment...")
            start_time = time.time()
            enriched_context = await context_agent.enrich_context(
                base_context=context,
                enrichment_type="examples",
                target_audience="beginners"
            )
            end_time = time.time()
            
            print(f"   ✅ Context enrichment successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📏 Enriched context: {len(enriched_context)} characters")
            
            return True
            
        except Exception as e:
            print(f"❌ Context assembly agent test failed: {e}")
            return False
    
    async def test_premium_routing_agent(self):
        """Test premium routing agent with real API calls."""
        print("\n🔍 Testing Premium Routing Agent")
        print("-" * 50)
        
        try:
            # Initialize agent
            routing_agent = PremiumRoutingAgent()
            
            # Test request routing
            print("   🚦 Testing request routing...")
            start_time = time.time()
            route_decision = await routing_agent.route_request(
                request_type="content_enhancement",
                complexity="high",
                user_tier="premium"
            )
            end_time = time.time()
            
            print(f"   ✅ Request routing successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   🎯 Route decision: {route_decision}")
            
            # Test agent selection
            print("   🤖 Testing agent selection...")
            start_time = time.time()
            selected_agents = await routing_agent.select_agents(
                task_type="content_curation",
                complexity_level="medium",
                available_agents=["curator", "explainer", "context_builder"]
            )
            end_time = time.time()
            
            print(f"   ✅ Agent selection successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   🤖 Selected agents: {selected_agents}")
            
            return True
            
        except Exception as e:
            print(f"❌ Premium routing agent test failed: {e}")
            return False
    
    async def test_agent_orchestration(self):
        """Test agent orchestration workflow."""
        print("\n🔍 Testing Agent Orchestration")
        print("-" * 50)
        
        try:
            # Initialize all agents
            curator_agent = ContentCuratorAgent()
            explanation_agent = ExplanationAgent()
            context_agent = ContextAssemblyAgent()
            
            # Test multi-agent workflow
            print("   🔄 Testing multi-agent workflow...")
            start_time = time.time()
            
            # Step 1: Build context
            context = await context_agent.build_context(
                query="Explain machine learning concepts",
                max_context_length=800
            )
            
            # Step 2: Generate explanation
            explanation = await explanation_agent.explain_concept(
                concept="machine learning",
                context=context,
                difficulty_level="beginner"
            )
            
            # Step 3: Curate content
            curated_content = await curator_agent.curate_content(
                content=explanation,
                instruction="Make this more engaging",
                content_type="educational"
            )
            
            end_time = time.time()
            
            print(f"   ✅ Multi-agent workflow successful")
            print(f"   ⏱️  Total time: {end_time - start_time:.2f}s")
            print(f"   📝 Final content: {len(curated_content)} characters")
            
            return True
            
        except Exception as e:
            print(f"❌ Agent orchestration test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all premium agent tests."""
        print("🚀 Premium Agents Test Suite")
        print("=" * 60)
        
        tests = [
            ("Content Curator Agent", self.test_content_curator_agent),
            ("Explanation Agent", self.test_explanation_agent),
            ("Routing Agent", self.test_routing_agent),
            ("Context Assembly Agent", self.test_context_assembly_agent),
            ("Agent Orchestration", self.test_agent_orchestration)
        ]
        
        for test_name, test_func in tests:
            try:
                success = await test_func()
                self.test_results.append((test_name, success))
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                self.test_results.append((test_name, False))
        
        # Print summary
        print("\n📊 Premium Agents Test Results")
        print("-" * 40)
        passed = sum(1 for _, success in self.test_results if success)
        total = len(self.test_results)
        
        for test_name, success in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {status} {test_name}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        return passed == total


async def main():
    """Run premium agents tests."""
    tester = PremiumAgentsTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())

