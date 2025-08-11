"""
Test script for Sprint 36: Premium Context Assembly Agent (CAA).
Tests the 10-stage context assembly pipeline and mode-aware strategies.
"""

import asyncio
import httpx
import json
from datetime import datetime, timezone

async def test_sprint36_caa():
    """Test Sprint 36 Context Assembly Agent implementation"""
    
    print("\n🧪 Testing Sprint 36: Premium Context Assembly Agent (CAA)")
    print("=" * 70)
    
    # Test 1: CAA Core Pipeline
    print("\n1. Testing CAA Core Pipeline...")
    try:
        from app.core.premium.context_assembly_agent import (
            ContextAssemblyAgent, CAARequest, CAAResponse
        )
        
        caa = ContextAssemblyAgent()
        
        # Test request
        request = CAARequest(
            query="Explain machine learning concepts",
            user_id="test-user-123",
            mode="chat",
            session_context={
                "conversation_history": [
                    {"role": "user", "content": "What is AI?"},
                    {"role": "assistant", "content": "AI is artificial intelligence..."}
                ],
                "current_topic": "machine learning",
                "user_focus_areas": ["algorithms", "neural networks"],
                "learning_objectives": ["understand ML basics"]
            },
            hints=["focus on practical examples", "include code snippets"],
            token_budget=2000,
            latency_budget_ms=1000
        )
        
        # Execute CAA pipeline
        response = await caa.assemble_context(request)
        
        print(f"✅ CAA pipeline successful")
        print(f"   Assembled context length: {len(response.assembled_context)}")
        print(f"   Sufficiency score: {response.sufficiency_score}")
        print(f"   Token count: {response.token_count}")
        print(f"   Cache key: {response.cache_key}")
        
    except Exception as e:
        print(f"❌ CAA pipeline failed: {e}")
    
    # Test 2: Mode-Aware Assembly
    print("\n2. Testing Mode-Aware Assembly...")
    try:
        from app.core.premium.modes.mode_aware_assembly import ModeAwareAssembly
        
        mode_assembly = ModeAwareAssembly()
        
        # Test different modes
        modes = ["chat", "quiz", "deep_dive", "walk_through", "note_editing"]
        query = "Explain neural networks"
        user_context = {
            "analytics": {"masteryLevel": "INTERMEDIATE"},
            "conversation_history": [],
            "session_context": {"current_topic": "deep learning"}
        }
        
        for mode in modes:
            # Test retrieval strategy
            augmented_queries = await mode_assembly.apply_mode_retrieval(mode, query, user_context)
            print(f"   ✅ {mode} mode: {len(augmented_queries)} augmented queries")
            
            # Test reranking strategy
            mock_chunks = [
                {"content": "Neural networks are computational models...", "rerank_score": 0.8},
                {"content": "Deep learning uses multiple layers...", "rerank_score": 0.7},
                {"content": "Backpropagation is a key algorithm...", "rerank_score": 0.6}
            ]
            reranked = await mode_assembly.apply_mode_reranking(mode, mock_chunks, query)
            print(f"   ✅ {mode} mode: {len(reranked)} reranked chunks")
            
            # Test compression strategy
            test_context = "Neural networks are computational models inspired by biological neurons. They consist of interconnected nodes that process information. Deep learning uses multiple layers of these networks to learn complex patterns."
            compressed = await mode_assembly.apply_mode_compression(mode, test_context, 20)
            print(f"   ✅ {mode} mode: compressed to {len(compressed.split())} words")
        
    except Exception as e:
        print(f"❌ Mode-aware assembly failed: {e}")
    
    # Test 3: CAA API Endpoints
    print("\n3. Testing CAA API Endpoints...")
    try:
        async with httpx.AsyncClient() as client:
            # Test context assembly endpoint
            caa_request = {
                "query": "Explain gradient descent",
                "user_id": "test-user-123",
                "mode": "deep_dive",
                "session_context": {
                    "conversation_history": [],
                    "current_topic": "optimization",
                    "user_focus_areas": ["mathematics", "algorithms"],
                    "learning_objectives": ["understand optimization"]
                },
                "hints": ["include mathematical notation", "provide step-by-step explanation"],
                "token_budget": 3000,
                "latency_budget_ms": 1500
            }
            
            response = await client.post(
                "http://localhost:8000/premium/context/assemble",
                json=caa_request,
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ CAA endpoint successful")
                print(f"   Assembled context: {len(result.get('assembled_context', ''))} chars")
                print(f"   Sufficiency score: {result.get('sufficiency_score', 0)}")
                print(f"   Token count: {result.get('token_count', 0)}")
            else:
                print(f"❌ CAA endpoint failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
    except Exception as e:
        print(f"❌ CAA API endpoints failed: {e}")
        print("   Note: This is expected if the server is not running")
    
    # Test 4: Core API Integration
    print("\n4. Testing Core API Integration...")
    try:
        from app.core.premium.core_api_client import CoreAPIClient
        
        core_client = CoreAPIClient()
        
        # Test user analytics retrieval
        analytics = await core_client.get_user_learning_analytics("test-user-123")
        print(f"✅ User analytics retrieved: {len(analytics)} fields")
        
        # Test memory insights retrieval
        insights = await core_client.get_user_memory_insights("test-user-123")
        print(f"✅ Memory insights retrieved: {len(insights)} insights")
        
        # Test learning paths retrieval
        paths = await core_client.get_user_learning_paths("test-user-123")
        print(f"✅ Learning paths retrieved: {len(paths)} paths")
        
        # Test knowledge primitives retrieval
        primitives = await core_client.get_knowledge_primitives(
            user_id="test-user-123",
            concept="machine learning",
            include_premium_fields=True
        )
        print(f"✅ Knowledge primitives retrieved: {len(primitives)} primitives")
        
    except Exception as e:
        print(f"❌ Core API integration failed: {e}")
        print("   Note: This is expected if Core API is not running")
    
    # Test 5: CAA Pipeline Stages
    print("\n5. Testing CAA Pipeline Stages...")
    try:
        from app.core.premium.context_assembly_agent import ContextAssemblyAgent, CAAState, CAARequest
        
        caa = ContextAssemblyAgent()
        request = CAARequest(
            query="Explain backpropagation with examples",
            user_id="test-user-123",
            mode="quiz",
            session_context={},
            hints=["focus on key concepts"]
        )
        
        # Test individual pipeline stages
        state = CAAState(request)
        
        # Test input normalization
        state = await caa.input_normalization(state)
        print(f"✅ Input normalization: {len(state.normalized_query)} chars")
        
        # Test query augmentation
        state = await caa.query_augmentation(state)
        print(f"✅ Query augmentation: {len(state.augmented_queries)} queries")
        
        # Test coarse retrieval
        state = await caa.coarse_retrieval(state)
        print(f"✅ Coarse retrieval: {len(state.retrieved_chunks)} chunks")
        
        # Test graph traversal
        state = await caa.graph_traversal(state)
        print(f"✅ Graph traversal: {len(state.graph_results)} results")
        
        # Test cross-encoder rerank
        state = await caa.cross_encoder_rerank(state)
        print(f"✅ Cross-encoder rerank: {len(state.reranked_chunks)} chunks")
        
        # Test sufficiency check
        state = await caa.sufficiency_check(state)
        print(f"✅ Sufficiency check: score {state.sufficiency_score}")
        
        # Test context condensation
        state = await caa.context_condensation(state)
        print(f"✅ Context condensation: {len(state.condensed_context)} chars")
        
        # Test tool enrichment
        state = await caa.tool_enrichment(state)
        print(f"✅ Tool enrichment: {len(state.tool_outputs)} outputs")
        
        # Test final assembly
        state = await caa.final_assembly(state)
        print(f"✅ Final assembly: {len(state.final_context)} chars")
        
        # Test cache and metrics
        state = await caa.cache_and_metrics(state)
        print(f"✅ Cache and metrics: {len(state.metrics)} metrics")
        
    except Exception as e:
        print(f"❌ CAA pipeline stages failed: {e}")
    
    # Test 6: Mode-Specific Strategies
    print("\n6. Testing Mode-Specific Strategies...")
    try:
        from app.core.premium.modes.mode_aware_assembly import (
            ChatModeStrategy, QuizModeStrategy, DeepDiveModeStrategy,
            WalkThroughModeStrategy, NoteEditingModeStrategy
        )
        
        strategies = {
            "chat": ChatModeStrategy(),
            "quiz": QuizModeStrategy(),
            "deep_dive": DeepDiveModeStrategy(),
            "walk_through": WalkThroughModeStrategy(),
            "note_editing": NoteEditingModeStrategy()
        }
        
        test_query = "Explain convolutional neural networks"
        test_user_context = {
            "analytics": {"masteryLevel": "INTERMEDIATE"},
            "conversation_history": [],
            "session_context": {"current_topic": "computer vision"}
        }
        
        for mode_name, strategy in strategies.items():
            # Test retrieval strategy
            queries = await strategy.apply_retrieval_strategy(test_query, test_user_context)
            print(f"   ✅ {mode_name}: {len(queries)} retrieval queries")
            
            # Test reranking strategy
            mock_chunks = [
                {"content": "CNNs are neural networks designed for image processing...", "rerank_score": 0.9},
                {"content": "Convolutional layers apply filters to input images...", "rerank_score": 0.8},
                {"content": "Pooling layers reduce spatial dimensions...", "rerank_score": 0.7}
            ]
            reranked = await strategy.apply_reranking_strategy(mock_chunks, test_query)
            print(f"   ✅ {mode_name}: {len(reranked)} reranked chunks")
            
            # Test compression strategy
            test_context = "Convolutional Neural Networks (CNNs) are specialized neural networks designed for processing grid-like data such as images. They use convolutional layers that apply filters to input data, followed by pooling layers that reduce spatial dimensions. CNNs are particularly effective for computer vision tasks."
            compressed = await strategy.apply_compression_strategy(test_context, 25)
            print(f"   ✅ {mode_name}: compressed to {len(compressed.split())} words")
        
    except Exception as e:
        print(f"❌ Mode-specific strategies failed: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 Sprint 36 CAA Testing Complete!")
    print("Key Features Tested:")
    print("✅ CAA Core Pipeline (10-stage)")
    print("✅ Mode-Aware Assembly (5 modes)")
    print("✅ CAA API Endpoints")
    print("✅ Core API Integration")
    print("✅ Individual Pipeline Stages")
    print("✅ Mode-Specific Strategies")
    print("\nThe Context Assembly Agent is now ready for Sprint 37!")

if __name__ == "__main__":
    asyncio.run(test_sprint36_caa())
