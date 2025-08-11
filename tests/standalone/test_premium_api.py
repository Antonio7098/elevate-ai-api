"""
Test script for Premium API implementation (Sprint 34).
Tests the basic functionality of premium endpoints and services.
"""

import asyncio
import httpx
import json

async def test_premium_api():
    """Test premium API endpoints"""
    
    # Test configuration
    base_url = "http://localhost:8000"
    api_key = "test-api-key"
    
    print("🧪 Testing Premium API Implementation (Sprint 34)")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing Premium Health Check...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url}/api/v1/premium/health",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            print(f"✅ Health check status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Services: {data.get('services', {})}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test 2: Premium chat endpoint
    print("\n2. Testing Premium Chat Endpoint...")
    try:
        chat_request = {
            "query": "Can you explain machine learning concepts?",
            "user_id": "test-user-123",
            "user_context": {
                "learning_style": "visual",
                "mastery_level": "intermediate"
            },
            "mode": "chat",
            "max_tokens": 500
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/api/v1/premium/chat/advanced",
                headers={"Authorization": f"Bearer {api_key}"},
                json=chat_request
            )
            print(f"✅ Chat endpoint status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Response length: {len(data.get('response', ''))}")
                print(f"   Experts used: {data.get('experts_used', [])}")
                print(f"   Confidence: {data.get('confidence_score', 0)}")
    except Exception as e:
        print(f"❌ Chat endpoint failed: {e}")
    
    # Test 3: Graph search endpoint
    print("\n3. Testing Graph Search Endpoint...")
    try:
        graph_request = {
            "query": "machine learning neural networks",
            "user_id": "test-user-123",
            "depth": 2,
            "max_results": 5
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/api/v1/premium/chat/graph-search",
                headers={"Authorization": f"Bearer {api_key}"},
                json=graph_request
            )
            print(f"✅ Graph search status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Results count: {len(data.get('results', []))}")
                print(f"   Query: {data.get('query', '')}")
    except Exception as e:
        print(f"❌ Graph search failed: {e}")
    
    # Test 4: Core API integration
    print("\n4. Testing Core API Integration...")
    try:
        from app.core.premium.core_api_client import CoreAPIClient
        
        client = CoreAPIClient()
        
        # Test user memory retrieval
        memory = await client.get_user_memory("test-user-123")
        print(f"✅ User memory retrieved: {memory.get('cognitiveApproach', 'N/A')}")
        
        # Test learning analytics
        analytics = await client.get_user_learning_analytics("test-user-123")
        print(f"✅ Learning analytics retrieved: {analytics.get('learningEfficiency', 0)}")
        
        # Test memory insights
        insights = await client.get_user_memory_insights("test-user-123")
        print(f"✅ Memory insights retrieved: {len(insights)} insights")
        
    except Exception as e:
        print(f"❌ Core API integration failed: {e}")
    
    # Test 5: Memory system
    print("\n5. Testing Memory System...")
    try:
        from app.core.premium.memory_system import PremiumMemorySystem
        
        memory_system = PremiumMemorySystem()
        
        # Test memory retrieval
        memories = await memory_system.retrieve_with_attention("machine learning")
        print(f"✅ Memory retrieval successful: {len(memories.get('concepts', []))} concepts")
        
        # Test memory stats
        stats = await memory_system.get_memory_stats()
        print(f"✅ Memory stats: {stats}")
        
    except Exception as e:
        print(f"❌ Memory system failed: {e}")
    
    # Test 6: Routing agent
    print("\n6. Testing Routing Agent...")
    try:
        from app.core.premium.routing_agent import PremiumRoutingAgent
        
        routing_agent = PremiumRoutingAgent()
        
        # Test query routing
        routing_result = await routing_agent.route_query(
            "Explain neural networks step by step",
            {"user_id": "test-user-123", "learning_style": "visual"}
        )
        print(f"✅ Query routing successful: {routing_result.get('experts', [])}")
        
        # Test expert orchestration
        orchestration_result = await routing_agent.orchestrate_experts(
            "Explain neural networks step by step",
            ["explainer"]
        )
        print(f"✅ Expert orchestration successful: {orchestration_result.get('experts_used', [])}")
        
    except Exception as e:
        print(f"❌ Routing agent failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Premium API Testing Complete!")
    print("Note: Some tests may fail if services are not running.")
    print("This is expected for development testing.")

if __name__ == "__main__":
    asyncio.run(test_premium_api())










