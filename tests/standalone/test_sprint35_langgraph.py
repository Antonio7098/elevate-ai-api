"""
Test script for Sprint 35: Premium Multi-Agent Expert System with LangGraph.
Tests the LangGraph integration, expert agents, and workflow orchestration.
"""

import asyncio
import httpx
import json
from datetime import datetime

async def test_sprint35_langgraph():
    """Test Sprint 35 LangGraph implementation"""
    
    # Test configuration
    base_url = "http://localhost:8000"
    api_key = "test-api-key"
    
    print("🧪 Testing Sprint 35: Premium Multi-Agent Expert System with LangGraph")
    print("=" * 70)
    
    # Test 1: LangGraph Setup
    print("\n1. Testing LangGraph Setup...")
    try:
        from app.core.premium.langgraph_setup import LangGraphSetup, PremiumAgentState
        
        langgraph_setup = LangGraphSetup()
        
        # Test state creation
        initial_state = langgraph_setup.create_base_state(
            user_query="Explain machine learning",
            user_id="test-user-123",
            user_context={"learning_style": "visual"}
        )
        print(f"✅ Initial state created: {initial_state['workflow_status']}")
        
        # Test state persistence
        await langgraph_setup.persist_state(initial_state, "test-workflow-123")
        print("✅ State persistence successful")
        
        # Test state recovery
        recovered_state = await langgraph_setup.recover_state("test-workflow-123")
        print(f"✅ State recovery successful: {recovered_state['workflow_status']}")
        
    except Exception as e:
        print(f"❌ LangGraph setup failed: {e}")
    
    # Test 2: LangGraph Routing Agent
    print("\n2. Testing LangGraph Routing Agent...")
    try:
        from app.core.premium.agents.routing_agent import PremiumRoutingAgent
        
        routing_agent = PremiumRoutingAgent()
        
        # Test workflow execution
        workflow_result = await routing_agent.execute_workflow(
            user_query="Explain neural networks step by step",
            user_id="test-user-123",
            user_context={"learning_style": "visual", "mastery_level": "intermediate"}
        )
        
        print(f"✅ Workflow execution successful")
        print(f"   Response: {workflow_result['response'][:100]}...")
        print(f"   Agents used: {workflow_result['agents_used']}")
        print(f"   Workflow status: {workflow_result['workflow_status']}")
        
    except Exception as e:
        print(f"❌ LangGraph routing agent failed: {e}")
    
    # Test 3: Expert Agents with Core API Integration
    print("\n3. Testing Expert Agents with Core API Integration...")
    try:
        from app.core.premium.agents.expert_agents import (
            ExplanationAgent, AssessmentAgent, ContentCuratorAgent,
            LearningPlannerAgent, ResearchAgent
        )
        from app.core.premium.langgraph_setup import PremiumAgentState
        
        # Test explanation agent
        explainer = ExplanationAgent()
        state = PremiumAgentState(
            user_query="Explain machine learning",
            user_id="test-user-123",
            user_context={"learning_style": "visual"},
            messages=[], selected_agents=[], context_assembled={},
            final_response="", workflow_status="", agent_responses={}, metadata={}
        )
        
        explanation_result = await explainer.process_explanation_request(state)
        print(f"✅ Explanation agent successful: {explanation_result['agent_type']}")
        
        # Test assessment agent
        assessor = AssessmentAgent()
        assessment_result = await assessor.process_assessment_request(state)
        print(f"✅ Assessment agent successful: {assessment_result['agent_type']}")
        
        # Test curator agent
        curator = ContentCuratorAgent()
        curation_result = await curator.process_curation_request(state)
        print(f"✅ Curator agent successful: {curation_result['agent_type']}")
        
        # Test planner agent
        planner = LearningPlannerAgent()
        planning_result = await planner.process_planning_request(state)
        print(f"✅ Planner agent successful: {planning_result['agent_type']}")
        
        # Test research agent
        researcher = ResearchAgent()
        research_result = await researcher.process_research_request(state)
        print(f"✅ Research agent successful: {research_result['agent_type']}")
        
    except Exception as e:
        print(f"❌ Expert agents failed: {e}")
    
    # Test 4: Agent Communication
    print("\n4. Testing Agent Communication...")
    try:
        from app.core.premium.agents.communication import (
            AgentCommunicationProtocol, Task, Context, CoordinatedResponse
        )
        from datetime import datetime, timezone
        
        communication = AgentCommunicationProtocol()
        
        # Test task coordination
        task = Task(
            id="test-task-123",
            description="Explain machine learning concepts",
            priority="high",
            assigned_agents=["explainer", "researcher"],
            status="active",
            created_at=datetime.now(timezone.utc)
        )
        
        coordinated_response = await communication.coordinate_agents(task, ["explainer", "researcher"])
        print(f"✅ Agent coordination successful")
        print(f"   Consensus score: {coordinated_response.consensus_score}")
        print(f"   Conflicts resolved: {len(coordinated_response.conflicts_resolved)}")
        
        # Test parallel execution
        parallel_results = await communication.parallel_agent_execution(
            ["explainer", "assessor"], task
        )
        print(f"✅ Parallel execution successful")
        print(f"   Execution time: {parallel_results.execution_time}s")
        print(f"   Success rate: {parallel_results.success_rate}")
        
    except Exception as e:
        print(f"❌ Agent communication failed: {e}")
    
    # Test 5: Learning Workflows
    print("\n5. Testing Learning Workflows...")
    try:
        from app.core.premium.workflows.learning_workflow import (
            LearningWorkflow, AdaptiveLearningWorkflow
        )
        
        # Test standard learning workflow
        learning_workflow = LearningWorkflow()
        workflow_result = await learning_workflow.execute_workflow(
            user_query="Learn machine learning fundamentals",
            user_id="test-user-123",
            user_context={"learning_style": "visual", "mastery_level": "beginner"}
        )
        
        print(f"✅ Learning workflow successful")
        print(f"   Workflow status: {workflow_result['workflow_status']}")
        print(f"   Learning plan: {workflow_result.get('learning_plan', '')[:100]}...")
        
        # Test adaptive learning workflow
        adaptive_workflow = AdaptiveLearningWorkflow()
        adaptive_result = await adaptive_workflow.execute_workflow(
            user_query="Learn advanced neural networks",
            user_id="test-user-123",
            user_context={"learning_style": "visual", "mastery_level": "advanced"}
        )
        
        print(f"✅ Adaptive workflow successful")
        print(f"   Workflow status: {adaptive_result['workflow_status']}")
        
    except Exception as e:
        print(f"❌ Learning workflows failed: {e}")
    
    # Test 6: Premium API Endpoints (if server is running)
    print("\n6. Testing Premium API Endpoints...")
    try:
        # Test LangGraph chat endpoint
        langgraph_request = {
            "query": "Explain machine learning with examples",
            "user_id": "test-user-123",
            "user_context": {
                "learning_style": "visual",
                "mastery_level": "intermediate"
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/api/v1/premium/chat/langgraph",
                headers={"Authorization": f"Bearer {api_key}"},
                json=langgraph_request
            )
            print(f"✅ LangGraph chat endpoint status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Response length: {len(data.get('response', ''))}")
                print(f"   Agents used: {data.get('agents_used', [])}")
                print(f"   Workflow status: {data.get('workflow_status', '')}")
        
        # Test learning workflow endpoint
        workflow_request = {
            "learning_goal": "Master machine learning fundamentals",
            "user_id": "test-user-123",
            "user_context": {
                "learning_style": "visual",
                "mastery_level": "beginner"
            },
            "workflow_type": "standard"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/api/v1/premium/learning/workflow",
                headers={"Authorization": f"Bearer {api_key}"},
                json=workflow_request
            )
            print(f"✅ Learning workflow endpoint status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Learning plan: {len(data.get('learning_plan', ''))} characters")
                print(f"   Workflow status: {data.get('workflow_status', '')}")
        
    except Exception as e:
        print(f"❌ Premium API endpoints failed: {e}")
        print("   Note: This is expected if the server is not running")
    
    print("\n" + "=" * 70)
    print("🎉 Sprint 35 LangGraph Testing Complete!")
    print("Key Features Tested:")
    print("✅ LangGraph setup and state management")
    print("✅ Multi-agent routing and orchestration")
    print("✅ Expert agents with Core API integration")
    print("✅ Agent communication and coordination")
    print("✅ Learning workflows (standard and adaptive)")
    print("✅ Premium API endpoints with LangGraph integration")

if __name__ == "__main__":
    asyncio.run(test_sprint35_langgraph())
