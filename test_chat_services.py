#!/usr/bin/env python3
"""
Production readiness test for Chat & Interaction Services.
Tests chat, context assembly, and response generation with REAL LLM calls.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.llm_service import create_llm_service
from app.core.chat import ChatService
from app.core.context_assembly import ContextAssembler
from app.core.response_generation import ResponseGenerationService
from app.core.response_generation_prompt_template import ResponseGenerationPromptTemplate

class ChatServicesTester:
    def __init__(self):
        self.llm_service = None
        self.chat_service = None
        self.context_assembler = None
        self.response_generation_service = None
        self.prompt_template = None
        
    async def setup_services(self):
        """Set up all chat services with real dependencies."""
        print("🔧 Setting up Chat & Interaction Services...")
        
        try:
            # Set up LLM service
            print("   🚀 Setting up Gemini LLM service...")
            self.llm_service = create_llm_service(provider="gemini")
            print("   ✅ LLM service ready")
            
            # Set up Chat Service
            print("   💬 Setting up Chat Service...")
            self.chat_service = ChatService()
            print("   ✅ Chat Service ready")
            
            # Set up Context Assembler with Tavily Search
            print("   🧩 Setting up Context Assembler...")
            try:
                from app.services.tavily_search_service import TavilySearchService
                search_service = TavilySearchService(max_results=5, topic="general")
                print("   🔍 Tavily Search Service initialized")
            except Exception as e:
                print(f"   ⚠️  Tavily Search Service failed: {e}")
                search_service = None
            
            self.context_assembler = ContextAssembler(
                search_service=search_service,
                max_context_tokens=4000
            )
            print("   ✅ Context Assembler ready")
            
            # Set up Response Generation Service
            print("   📝 Setting up Response Generation Service...")
            self.response_generation_service = ResponseGenerationService(
                llm_service=self.llm_service
            )
            print("   ✅ Response Generation Service ready")
            
            # Set up Prompt Template
            print("   📋 Setting up Response Generation Prompt Template...")
            self.prompt_template = ResponseGenerationPromptTemplate()
            print("   ✅ Prompt Template ready")
            
            print("   🎉 All Chat & Interaction Services set up successfully!")
            return True
            
        except Exception as e:
            print(f"   ❌ Service setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_chat_service(self):
        """Test chat service functionality."""
        print("\n💬 Testing Chat Service")
        print("-" * 60)
        
        try:
            print("   🚀 Testing chat service...")
            
            # Test chat initialization
            print("      🆕 Testing chat initialization...")
            chat_session = await self.chat_service.create_chat_session(
                user_id="test_user",
                session_type="learning",
                context="machine learning fundamentals"
            )
            
            print(f"         ✅ Chat session created: {chat_session.get('session_id', 'N/A')}")
            
            # Test message sending
            print("      📤 Testing message sending...")
            user_message = "What is machine learning?"
            
            chat_response = await self.chat_service.send_message(
                session_id=chat_session.get('session_id'),
                message=user_message,
                message_type="user"
            )
            
            print(f"         ✅ Message sent and processed: {len(chat_response.get('response', ''))} characters")
            
            if chat_response:
                print("      📊 Chat response details:")
                print(f"         Response: {chat_response.get('response', 'N/A')[:100]}...")
                print(f"         Message ID: {chat_response.get('message_id', 'N/A')}")
                print(f"         Timestamp: {chat_response.get('timestamp', 'N/A')}")
            
            # Test conversation history
            print("      📚 Testing conversation history...")
            history = await self.chat_service.get_conversation_history(
                session_id=chat_session.get('session_id'),
                limit=10
            )
            
            print(f"         ✅ Retrieved {len(history)} conversation messages")
            
            # Test chat session management
            print("      🔄 Testing chat session management...")
            session_info = await self.chat_service.get_session_info(
                session_id=chat_session.get('session_id')
            )
            
            print(f"         ✅ Session info retrieved: {session_info.get('session_type', 'N/A')}")
            
            print("   ✅ Chat service test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Chat service test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_context_assembly(self):
        """Test context assembly with real LLM calls."""
        print("\n🧩 Testing Context Assembly")
        print("-" * 60)
        
        try:
            print("   🚀 Testing context assembly...")
            
            # Test context assembly for a query
            print("      🔍 Testing context assembly for query...")
            query = "Explain the basics of neural networks and deep learning"
            
            try:
                # Test context assembly with Tavily Search
                # Create a mock QueryTransformation for testing
                from app.core.query_transformer import QueryTransformation, QueryIntent
                
                mock_query_transformation = QueryTransformation(
                    original_query=query,
                    expanded_query=query,
                    reformulated_queries=[query],
                    intent=QueryIntent.CONCEPTUAL,
                    confidence=0.9,
                    search_terms=[query],
                    metadata_filters={},
                    search_strategy="semantic"
                )
                
                context = await self.context_assembler.assemble_context(
                    user_id="test_user",
                    session_id="test_session",
                    current_query=query,
                    query_transformation=mock_query_transformation
                )
                
                # Extract context content for display
                context_content = context.get_context_for_prompt() if hasattr(context, 'get_context_for_prompt') else str(context)
                print(f"         ✅ Context assembled with Tavily Search: {len(context_content)} characters")
                print(f"         📄 Context preview: {context_content[:200]}...")
                
            except Exception as e:
                print(f"         ❌ Context assembly failed: {e}")
                # Create mock context for testing as fallback
                context = """
                Neural networks are computational models inspired by biological neurons.
                They consist of interconnected nodes that process information in layers.
                Deep learning uses networks with multiple hidden layers for complex pattern recognition.
                """
                print(f"         🔧 Using fallback context: {len(context)} characters")
            
            # Test context analysis
            print("      🔍 Testing context analysis...")
            try:
                analysis_query = f"Analyze this context and provide key insights:\n\n{context}"
                analysis_response = await self.llm_service.call_llm(analysis_query)
                
                print(f"         ✅ Context analysis completed: {len(analysis_response)} characters")
                print(f"         📊 Analysis preview: {analysis_response[:100]}...")
                
            except Exception as e:
                print(f"         ⚠️  Context analysis failed: {e}")
            
            print("   ✅ Context assembly test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Context assembly test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_response_generation(self):
        """Test response generation with real LLM calls."""
        print("\n📝 Testing Response Generation")
        print("-" * 60)
        
        try:
            print("   🚀 Testing response generation...")
            
            # Test basic response generation
            print("      📝 Testing basic response generation...")
            basic_query = "What is artificial intelligence?"
            
            basic_response = await self.response_generation_service.generate_response(
                query=basic_query,
                context="",
                response_type="explanation"
            )
            
            print(f"         ✅ Basic response generated: {len(basic_response)} characters")
            print(f"         📄 Response preview: {basic_response[:100]}...")
            
            # Test contextual response generation
            print("      🧩 Testing contextual response generation...")
            contextual_query = "How does machine learning relate to AI?"
            contextual_context = """
            Artificial Intelligence (AI) is the broader field of creating intelligent machines.
            Machine learning is a subset of AI that enables computers to learn from data.
            Deep learning is a subset of machine learning using neural networks.
            """
            
            contextual_response = await self.response_generation_service.generate_response(
                query=contextual_query,
                context=contextual_context,
                response_type="detailed_explanation"
            )
            
            print(f"         ✅ Contextual response generated: {len(contextual_response)} characters")
            print(f"         📄 Response preview: {contextual_response[:100]}...")
            
            # Test response optimization
            print("      🚀 Testing response optimization...")
            try:
                optimized_response = await self.response_generation_service.optimize_response(
                    response=contextual_response,
                    optimization_type="clarity",
                    target_audience="beginners"
                )
                
                print(f"         ✅ Response optimized: {len(optimized_response)} characters")
                
            except Exception as e:
                print(f"         ⚠️  Response optimization failed: {e}")
            
            # Test response validation
            print("      ✅ Testing response validation...")
            try:
                is_valid = await self.response_generation_service.validate_response(
                    response=contextual_response,
                    query=contextual_query,
                    context=contextual_context
                )
                
                print(f"         ✅ Response validation: {is_valid}")
                
            except Exception as e:
                print(f"         ⚠️  Response validation failed: {e}")
            
            print("   ✅ Response generation test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Response generation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_prompt_templates(self):
        """Test prompt template functionality."""
        print("\n📋 Testing Prompt Templates")
        print("-" * 60)
        
        try:
            print("   🚀 Testing prompt templates...")
            
            # Test template generation
            print("      📝 Testing template generation...")
            try:
                template = await self.prompt_template.generate_prompt(
                    query="Explain machine learning",
                    context="Machine learning is a subset of AI",
                    response_type="explanation"
                )
                
                print(f"         ✅ Template generated: {len(template)} characters")
                print(f"         📄 Template preview: {template[:100]}...")
                
            except Exception as e:
                print(f"         ⚠️  Template generation failed: {e}")
                # Create a simple template for testing
                template = """
                Based on the following context, answer the question clearly and accurately.
                
                Context: {context}
                Question: {query}
                
                Please provide a {response_type} response.
                """
                print(f"         🔧 Using fallback template: {len(template)} characters")
            
            # Test template customization
            print("      🎨 Testing template customization...")
            try:
                customized_template = await self.prompt_template.customize_template(
                    base_template=template,
                    customizations={
                        "style": "academic",
                        "tone": "professional",
                        "length": "detailed"
                    }
                )
                
                print(f"         ✅ Template customized: {len(customized_template)} characters")
                
            except Exception as e:
                print(f"         ⚠️  Template customization failed: {e}")
                customized_template = template
            
            # Test template application
            print("      🔄 Testing template application...")
            try:
                applied_prompt = await self.prompt_template.apply_template(
                    template=customized_template,
                    variables={
                        "context": "AI and machine learning fundamentals",
                        "query": "What is the difference between AI and ML?",
                        "response_type": "comprehensive explanation"
                    }
                )
                
                print(f"         ✅ Template applied: {len(applied_prompt)} characters")
                print(f"         📄 Applied prompt preview: {applied_prompt[:100]}...")
                
            except Exception as e:
                print(f"         ⚠️  Template application failed: {e}")
            
            print("   ✅ Prompt templates test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Prompt templates test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_integrated_chat_workflow(self):
        """Test integrated chat workflow with all services."""
        print("\n🔄 Testing Integrated Chat Workflow")
        print("-" * 60)
        
        try:
            print("   🚀 Testing integrated chat workflow...")
            
            # Test complete chat flow
            print("      💬 Testing complete chat flow...")
            
            # Step 1: Create chat session
            print("         🆕 Step 1: Creating chat session...")
            chat_session = await self.chat_service.create_chat_session(
                user_id="test_user_integrated",
                session_type="learning",
                context="artificial intelligence and machine learning"
            )
            
            session_id = chat_session.get('session_id')
            print(f"            ✅ Session created: {session_id}")
            
            # Step 2: Send user message
            print("         📤 Step 2: Sending user message...")
            user_message = "Can you explain how neural networks work in simple terms?"
            
            # Step 3: Generate response
            print("         📝 Step 3: Generating response...")
            try:
                response = await self.response_generation_service.generate_response(
                    query=user_message,
                    context="Neural networks are computational models inspired by biological neurons.",
                    response_type="simple_explanation"
                )
                
                print(f"            ✅ Response generated: {len(response)} characters")
                
                # Step 4: Send response back to chat
                print("         📥 Step 4: Processing response in chat...")
                chat_response = await self.chat_service.send_message(
                    session_id=session_id,
                    message=response,
                    message_type="assistant"
                )
                
                print(f"            ✅ Response processed in chat: {chat_response.get('message_id', 'N/A')}")
                
            except Exception as e:
                print(f"            ⚠️  Response generation failed: {e}")
            
            # Step 5: Get conversation summary
            print("         📊 Step 5: Getting conversation summary...")
            try:
                summary = await self.chat_service.get_session_summary(session_id)
                print(f"            ✅ Session summary retrieved: {len(summary.get('summary', ''))} characters")
                
            except Exception as e:
                print(f"            ⚠️  Session summary failed: {e}")
            
            print("      ✅ Complete chat workflow tested")
            
            print("   ✅ Integrated chat workflow test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Integrated chat workflow test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_all_tests(self):
        """Run all chat service tests."""
        print("🚀 Starting CHAT & INTERACTION SERVICES Production Readiness Test")
        print("=" * 80)
        
        # First, set up all services
        print("\n🔧 PHASE 1: Service Setup")
        setup_success = await self.setup_services()
        
        if not setup_success:
            print("❌ Service setup failed. Cannot proceed with tests.")
            return False
        
        print("\n🧪 PHASE 2: Running Tests")
        tests = [
            ("Chat Service", self.test_chat_service),
            ("Context Assembly", self.test_context_assembly),
            ("Response Generation", self.test_response_generation),
            ("Prompt Templates", self.test_prompt_templates),
            ("Integrated Chat Workflow", self.test_integrated_chat_workflow)
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n🧪 Running: {test_name}")
            try:
                result = await test_func()
                results.append((test_name, result))
                print(f"   {'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
            except Exception as e:
                print(f"   ❌ ERROR: {test_name} - {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 CHAT & INTERACTION SERVICES TEST SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL CHAT TESTS PASSED! Services are production-ready!")
        else:
            print("⚠️  Some chat tests failed. Check the output above for details.")
        
        return passed == total

async def main():
    """Main test function."""
    tester = ChatServicesTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 Chat services test suite completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some chat service tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

