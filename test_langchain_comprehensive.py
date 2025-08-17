#!/usr/bin/env python3
"""
Comprehensive test script for the LangChain-based Blueprint Editing Agent.
This tests all the major functionality with real LLM calls.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import the agent
from app.core.premium.agents.blueprint_editing_agent import BlueprintEditingAgent

class LangChainComprehensiveTester:
    """Comprehensive test class for the LangChain-based blueprint editing agent."""
    
    def __init__(self):
        """Initialize the tester with the LangChain agent."""
        print("🚀 Initializing LangChain Blueprint Editing Agent Tester...")
        
        try:
            self.agent = BlueprintEditingAgent()
            print("✅ LangChain agent initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize LangChain agent: {e}")
            raise
    
    async def test_blueprint_editing(self) -> bool:
        """Test blueprint editing with LangChain agent."""
        print("\n📝 Test 1: Editing Blueprint for Clarity Improvement")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Mock blueprint data for testing
            blueprint_id = "test_blueprint_001"
            user_id = "test_user_001"
            edit_instruction = "Make the content clearer and more engaging for beginners"
            
            result = await self.agent.edit_content_agentically(
                content_id=blueprint_id,
                content_type="blueprint",
                edit_instruction=edit_instruction,
                user_id=user_id,
                edit_type="improve_clarity"
            )
            
            elapsed_time = time.time() - start_time
            
            if result["success"]:
                print(f"✅ Blueprint editing completed in {elapsed_time:.2f}s")
                print(f"📊 Success: {result['success']}")
                
                # Show original vs edited content
                print(f"\n📝 EDIT RESULT:")
                print(f"   Service Used: Gemini 2.5 Flash (direct)")
                print(f"   Original Content: {json.dumps(result['original_content'], indent=2)}")
                print(f"   Edited Content: {result['edited_content']}")
                print(f"   Original content length: {len(str(result['original_content']))} characters")
                print(f"   Edited content length: {len(str(result['edited_content']))} characters")
                
                return True
            else:
                print(f"❌ Blueprint editing failed in {elapsed_time:.2f}s")
                print(f"📊 Success: {result['success']}")
                print(f"💬 Message: Error editing blueprint: {result['error']}")
                return False
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ Blueprint editing failed in {elapsed_time:.2f}s")
            print(f"💬 Message: Exception during blueprint editing: {str(e)}")
            return False
    
    async def test_primitive_editing(self) -> bool:
        """Test primitive editing with LangChain agent."""
        print("\n🔬 Test 2: Editing Knowledge Primitive for Clarity")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            primitive_id = "test_primitive_001"
            user_id = "test_user_001"
            edit_instruction = "Make the concept explanation clearer and add more examples"
            
            result = await self.agent.edit_content_agentically(
                content_id=primitive_id,
                content_type="primitive",
                edit_instruction=edit_instruction,
                user_id=user_id,
                edit_type="improve_clarity"
            )
            
            elapsed_time = time.time() - start_time
            
            if result["success"]:
                print(f"✅ Primitive editing completed in {elapsed_time:.2f}s")
                print(f"📊 Success: {result['success']}")
                
                # Show original vs edited content
                print(f"\n📝 EDIT RESULT:")
                print(f"   Service Used: Gemini 2.5 Flash (direct)")
                print(f"   Original Content: {json.dumps(result['original_content'], indent=2)}")
                print(f"   Edited Content: {result['edited_content']}")
                print(f"   Original content length: {len(str(result['original_content']))} characters")
                print(f"   Edited content length: {len(str(result['edited_content']))} characters")
                
                return True
            else:
                print(f"❌ Primitive editing failed in {elapsed_time:.2f}s")
                print(f"📊 Success: {result['success']}")
                print(f"💬 Message: Error editing primitive: {result['error']}")
                return False
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ Primitive editing failed in {elapsed_time:.2f}s")
            print(f"💬 Message: Exception during primitive editing: {str(e)}")
            return False
    
    async def test_suggestions_generation(self) -> bool:
        """Test editing suggestions generation with LangChain agent."""
        print("\n💡 Test 3: Generating Editing Suggestions")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            content_id = "test_content_001"
            user_id = "test_user_001"
            
            result = await self.agent.get_editing_suggestions_agentically(
                content_id=content_id,
                content_type="blueprint",
                user_id=user_id,
                suggestion_type="general"
            )
            
            elapsed_time = time.time() - start_time
            
            if result["success"]:
                print(f"✅ Suggestions generated in {elapsed_time:.2f}s")
                print(f"📊 Success: {result['success']}")
                print(f"💡 Service Used: Gemini 2.5 Flash (direct)")
                print(f"💡 Suggestions: {result['suggestions']}")
                return True
            else:
                print(f"❌ Suggestions generation failed in {elapsed_time:.2f}s")
                print(f"📊 Success: {result['success']}")
                print(f"💬 Message: Error generating suggestions: {result['error']}")
                return False
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ Suggestions generation failed in {elapsed_time:.2f}s")
            print(f"💬 Message: Exception during suggestions generation: {str(e)}")
            return False
    
    async def test_granular_editing(self) -> bool:
        """Test granular editing operations with LangChain agent."""
        print("\n🔧 Test 4: Executing Granular Edit Operations")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Test adding a new section
            content = "Sample blueprint content with sections"
            parameters = {"section_name": "Advanced Topics", "content": "Advanced concepts and applications"}
            
            result = await self.agent.execute_granular_edit(
                edit_type="add_section",
                content=content,
                parameters=parameters
            )
            
            elapsed_time = time.time() - start_time
            
            if "successfully" in result.lower():
                print(f"✅ Granular edit completed in {elapsed_time:.2f}s")
                print(f"🔧 Service Used: Gemini 2.5 Flash (direct)")
                print(f"🔧 Result: {result}")
                return True
            else:
                print(f"❌ Granular edit failed in {elapsed_time:.2f}s")
                print(f"🔧 Result: {result}")
                return False
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ Granular edit failed in {elapsed_time:.2f}s")
            print(f"💬 Message: Exception during granular edit: {str(e)}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        print("🧪 LangChain Blueprint Editing Agent Comprehensive Test Suite")
        print("=" * 70)
        print("🚀 Starting LangChain-based editing tests...")
        print("⚠️  This will make actual API calls to Gemini 2.5 Flash")
        print("💰 Be aware of potential costs and rate limits")
        print("=" * 70)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "total_time": 0,
            "test_results": {}
        }
        
        start_time = time.time()
        
        # Test 1: Blueprint editing
        test_result = await self.test_blueprint_editing()
        results["test_results"]["blueprint_editing"] = test_result
        results["tests_run"] += 1
        if test_result:
            results["tests_passed"] += 1
        else:
            results["tests_failed"] += 1
        
        # Test 2: Primitive editing
        test_result = await self.test_primitive_editing()
        results["test_results"]["primitive_editing"] = test_result
        results["tests_run"] += 1
        if test_result:
            results["tests_passed"] += 1
        else:
            results["tests_failed"] += 1
        
        # Test 3: Suggestions generation
        test_result = await self.test_suggestions_generation()
        results["test_results"]["suggestions_generation"] = test_result
        results["tests_run"] += 1
        if test_result:
            results["tests_passed"] += 1
        else:
            results["tests_failed"] += 1
        
        # Test 4: Granular editing
        test_result = await self.test_granular_editing()
        results["test_results"]["granular_editing"] = test_result
        results["tests_run"] += 1
        if test_result:
            results["tests_passed"] += 1
        else:
            results["tests_failed"] += 1
        
        results["total_time"] = time.time() - start_time
        
        # Print summary
        print("\n" + "=" * 70)
        if results["tests_failed"] == 0:
            print("🎉 ALL LANGCHAIN TESTS PASSED!")
        else:
            print(f"❌ SOME LANGCHAIN TESTS FAILED!")
        print("=" * 70)
        print(f"📊 Test Results: {results['tests_passed']}/{results['tests_run']} tests passed")
        print(f"⏱️  Total Time: {results['total_time']:.2f}s")
        
        if results["tests_failed"] > 0:
            print("❌ Failed tests - check error messages above")
        else:
            print("✅ All tests completed successfully")
        
        return results

async def main():
    """Main test execution function."""
    try:
        tester = LangChainComprehensiveTester()
        results = await tester.run_all_tests()
        
        # Save results to file
        with open("langchain_comprehensive_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Test results saved to: langchain_comprehensive_test_results.json")
        
    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
