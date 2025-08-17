#!/usr/bin/env python3
"""
Simple Real LLM Test for Sequential Generation Workflow
Uses Gemini 2.5 Flash service with a shorter source text to test the workflow.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Add the app directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.generation_orchestrator import GenerationOrchestrator, GenerationStep, GenerationStatus
from app.services.llm_service import GeminiLLMService

class SimpleLLMTest:
    """Simple test class for running the sequential workflow with real LLM calls."""
    
    def __init__(self):
        """Initialize the test workflow with real services."""
        self.orchestrator = GenerationOrchestrator()
        self.llm_service = GeminiLLMService()
        
        # Much shorter test content to avoid timeouts
        self.test_source = """
        Python Basics
        
        Python is a high-level programming language known for its simplicity and readability.
        
        Key concepts:
        - Variables store data values
        - Functions are reusable code blocks
        - Loops repeat code execution
        - Lists store multiple items
        """
        
        self.session_id = f"simple_test_{int(time.time())}"
        
    async def test_basic_workflow(self) -> Dict[str, Any]:
        """Test the basic workflow with a simple source."""
        print("🚀 Starting Simple Real LLM Test")
        print("=" * 50)
        
        results = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "steps_completed": [],
            "errors": [],
            "total_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Start generation session
            print("\n📝 Step 1: Starting Generation Session")
            print("-" * 30)
            
            user_preferences = {
                "difficulty_level": "beginner",
                "target_audience": "students",
                "max_sections": 3,
                "max_primitives_per_section": 3
            }
            
            await self.orchestrator.start_generation_session(
                session_id=self.session_id,
                source_content=self.test_source,
                source_type="textbook",
                user_preferences=user_preferences
            )
            
            progress = await self.orchestrator.get_generation_progress(self.session_id)
            print(f"✅ Session started: {progress.status}")
            print(f"📊 Current step: {progress.current_step}")
            results["steps_completed"].append("session_started")
            
            # Step 2: Generate Blueprint (this is where the LLM calls happen)
            print("\n🔍 Step 2: Generating Learning Blueprint")
            print("-" * 30)
            print("⏳ This step will make real LLM calls to Gemini 2.5 Flash...")
            
            await self.orchestrator.proceed_to_next_step(self.session_id)
            progress = await self.orchestrator.get_generation_progress(self.session_id)
            
            if progress.current_step == GenerationStep.BLUEPRINT_CREATION:
                print("✅ Blueprint generated successfully!")
                print(f"📊 Current step: {progress.current_step}")
                results["steps_completed"].append("blueprint_generated")
                
                # Show what was generated
                blueprint = progress.current_content.get("blueprint", {})
                if isinstance(blueprint, dict):
                    print(f"📋 Blueprint Title: {blueprint.get('title', 'N/A')}")
                    print(f"📋 Blueprint Description: {blueprint.get('description', 'N/A')[:100]}...")
                    
                    # Show sections if they exist
                    sections = blueprint.get("sections", [])
                    if sections:
                        print(f"📚 Number of sections: {len(sections)}")
                        for i, section in enumerate(sections[:2]):
                            print(f"   {i+1}. {section.get('title', 'N/A')}")
                
            elif progress.current_step == GenerationStep.SECTION_GENERATION:
                print("✅ Blueprint generated and moved to sections!")
                print(f"📊 Current step: {progress.current_step}")
                results["steps_completed"].append("blueprint_generated")
                
            else:
                print(f"❌ Unexpected step: {progress.current_step}")
                results["errors"].append(f"Unexpected step: {progress.current_step}")
            
            # Step 3: Try to generate sections
            print("\n📚 Step 3: Generating Sections")
            print("-" * 30)
            
            try:
                await self.orchestrator.proceed_to_next_step(self.session_id)
                progress = await self.orchestrator.get_generation_progress(self.session_id)
                
                if progress.current_step == GenerationStep.PRIMITIVE_EXTRACTION:
                    print("✅ Sections generated successfully!")
                    print(f"📊 Current step: {progress.current_step}")
                    results["steps_completed"].append("sections_generated")
                    
                    # Show sections
                    sections = progress.current_content.get("sections", [])
                    if sections:
                        print(f"📚 Number of sections: {len(sections)}")
                        for i, section in enumerate(sections[:2]):
                            print(f"   {i+1}. {section.get('title', 'N/A')}")
                            
                else:
                    print(f"📊 Current step: {progress.current_step}")
                    results["steps_completed"].append("sections_attempted")
                    
            except Exception as e:
                print(f"⚠️  Section generation encountered an issue: {str(e)}")
                results["errors"].append(f"Section generation: {str(e)}")
            
            print(f"\n🎉 Test completed! Steps completed: {len(results['steps_completed'])}")
            
        except Exception as e:
            print(f"❌ Error during workflow: {str(e)}")
            results["errors"].append(f"Workflow error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            end_time = time.time()
            results["total_time"] = end_time - start_time
            results["end_time"] = datetime.now().isoformat()
            
            print(f"\n⏱️  Total execution time: {results['total_time']:.2f} seconds")
            print("=" * 50)
            
        return results
    
    async def test_llm_service_directly(self):
        """Test the LLM service directly with a simple prompt."""
        print("\n🧪 Testing LLM Service Directly")
        print("-" * 30)
        
        try:
            # Simple test prompt
            test_prompt = """
            Please analyze this text and extract 2-3 main concepts:
            
            "Python is a programming language. It has variables, functions, and loops."
            
            Respond with a simple JSON list of concepts.
            """
            
            print("⏳ Making direct LLM call to Gemini 2.5 Flash...")
            response = await self.llm_service.call_llm(
                prompt=test_prompt,
                max_tokens=500,
                temperature=0.3
            )
            
            print(f"✅ LLM Response received: {response[:200]}...")
            return True
            
        except Exception as e:
            print(f"❌ Direct LLM test failed: {str(e)}")
            return False

async def main():
    """Main function to run the tests."""
    # Check if we have the required environment variables
    required_vars = ["GOOGLE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables before running the test.")
        return 1
    
    print("🔑 Environment variables check: PASSED")
    print(f"📡 Using model: gemini-2.5-flash (default)")
    
    try:
        test_workflow = SimpleLLMTest()
        
        # Test 1: Direct LLM service test
        print("\n🔬 Test 1: Direct LLM Service Test")
        llm_success = await test_workflow.test_llm_service_directly()
        
        if llm_success:
            # Test 2: Basic workflow test
            print("\n🔬 Test 2: Basic Workflow Test")
            workflow_results = await test_workflow.test_basic_workflow()
            
            # Summary
            print("\n📊 TEST SUMMARY")
            print("=" * 50)
            print(f"✅ LLM Service Test: {'PASSED' if llm_success else 'FAILED'}")
            print(f"✅ Workflow Test: {'PASSED' if not workflow_results['errors'] else 'FAILED'}")
            
            if workflow_results['errors']:
                print(f"\n❌ Errors encountered:")
                for error in workflow_results['errors']:
                    print(f"   - {error}")
            
            print(f"\n⏱️  Total execution time: {workflow_results['total_time']:.2f} seconds")
            print(f"📝 Steps completed: {len(workflow_results['steps_completed'])}")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"simple_test_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(workflow_results, f, indent=2, default=str)
            
            print(f"\n💾 Test results saved to: {results_file}")
            
            return 0 if not workflow_results['errors'] else 1
        else:
            print("❌ LLM service test failed, skipping workflow test")
            return 1
        
    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)




