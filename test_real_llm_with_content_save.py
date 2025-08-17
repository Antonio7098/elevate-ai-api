#!/usr/bin/env python3
"""
Real LLM Test with Content Saving
Uses Gemini 2.5 Flash service and saves the generated content for inspection.
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

class ContentSavingLLMTest:
    """Test class that saves generated content for inspection."""
    
    def __init__(self):
        """Initialize the test workflow with real services."""
        self.orchestrator = GenerationOrchestrator()
        self.llm_service = GeminiLLMService()
        
        # Test content
        self.test_source = """
        Python Basics
        
        Python is a high-level programming language known for its simplicity and readability.
        
        Key concepts:
        - Variables store data values
        - Functions are reusable code blocks
        - Loops repeat code execution
        - Lists store multiple items
        """
        
        self.session_id = f"content_test_{int(time.time())}"
        
    async def test_workflow_and_save_content(self) -> Dict[str, Any]:
        """Test the workflow and save all generated content."""
        print("🚀 Starting Real LLM Test with Content Saving")
        print("=" * 60)
        
        results = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "steps_completed": [],
            "errors": [],
            "total_time": 0,
            "source_content": self.test_source,
            "source_type": "textbook",
            "generated_content": {}
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Start generation session
            print("\n📝 Step 1: Starting Generation Session")
            print("-" * 40)
            
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
            
            # Save current content after session start
            results["generated_content"]["session_start"] = progress.current_content
            
            # Step 2: Generate Blueprint
            print("\n🔍 Step 2: Generating Learning Blueprint")
            print("-" * 40)
            print("⏳ This step will make real LLM calls to Gemini 2.5 Flash...")
            
            await self.orchestrator.proceed_to_next_step(self.session_id)
            progress = await self.orchestrator.get_generation_progress(self.session_id)
            
            if progress.current_step == GenerationStep.BLUEPRINT_CREATION:
                print("✅ Blueprint generated successfully!")
                print(f"📊 Current step: {progress.current_step}")
                results["steps_completed"].append("blueprint_generated")
                
                # Save blueprint content
                results["generated_content"]["blueprint"] = progress.current_content.get("blueprint", {})
                
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
            
            # Step 3: Generate Sections
            print("\n📚 Step 3: Generating Sections")
            print("-" * 40)
            
            try:
                await self.orchestrator.proceed_to_next_step(self.session_id)
                progress = await self.orchestrator.get_generation_progress(self.session_id)
                
                if progress.current_step == GenerationStep.PRIMITIVE_EXTRACTION:
                    print("✅ Sections generated successfully!")
                    print(f"📊 Current step: {progress.current_step}")
                    results["steps_completed"].append("sections_generated")
                    
                    # Save sections content
                    results["generated_content"]["sections"] = progress.current_content.get("sections", [])
                    
                    # Show sections
                    sections = progress.current_content.get("sections", [])
                    if sections:
                        print(f"📚 Number of sections: {len(sections)}")
                        for i, section in enumerate(sections[:2]):
                            print(f"   {i+1}. {section.get('title', 'N/A')}")
                            
                else:
                    print(f"📊 Current step: {progress.current_step}")
                    results["steps_completed"].append("sections_attempted")
                    
                    # Save whatever content we have
                    results["generated_content"]["current_content"] = progress.current_content
                    
            except Exception as e:
                print(f"⚠️  Section generation encountered an issue: {str(e)}")
                results["errors"].append(f"Section generation: {str(e)}")
                
                # Try to save whatever content we have
                try:
                    progress = await self.orchestrator.get_generation_progress(self.session_id)
                    results["generated_content"]["error_content"] = progress.current_content
                except:
                    pass
            
            print(f"\n🎉 Test completed! Steps completed: {len(results['steps_completed'])}")
            
        except Exception as e:
            print(f"❌ Error during workflow: {str(e)}")
            results["errors"].append(f"Workflow error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Try to save whatever content we have
            try:
                progress = await self.orchestrator.get_generation_progress(self.session_id)
                results["generated_content"]["error_content"] = progress.current_content
            except:
                pass
        
        finally:
            end_time = time.time()
            results["total_time"] = end_time - start_time
            results["end_time"] = datetime.now().isoformat()
            
            print(f"\n⏱️  Total execution time: {results['total_time']:.2f} seconds")
            print("=" * 60)
            
        return results

async def main():
    """Main function to run the test."""
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
        test_workflow = ContentSavingLLMTest()
        
        # Run the test and save content
        print("\n🔬 Running Workflow Test with Content Saving")
        workflow_results = await test_workflow.test_workflow_and_save_content()
        
        # Summary
        print("\n📊 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Workflow Test: {'PASSED' if not workflow_results['errors'] else 'FAILED'}")
        
        if workflow_results['errors']:
            print(f"\n❌ Errors encountered:")
            for error in workflow_results['errors']:
                print(f"   - {error}")
        
        print(f"\n⏱️  Total execution time: {workflow_results['total_time']:.2f} seconds")
        print(f"📝 Steps completed: {len(workflow_results['steps_completed'])}")
        
        # Save results with content
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"content_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(workflow_results, f, indent=2, default=str)
        
        print(f"\n💾 Test results with content saved to: {results_file}")
        print("📄 You can now examine the generated content in this file!")
        
        return 0 if not workflow_results['errors'] else 1
        
    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)




