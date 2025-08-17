#!/usr/bin/env python3
"""
Test script for the new LangGraph-based sequential generation workflow.
This demonstrates the modern LangGraph functional API approach.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.core.premium.workflows.sequential_generation_workflow import SequentialGenerationWorkflow

async def test_langgraph_workflow():
    """Test the LangGraph-based sequential generation workflow"""
    
    print("🚀 Testing LangGraph-based Sequential Generation Workflow")
    print("=" * 60)
    
    # Initialize the workflow
    workflow = SequentialGenerationWorkflow()
    
    # Sample source content
    source_content = """
    Photosynthesis is the process by which plants convert light energy into chemical energy.
    This process occurs in the chloroplasts of plant cells and involves several key steps:
    
    1. Light Absorption: Chlorophyll molecules absorb light energy from the sun
    2. Water Splitting: Water molecules are split into oxygen, protons, and electrons
    3. Carbon Fixation: Carbon dioxide is converted into organic compounds
    4. Sugar Production: Glucose and other sugars are synthesized
    
    The overall chemical equation for photosynthesis is:
    6CO2 + 6H2O + light energy → C6H12O6 + 6O2
    
    This process is essential for life on Earth as it provides oxygen and food for other organisms.
    """
    
    source_type = "educational_text"
    user_preferences = {
        "difficulty_level": "intermediate",
        "target_audience": "high_school_students",
        "include_examples": True,
        "include_diagrams": False
    }
    
    try:
        # Start the workflow
        print("\n📋 Starting workflow...")
        workflow_id = await workflow.start_workflow(source_content, source_type, user_preferences)
        print(f"✅ Workflow started with ID: {workflow_id}")
        
        # Get workflow status
        print("\n📊 Getting workflow status...")
        status = await workflow.get_workflow_status(workflow_id)
        
        if status:
            print(f"✅ Workflow status retrieved")
            print(f"   - Current step: {status.get('current_step', 'N/A')}")
            print(f"   - Status: {status.get('status', 'N/A')}")
            print(f"   - Started at: {status.get('started_at', 'N/A')}")
            print(f"   - Last updated: {status.get('last_updated', 'N/A')}")
            
            # Display generated content
            print(f"\n📚 Generated Content Summary:")
            print(f"   - Sections: {len(status.get('sections', []))}")
            print(f"   - Primitives: {len(status.get('primitives', []))}")
            print(f"   - Mastery Criteria: {len(status.get('mastery_criteria', []))}")
            print(f"   - Questions: {len(status.get('questions', []))}")
            print(f"   - Notes: {len(status.get('notes', []))}")
            
            # Show some details
            if status.get('sections'):
                print(f"\n📖 Sample Sections:")
                for i, section in enumerate(status.get('sections', [])[:3]):
                    print(f"   {i+1}. {section.get('title', 'N/A')}")
            
            if status.get('primitives'):
                print(f"\n🧠 Sample Primitives:")
                for i, primitive in enumerate(status.get('primitives', [])[:3]):
                    print(f"   {i+1}. {primitive.get('title', 'N/A')}")
            
            if status.get('mastery_criteria'):
                print(f"\n🎯 Sample Mastery Criteria:")
                for i, criterion in enumerate(status.get('mastery_criteria', [])[:3]):
                    print(f"   {i+1}. {criterion.get('title', 'N/A')}")
            
            if status.get('questions'):
                print(f"\n❓ Sample Questions:")
                for i, question in enumerate(status.get('questions', [])[:3]):
                    print(f"   {i+1}. {question.get('question_text', 'N/A')[:60]}...")
            
            if status.get('notes'):
                print(f"\n📝 Sample Notes:")
                for i, note in enumerate(status.get('notes', [])[:3]):
                    print(f"   {i+1}. {note.get('title', 'N/A')}")
            
            # Check for errors
            if status.get('errors'):
                print(f"\n❌ Errors encountered:")
                for error in status.get('errors', []):
                    print(f"   - {error}")
            
            # Check user edits
            if status.get('user_edits'):
                print(f"\n✏️ User edits applied:")
                for edit in status.get('user_edits', []):
                    print(f"   - {edit.get('timestamp', 'N/A')}: {len(edit.get('edits', {}))} changes")
            
        else:
            print("❌ Failed to retrieve workflow status")
        
        print(f"\n🎉 LangGraph workflow test completed!")
        
    except Exception as e:
        print(f"❌ Error during workflow test: {e}")
        import traceback
        traceback.print_exc()

async def test_workflow_resumption():
    """Test workflow resumption capabilities"""
    
    print("\n🔄 Testing Workflow Resumption")
    print("=" * 40)
    
    workflow = SequentialGenerationWorkflow()
    
    # Start a workflow
    source_content = "Simple test content for resumption testing."
    source_type = "test_text"
    
    try:
        workflow_id = await workflow.start_workflow(source_content, source_type)
        print(f"✅ Started workflow: {workflow_id}")
        
        # Get initial status
        status = await workflow.get_workflow_status(workflow_id)
        print(f"✅ Initial status: {status.get('status', 'N/A')}")
        
        # Resume workflow (this would normally happen after user input)
        print("🔄 Resuming workflow...")
        resumed_status = await workflow.resume_workflow(workflow_id)
        print(f"✅ Resumed workflow status: {resumed_status.get('status', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error during resumption test: {e}")

async def main():
    """Main test function"""
    print("🧪 LangGraph Sequential Generation Workflow Tests")
    print("=" * 60)
    
    # Test basic workflow
    await test_langgraph_workflow()
    
    # Test workflow resumption
    await test_workflow_resumption()
    
    print("\n🎯 All tests completed!")

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("app"):
        print("❌ Please run this script from the elevate-ai-api directory")
        sys.exit(1)
    
    # Run the tests
    asyncio.run(main())

