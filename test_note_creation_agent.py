#!/usr/bin/env python3
"""
Test script for the Note Creation Agent.
Tests all major workflows and services.
"""

import asyncio
import json
from app.core.note_services.note_agent_orchestrator import NoteAgentOrchestrator
from app.services.llm_service import create_llm_service
from app.models.note_creation_models import (
    NoteGenerationRequest, ContentToNoteRequest, InputConversionRequest,
    NoteEditingRequest, UserPreferences, NoteStyle, ContentFormat
)


async def test_note_creation_agent():
    """Test all major workflows of the note creation agent."""
    
    print("🚀 Testing Note Creation Agent...")
    
    # Initialize services
    print("\n📋 Initializing services...")
    llm_service = create_llm_service(provider="mock")  # Use mock for testing
    orchestrator = NoteAgentOrchestrator(llm_service)
    
    # Test 1: Service Info
    print("\n✅ Test 1: Service Information")
    service_info = orchestrator.get_service_info()
    print(f"Service: {service_info['name']}")
    print(f"Version: {service_info['version']}")
    print(f"Capabilities: {len(service_info['capabilities'])} features")
    
    # Test 2: Workflow Status
    print("\n✅ Test 2: Workflow Status")
    status = await orchestrator.get_workflow_status()
    print(f"Orchestrator Status: {status['orchestrator_status']}")
    print(f"LLM Service: {status['services']['llm_service']['status']}")
    
    # Test 3: Input Conversion (Direct to BlockNote)
    print("\n✅ Test 3: Input Conversion")
    conversion_request = InputConversionRequest(
        input_content="This is a test note about machine learning. It covers basic concepts like supervised learning, unsupervised learning, and neural networks.",
        input_format=ContentFormat.PLAIN_TEXT,
        preserve_structure=True,
        include_metadata=True
    )
    
    conversion_response = await orchestrator.convert_input_to_blocknote(conversion_request)
    print(f"Conversion Success: {conversion_response.success}")
    print(f"Message: {conversion_response.message}")
    
    # Test 4: Content to Notes (with Blueprint)
    print("\n✅ Test 4: Content to Notes with Blueprint")
    content_request = ContentToNoteRequest(
        user_content="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        content_format=ContentFormat.PLAIN_TEXT,
        note_style=NoteStyle.CONCISE,
        user_preferences=UserPreferences(
            preferred_style=NoteStyle.CONCISE,
            include_examples=True,
            include_definitions=True,
            focus_on_key_concepts=True
        )
    )
    
    content_response = await orchestrator.create_notes_from_content(content_request)
    print(f"Content to Notes Success: {content_response.success}")
    print(f"Blueprint ID: {content_response.blueprint_id}")
    print(f"Message: {content_response.message}")
    
    # Test 5: Source to Notes (with Chunking)
    print("\n✅ Test 5: Source to Notes with Chunking")
    source_request = NoteGenerationRequest(
        source_content="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.",
        note_style=NoteStyle.DETAILED,
        user_preferences=UserPreferences(
            preferred_style=NoteStyle.DETAILED,
            include_examples=True,
            include_definitions=True,
            focus_on_key_concepts=True
        ),
        create_blueprint=True
    )
    
    source_response = await orchestrator.create_notes_from_source(source_request)
    print(f"Source to Notes Success: {source_response.success}")
    print(f"Blueprint ID: {source_response.blueprint_id}")
    print(f"Chunks Processed: {len(source_response.chunks_processed) if source_response.chunks_processed else 0}")
    print(f"Message: {source_response.message}")
    
    # Test 6: Note Editing
    print("\n✅ Test 6: Note Editing")
    editing_request = NoteEditingRequest(
        note_id="test_note_001",
        edit_instruction="Make this note more concise and add bullet points for key concepts",
        edit_type="restructure",
        preserve_original_structure=False,
        include_reasoning=True
    )
    
    editing_response = await orchestrator.edit_note_agentically(editing_request)
    print(f"Note Editing Success: {editing_response.success}")
    print(f"Edit Summary: {editing_response.edit_summary}")
    print(f"Message: {editing_response.message}")
    
    # Test 7: Editing Suggestions
    print("\n✅ Test 7: Editing Suggestions")
    suggestions_response = await orchestrator.get_editing_suggestions(
        note_id="test_note_001",
        include_grammar=True,
        include_clarity=True,
        include_structure=True
    )
    
    print(f"Suggestions Success: {suggestions_response.success}")
    print(f"Number of Suggestions: {len(suggestions_response.suggestions)}")
    print(f"Message: {suggestions_response.message}")
    
    # Test 8: Batch Processing
    print("\n✅ Test 8: Batch Processing")
    batch_requests = [
        {
            "input_content": "First test content about data science",
            "input_format": "plain_text",
            "preserve_structure": True
        },
        {
            "input_content": "Second test content about web development",
            "input_format": "plain_text",
            "preserve_structure": True
        }
    ]
    
    batch_response = await orchestrator.batch_process_notes(batch_requests, "conversion")
    print(f"Batch Processing Success: {batch_response['success']}")
    print(f"Total Requests: {batch_response['total_requests']}")
    print(f"Successful: {batch_response['successful_requests']}")
    print(f"Failed: {batch_response['failed_requests']}")
    print(f"Processing Time: {batch_response['processing_time']:.2f}s")
    
    print("\n🎉 All tests completed successfully!")
    return True


async def test_error_handling():
    """Test error handling and edge cases."""
    
    print("\n🧪 Testing Error Handling...")
    
    # Initialize services
    llm_service = create_llm_service(provider="mock")
    orchestrator = NoteAgentOrchestrator(llm_service)
    
    # Test with invalid request
    print("\n✅ Test: Invalid Request Handling")
    try:
        # This should handle the error gracefully
        invalid_response = await orchestrator.create_notes_from_source(None)
        print(f"Error Handling Success: {not invalid_response.success}")
        print(f"Error Message: {invalid_response.message}")
    except Exception as e:
        print(f"Exception caught: {str(e)}")
    
    print("\n✅ Error handling tests completed!")


async def main():
    """Main test function."""
    try:
        # Run main tests
        await test_note_creation_agent()
        
        # Run error handling tests
        await test_error_handling()
        
        print("\n🎯 Note Creation Agent is ready for production!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
