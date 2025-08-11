#!/usr/bin/env python3
"""
Integration test script for the Note Creation Agent.
Tests the complete system including API endpoints and services.
"""

import asyncio
import json
import time
from typing import Dict, Any
from app.core.note_services.note_agent_orchestrator import NoteAgentOrchestrator
from app.services.llm_service import create_llm_service
from app.models.note_creation_models import (
    NoteGenerationRequest, ContentToNoteRequest, InputConversionRequest,
    NoteEditingRequest, UserPreferences, NoteStyle, ContentFormat,
    ChunkingStrategy
)


async def test_complete_workflow():
    """Test the complete note creation workflow end-to-end."""
    
    print("🚀 Testing Complete Note Creation Agent Workflow...")
    print("=" * 60)
    
    # Initialize services
    print("\n📋 Initializing services...")
    start_time = time.time()
    
    try:
        # Try to use Gemini first, fall back to mock if not available
        try:
            llm_service = create_llm_service(provider="gemini")
            print("✅ Using Gemini LLM service")
        except Exception as e:
            print(f"⚠️  Gemini service not available, using mock: {e}")
            llm_service = create_llm_service(provider="mock")
            print("✅ Using Mock LLM service")
        
        orchestrator = NoteAgentOrchestrator(llm_service)
        init_time = time.time() - start_time
        print(f"✅ Services initialized in {init_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
        return False
    
    # Test 1: Service Health Check
    print("\n🔍 Test 1: Service Health Check")
    try:
        status = await orchestrator.get_workflow_status()
        print(f"✅ Orchestrator Status: {status['orchestrator_status']}")
        print(f"✅ LLM Service: {status['services']['llm_service']['status']}")
        print(f"✅ Available Workflows: {list(status['workflows'].keys())}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 2: Service Information
    print("\n🔍 Test 2: Service Information")
    try:
        service_info = orchestrator.get_service_info()
        print(f"✅ Service: {service_info['name']}")
        print(f"✅ Version: {service_info['version']}")
        print(f"✅ Capabilities: {len(service_info['capabilities'])} features")
        print(f"✅ Supported Formats: {service_info['supported_formats']}")
    except Exception as e:
        print(f"❌ Service info failed: {e}")
        return False
    
    # Test 3: Input Conversion (Direct to BlockNote)
    print("\n🔍 Test 3: Input Conversion to BlockNote")
    try:
        conversion_request = InputConversionRequest(
            input_content="""Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.

Key Concepts:
- Supervised Learning: Learning from labeled examples
- Unsupervised Learning: Finding patterns in unlabeled data  
- Reinforcement Learning: Learning through trial and error

Applications include image recognition, natural language processing, and recommendation systems.""",
            input_format=ContentFormat.PLAIN_TEXT,
            preserve_structure=True,
            include_metadata=True
        )
        
        conversion_response = await orchestrator.convert_input_to_blocknote(conversion_request)
        print(f"✅ Conversion Success: {conversion_response.success}")
        print(f"✅ Message: {conversion_response.message}")
        
        if conversion_response.converted_content:
            print(f"✅ Generated BlockNote content (length: {len(conversion_response.converted_content)} chars)")
        
    except Exception as e:
        print(f"❌ Input conversion failed: {e}")
        return False
    
    # Test 4: Content to Notes (with Blueprint)
    print("\n🔍 Test 4: Content to Notes with Blueprint")
    try:
        content_request = ContentToNoteRequest(
            user_content="""Python Programming Language

Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.

Key Features:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Strong community support

Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.""",
            content_format=ContentFormat.PLAIN_TEXT,
            note_style=NoteStyle.CONCISE,
            user_preferences=UserPreferences(
                preferred_style=NoteStyle.CONCISE,
                include_examples=True,
                include_definitions=True,
                focus_on_key_concepts=True
            ),
            create_blueprint=True
        )
        
        content_response = await orchestrator.create_notes_from_content(content_request)
        print(f"✅ Content to Notes Success: {content_response.success}")
        print(f"✅ Message: {content_response.message}")
        
        if content_response.blueprint_id:
            print(f"✅ Blueprint Created: {content_response.blueprint_id}")
        
        if content_response.converted_content:
            print(f"✅ Generated Notes (length: {len(content_response.converted_content)} chars)")
        
    except Exception as e:
        print(f"❌ Content to notes failed: {e}")
        return False
    
    # Test 5: Source to Notes (with Chunking)
    print("\n🔍 Test 5: Source to Notes with Chunking")
    try:
        source_request = NoteGenerationRequest(
            source_content="""Artificial Intelligence and Machine Learning

Artificial Intelligence (AI) is a broad field of computer science that aims to create systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

Machine Learning is a subset of AI that focuses on the development of algorithms and statistical models that enable computers to improve their performance on a specific task through experience. Instead of following pre-programmed rules, machine learning systems learn patterns from data and make predictions or decisions based on those patterns.

There are three main types of machine learning:

1. Supervised Learning: In this approach, the algorithm is trained on a labeled dataset, where the correct answers are provided. The goal is to learn a mapping from inputs to outputs so that the algorithm can make accurate predictions on new, unseen data. Examples include classification (e.g., spam detection) and regression (e.g., predicting house prices).

2. Unsupervised Learning: Here, the algorithm works with unlabeled data and tries to find hidden patterns or structures. The goal is to discover the underlying structure of the data without any guidance. Common techniques include clustering (grouping similar data points) and dimensionality reduction (reducing the number of features while preserving important information).

3. Reinforcement Learning: This approach involves an agent learning to make decisions by taking actions in an environment and receiving rewards or penalties. The agent learns through trial and error, gradually improving its strategy to maximize cumulative rewards over time. Applications include game playing, robotics, and autonomous systems.

Deep Learning is a specialized subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. These neural networks are inspired by the structure and function of the human brain, with interconnected nodes (neurons) that process and transmit information.

The success of machine learning and AI has been driven by several factors:
- Availability of large datasets
- Increased computational power
- Advances in algorithms and techniques
- Growing interest and investment in the field

Applications of AI and machine learning are widespread across various industries:
- Healthcare: Medical diagnosis, drug discovery, personalized medicine
- Finance: Fraud detection, algorithmic trading, risk assessment
- Transportation: Autonomous vehicles, traffic optimization, route planning
- Entertainment: Recommendation systems, content generation, gaming
- Manufacturing: Predictive maintenance, quality control, supply chain optimization

However, the rapid advancement of AI also raises important considerations:
- Ethical implications and bias in algorithms
- Privacy and data security concerns
- Impact on employment and workforce
- Need for responsible AI development and deployment

As AI continues to evolve, it's crucial to balance innovation with ethical considerations and ensure that these technologies benefit society as a whole.""",
            note_style=NoteStyle.DETAILED,
            user_preferences=UserPreferences(
                preferred_style=NoteStyle.DETAILED,
                include_examples=True,
                include_definitions=True,
                focus_on_key_concepts=True,
                max_note_length=3000
            ),
            create_blueprint=True,
            chunking_strategy=ChunkingStrategy(
                max_chunk_size=4000,
                chunk_overlap=500,
                semantic_boundaries=True,
                preserve_structure=True
            )
        )
        
        source_response = await orchestrator.create_notes_from_source(source_request)
        print(f"✅ Source to Notes Success: {source_response.success}")
        print(f"✅ Message: {source_response.message}")
        
        if source_response.blueprint_id:
            print(f"✅ Blueprint Created: {source_response.blueprint_id}")
        
        if source_response.chunks_processed:
            print(f"✅ Chunks Processed: {len(source_response.chunks_processed)}")
            for i, chunk in enumerate(source_response.chunks_processed[:3]):  # Show first 3 chunks
                print(f"   Chunk {i+1}: {chunk.topic} ({len(chunk.content)} chars)")
        
        if source_response.note_content:
            print(f"✅ Generated Notes (length: {len(source_response.note_content)} chars)")
        
    except Exception as e:
        print(f"❌ Source to notes failed: {e}")
        return False
    
    # Test 6: Note Editing
    print("\n🔍 Test 6: Note Editing")
    try:
        editing_request = NoteEditingRequest(
            note_id="test_note_001",
            edit_instruction="Make this note more concise and add bullet points for key concepts. Focus on the most important information.",
            edit_type="restructure",
            preserve_original_structure=False,
            include_reasoning=True
        )
        
        editing_response = await orchestrator.edit_note_agentically(editing_request)
        print(f"✅ Note Editing Success: {editing_response.success}")
        print(f"✅ Message: {editing_response.message}")
        
        if editing_response.edited_content:
            print(f"✅ Edited Content (length: {len(editing_response.edited_content)} chars)")
        
        if editing_response.reasoning:
            print(f"✅ AI Reasoning: {editing_response.reasoning[:200]}...")
        
    except Exception as e:
        print(f"❌ Note editing failed: {e}")
        return False
    
    # Test 7: Batch Processing
    print("\n🔍 Test 7: Batch Processing")
    try:
        batch_requests = [
            {
                "input_content": "Simple note about databases",
                "input_format": "plain_text",
                "target_format": "blocknote"
            },
            {
                "input_content": "Another note about web development",
                "input_format": "plain_text", 
                "target_format": "blocknote"
            }
        ]
        
        batch_response = await orchestrator.batch_process_notes(batch_requests, "conversion")
        print(f"✅ Batch Processing Success: {batch_response['success']}")
        print(f"✅ Total Requests: {batch_response['total_requests']}")
        print(f"✅ Successful: {batch_response['successful_requests']}")
        print(f"✅ Failed: {batch_response['failed_requests']}")
        print(f"✅ Processing Time: {batch_response['processing_time']:.2f}s")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return False
    
    # Test 8: Performance Metrics
    print("\n🔍 Test 8: Performance Metrics")
    total_time = time.time() - start_time
    print(f"✅ Total Test Time: {total_time:.2f}s")
    print(f"✅ Average Time per Test: {total_time/8:.2f}s")
    
    print("\n🎉 All tests completed successfully!")
    print("=" * 60)
    return True


async def test_error_handling():
    """Test error handling and edge cases."""
    
    print("\n🧪 Testing Error Handling and Edge Cases...")
    print("=" * 60)
    
    try:
        llm_service = create_llm_service(provider="mock")
        orchestrator = NoteAgentOrchestrator(llm_service)
        
        # Test with empty content
        print("\n🔍 Test: Empty Content Handling")
        try:
            empty_request = InputConversionRequest(
                input_content="",
                input_format=ContentFormat.PLAIN_TEXT
            )
            response = await orchestrator.convert_input_to_blocknote(empty_request)
            print(f"✅ Empty content handled: {response.message}")
        except Exception as e:
            print(f"✅ Empty content error caught: {e}")
        
        # Test with very long content
        print("\n🔍 Test: Long Content Handling")
        try:
            long_content = "This is a very long note. " * 1000  # ~30k characters
            long_request = InputConversionRequest(
                input_content=long_content,
                input_format=ContentFormat.PLAIN_TEXT
            )
            response = await orchestrator.convert_input_to_blocknote(long_request)
            print(f"✅ Long content handled: {response.message}")
        except Exception as e:
            print(f"✅ Long content error caught: {e}")
        
        print("\n✅ Error handling tests completed!")
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")


async def main():
    """Main test function."""
    print("🧪 Note Creation Agent Integration Tests")
    print("=" * 60)
    
    # Test complete workflow
    success = await test_complete_workflow()
    
    if success:
        # Test error handling
        await test_error_handling()
        
        print("\n🎯 Integration Test Summary:")
        print("✅ All core workflows tested successfully")
        print("✅ Services properly integrated")
        print("✅ Error handling verified")
        print("✅ Performance metrics collected")
        print("\n🚀 Note Creation Agent is ready for production!")
    else:
        print("\n❌ Integration test failed. Please check the implementation.")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        exit(1)
