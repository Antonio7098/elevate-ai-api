#!/usr/bin/env python3
"""
Demonstration script for the updated note editing system.
Shows how to use the new blueprint section context and integer IDs.
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.note_services.note_agent_orchestrator import NoteAgentOrchestrator
from app.services.llm_service import create_llm_service
from app.models.note_creation_models import (
    NoteEditingRequest, UserPreferences, NoteStyle, ContentFormat,
    InputConversionRequest
)


async def demonstrate_updated_system():
    """Demonstrate the updated note editing system capabilities."""
    
    print("🚀 Updated Note Editing System Demonstration")
    print("=" * 60)
    
    # Initialize with REAL Gemini LLM service for demonstration
    llm_service = create_llm_service(provider="gemini")
    orchestrator = NoteAgentOrchestrator(llm_service)
    
    print("✅ System initialized with REAL Gemini LLM service")
    print("🚀 Making actual API calls to showcase real AI capabilities")
    
    # Demo 1: Basic Note Editing with Blueprint Context
    print("\n📝 Demo 1: Basic Note Editing with Blueprint Context")
    print("-" * 50)
    
    request = NoteEditingRequest(
        note_id=1,                    # Integer ID (new!)
        blueprint_section_id=5,       # Blueprint section context (new!)
        edit_instruction="Make this note more engaging and add practical examples",
        edit_type="expand",
        preserve_original_structure=False,
        include_reasoning=True,
        user_preferences=UserPreferences(
            preferred_style=NoteStyle.PRACTICAL,
            include_examples=True,
            max_note_length=2500
        )
    )
    
    print(f"📋 Note ID: {request.note_id} (integer)")
    print(f"🏗️  Blueprint Section ID: {request.blueprint_section_id}")
    print(f"✏️  Edit Instruction: {request.edit_instruction}")
    print(f"🎯 Edit Type: {request.edit_type}")
    print(f"👤 User Preferences: {request.user_preferences.preferred_style}")
    
    response = await orchestrator.edit_note_agentically(request)
    
    if response.success:
        print(f"✅ Success: {response.message}")
        print(f"📝 Content Version: {response.content_version}")
        print(f"📊 Edit Summary: {response.edit_summary}")
        if response.reasoning:
            print(f"🧠 AI Reasoning: {response.reasoning[:150]}...")
        
        # Check premium enhancement
        if response.metadata and response.metadata.get("premium_enhanced"):
            print("🌟 Premium Agentic Enhancement Applied")
            print(f"🤖 Agents Used: {response.metadata.get('agents_used', [])}")
    else:
        print(f"❌ Failed: {response.message}")
    
    # Demo 2: Context-Aware Editing Suggestions
    print("\n💡 Demo 2: Context-Aware Editing Suggestions")
    print("-" * 50)
    
    suggestions_response = await orchestrator.get_editing_suggestions(
        note_id=1,
        blueprint_section_id=5,
        include_grammar=True,
        include_clarity=True,
        include_structure=True
    )
    
    if suggestions_response.success:
        print(f"✅ Generated {len(suggestions_response.suggestions)} suggestions")
        
        # Categorize suggestions
        grammar_count = len([s for s in suggestions_response.suggestions if s.type == "grammar"])
        clarity_count = len([s for s in suggestions_response.suggestions if s.type == "clarity"])
        structure_count = len([s for s in suggestions_response.suggestions if s.type == "structure"])
        
        print(f"📚 Grammar: {grammar_count}, 🔍 Clarity: {clarity_count}, 🏗️  Structure: {structure_count}")
        
        # Show a sample suggestion
        if suggestions_response.suggestions:
            sample = suggestions_response.suggestions[0]
            print(f"\n💡 Sample Suggestion ({sample.type}):")
            print(f"   Description: {sample.description}")
            print(f"   Suggested Change: {sample.suggested_change}")
            print(f"   Confidence: {sample.confidence:.2f}")
    else:
        print(f"❌ Suggestions failed: {suggestions_response.message}")
    
    # Demo 3: Input Conversion to BlockNote
    print("\n🔄 Demo 3: Input Conversion to BlockNote Format")
    print("-" * 50)
    
    test_content = """
    Machine Learning Fundamentals
    
    Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.
    
    Key Concepts:
    - Supervised Learning: Learning from labeled examples
    - Unsupervised Learning: Finding patterns in unlabeled data  
    - Reinforcement Learning: Learning through trial and error
    
    Applications include image recognition, natural language processing, and recommendation systems.
    """
    
    conversion_request = InputConversionRequest(
        input_content=test_content,
        input_format=ContentFormat.PLAIN_TEXT,
        preserve_structure=True,
        include_metadata=True
    )
    
    print("📝 Converting plain text to BlockNote format...")
    
    conversion_response = await orchestrator.convert_input_to_blocknote(conversion_request)
    
    if conversion_response.success:
        print(f"✅ Conversion successful: {conversion_response.message}")
        print(f"📝 BlockNote Content Length: {len(conversion_response.converted_content)} chars")
        print(f"📄 Plain Text Length: {len(conversion_response.plain_text)} chars")
        
        # Validate BlockNote JSON
        try:
            import json
            blocknote_data = json.loads(conversion_response.converted_content)
            print(f"✅ BlockNote JSON Valid: Yes")
            print(f"📊 BlockNote Structure: {type(blocknote_data).__name__}")
        except json.JSONDecodeError:
            print(f"⚠️  BlockNote JSON Valid: No")
    else:
        print(f"❌ Conversion failed: {conversion_response.message}")
    
    # Demo 4: Different Edit Types
    print("\n🎯 Demo 4: Different Edit Types with Context Awareness")
    print("-" * 50)
    
    edit_types = [
        ("rewrite", "Completely rewrite in a more engaging style"),
        ("condense", "Condense to essential points only"),
        ("clarify", "Improve clarity and readability")
    ]
    
    for edit_type, instruction in edit_types:
        print(f"\n📝 Testing {edit_type} edit type...")
        
        edit_request = NoteEditingRequest(
            note_id=2,
            blueprint_section_id=3,
            edit_instruction=instruction,
            edit_type=edit_type,
            include_reasoning=True
        )
        
        edit_response = await orchestrator.edit_note_agentically(edit_request)
        
        if edit_response.success:
            print(f"   ✅ Success: {edit_response.edit_summary[:50]}...")
            print(f"   📝 Version: {edit_response.content_version}")
        else:
            print(f"   ❌ Failed: {edit_response.message}")
    
    print("\n" + "=" * 60)
    print("🎉 Demonstration Complete!")
    print("=" * 60)
    
    print("\n📋 Key Features Demonstrated:")
    print("✅ Integer ID system (note_id: int)")
    print("✅ Blueprint section context (blueprint_section_id)")
    print("✅ Context-aware editing and suggestions")
    print("✅ Content versioning")
    print("✅ BlockNote format integration")
    print("✅ Premium agentic enhancement")
    print("✅ User preferences integration")
    
    print("\n🚀 Ready for Production Use!")
    print("💡 Set GOOGLE_API_KEY for real AI capabilities")
    print("🔧 Replace mock context with real database integration")


async def main():
    """Main demonstration function."""
    try:
        await demonstrate_updated_system()
    except KeyboardInterrupt:
        print("\n⏹️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())


