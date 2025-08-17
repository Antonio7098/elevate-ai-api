#!/usr/bin/env python3
"""
Comprehensive test script for the updated Note Editing System.
Tests the new blueprint section context awareness and integer ID system.
Uses real LLM calls to verify functionality.
"""

import asyncio
import json
import time
from typing import Dict, Any
from app.core.note_services.note_agent_orchestrator import NoteAgentOrchestrator
from app.services.llm_service import create_llm_service
from app.models.note_creation_models import (
    NoteEditingRequest, UserPreferences, NoteStyle, ContentFormat,
    InputConversionRequest
)


class NoteEditingSystemTester:
    """Test suite for the updated note editing system."""
    
    def __init__(self):
        self.orchestrator = None
        self.test_results = []
        
    async def setup(self):
        """Initialize the test environment."""
        print("🚀 Setting up Note Editing System Test Environment...")
        print("=" * 70)
        
        try:
            # Use REAL Gemini LLM service for testing with actual AI capabilities
            # This will make real API calls to test the system end-to-end
            print("🚀 Using REAL Gemini LLM service for end-to-end testing")
            print("💡 Making actual API calls to test real AI capabilities")
            
            llm_service = create_llm_service(provider="gemini")
            print("✅ Real Gemini LLM service initialized successfully")
            
            self.orchestrator = NoteAgentOrchestrator(llm_service)
            print("✅ Note Agent Orchestrator initialized successfully")
            
            # Test service health
            status = await self.orchestrator.get_workflow_status()
            print(f"✅ Service Status: {status.get('overall_status', 'unknown')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_basic_note_editing(self):
        """Test basic note editing functionality with new schema."""
        print("\n🔍 Test 1: Basic Note Editing with Blueprint Context")
        print("-" * 50)
        
        try:
            # Create a test note editing request
            request = NoteEditingRequest(
                note_id=1,
                blueprint_section_id=1,
                edit_instruction="Make this note more concise and clear",
                edit_type="clarify",
                preserve_original_structure=True,
                include_reasoning=True,
                user_preferences=UserPreferences(
                    preferred_style=NoteStyle.CONCISE,
                    include_examples=True,
                    max_note_length=1500
                )
            )
            
            print(f"📝 Testing note editing for note ID: {request.note_id}")
            print(f"📋 Blueprint Section ID: {request.blueprint_section_id}")
            print(f"✏️  Edit Instruction: {request.edit_instruction}")
            print(f"🎯 Edit Type: {request.edit_type}")
            
            # Execute the editing
            start_time = time.time()
            response = await self.orchestrator.edit_note_agentically(request)
            processing_time = time.time() - start_time
            
            print(f"⏱️  Processing Time: {processing_time:.2f}s")
            print(f"✅ Success: {response.success}")
            print(f"📄 Message: {response.message}")
            
            if response.success:
                print(f"📝 Content Version: {response.content_version}")
                print(f"📊 Edit Summary: {response.edit_summary}")
                if response.reasoning:
                    print(f"🧠 AI Reasoning: {response.reasoning[:200]}...")
                
                # Check if premium enhancement was used
                if response.metadata and response.metadata.get("premium_enhanced"):
                    print("🌟 Premium Agentic Enhancement Applied")
                    print(f"🤖 Agents Used: {response.metadata.get('agents_used', [])}")
                    print(f"📈 Quality Score: {response.metadata.get('quality_score', 'N/A')}")
                
                self.test_results.append({
                    "test": "basic_note_editing",
                    "success": True,
                    "processing_time": processing_time,
                    "content_version": response.content_version,
                    "premium_enhanced": response.metadata.get("premium_enhanced", False) if response.metadata else False
                })
                
            else:
                print(f"❌ Editing failed: {response.message}")
                self.test_results.append({
                    "test": "basic_note_editing",
                    "success": False,
                    "error": response.message
                })
                
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            self.test_results.append({
                "test": "basic_note_editing",
                "success": False,
                "error": str(e)
            })
    
    async def test_editing_suggestions(self):
        """Test the editing suggestions system with blueprint context."""
        print("\n🔍 Test 2: Editing Suggestions with Blueprint Context")
        print("-" * 50)
        
        try:
            print("📝 Testing editing suggestions for note ID: 1, Blueprint Section: 1")
            
            start_time = time.time()
            response = await self.orchestrator.get_editing_suggestions(
                note_id=1,
                blueprint_section_id=1,
                include_grammar=True,
                include_clarity=True,
                include_structure=True
            )
            processing_time = time.time() - start_time
            
            print(f"⏱️  Processing Time: {processing_time:.2f}s")
            print(f"✅ Success: {response.success}")
            print(f"📄 Message: {response.message}")
            
            if response.success:
                print(f"💡 Generated {len(response.suggestions)} suggestions")
                
                # Categorize suggestions
                grammar_count = len([s for s in response.suggestions if s.type == "grammar"])
                clarity_count = len([s for s in response.suggestions if s.type == "clarity"])
                structure_count = len([s for s in response.suggestions if s.type == "structure"])
                
                print(f"📚 Grammar Suggestions: {grammar_count}")
                print(f"🔍 Clarity Suggestions: {clarity_count}")
                print(f"🏗️  Structure Suggestions: {structure_count}")
                
                # Show a few sample suggestions
                for i, suggestion in enumerate(response.suggestions[:3]):
                    print(f"\n💡 Suggestion {i+1} ({suggestion.type}):")
                    print(f"   Description: {suggestion.description[:100]}...")
                    print(f"   Suggested Change: {suggestion.suggested_change[:100]}...")
                    print(f"   Confidence: {suggestion.confidence:.2f}")
                
                self.test_results.append({
                    "test": "editing_suggestions",
                    "success": True,
                    "processing_time": processing_time,
                    "total_suggestions": len(response.suggestions),
                    "grammar_count": grammar_count,
                    "clarity_count": clarity_count,
                    "structure_count": structure_count
                })
                
            else:
                print(f"❌ Suggestions failed: {response.message}")
                self.test_results.append({
                    "test": "editing_suggestions",
                    "success": False,
                    "error": response.message
                })
                
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            self.test_results.append({
                "test": "editing_suggestions",
                "success": False,
                "error": str(e)
            })
    
    async def test_different_edit_types(self):
        """Test different edit types with blueprint context awareness."""
        print("\n🔍 Test 3: Different Edit Types with Blueprint Context")
        print("-" * 50)
        
        edit_types = [
            ("rewrite", "Completely rewrite this note in a more engaging style"),
            ("expand", "Expand this note with more examples and explanations"),
            ("condense", "Condense this note to its essential points"),
            ("restructure", "Reorganize this note for better logical flow"),
            ("clarify", "Clarify any confusing parts and improve readability")
        ]
        
        for edit_type, instruction in edit_types:
            try:
                print(f"\n📝 Testing {edit_type} edit type...")
                
                request = NoteEditingRequest(
                    note_id=2,
                    blueprint_section_id=1,
                    edit_instruction=instruction,
                    edit_type=edit_type,
                    preserve_original_structure=(edit_type != "restructure"),
                    include_reasoning=True
                )
                
                start_time = time.time()
                response = await self.orchestrator.edit_note_agentically(request)
                processing_time = time.time() - start_time
                
                print(f"   ⏱️  Processing Time: {processing_time:.2f}s")
                print(f"   ✅ Success: {response.success}")
                
                if response.success:
                    print(f"   📊 Edit Summary: {response.edit_summary[:100]}...")
                    print(f"   📝 Content Version: {response.content_version}")
                    
                    # Check for premium enhancement on complex edits
                    if edit_type in ["restructure", "expand"] and response.metadata:
                        if response.metadata.get("premium_enhanced"):
                            print(f"   🌟 Premium Enhanced: Yes")
                        else:
                            print(f"   🌟 Premium Enhanced: No")
                    
                    self.test_results.append({
                        "test": f"edit_type_{edit_type}",
                        "success": True,
                        "processing_time": processing_time,
                        "content_version": response.content_version
                    })
                else:
                    print(f"   ❌ Failed: {response.message}")
                    self.test_results.append({
                        "test": f"edit_type_{edit_type}",
                        "success": False,
                        "error": response.message
                    })
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                self.test_results.append({
                    "test": f"edit_type_{edit_type}",
                    "success": False,
                    "error": str(e)
                })
    
    async def test_blueprint_context_awareness(self):
        """Test the blueprint context awareness functionality."""
        print("\n🔍 Test 4: Blueprint Context Awareness")
        print("-" * 50)
        
        try:
            # Test with different blueprint sections to see context awareness
            test_cases = [
                (1, 1, "Mathematics Fundamentals"),
                (2, 3, "Advanced Calculus"),
                (3, 5, "Machine Learning Basics")
            ]
            
            for note_id, blueprint_section_id, section_name in test_cases:
                print(f"\n📝 Testing context awareness for {section_name}...")
                
                request = NoteEditingRequest(
                    note_id=note_id,
                    blueprint_section_id=blueprint_section_id,
                    edit_instruction="Improve this note while maintaining consistency with the section context",
                    edit_type="clarify",
                    include_reasoning=True
                )
                
                start_time = time.time()
                response = await self.orchestrator.edit_note_agentically(request)
                processing_time = time.time() - start_time
                
                print(f"   📍 Note ID: {note_id}, Section: {blueprint_section_id}")
                print(f"   ⏱️  Processing Time: {processing_time:.2f}s")
                print(f"   ✅ Success: {response.success}")
                
                if response.success:
                    print(f"   📊 Edit Summary: {response.edit_summary[:100]}...")
                    
                    # Check if reasoning mentions blueprint context
                    if response.reasoning:
                        context_mentions = ["blueprint", "section", "context", "related"]
                        context_score = sum(1 for word in context_mentions if word.lower() in response.reasoning.lower())
                        print(f"   🧠 Context Awareness Score: {context_score}/4")
                    
                    self.test_results.append({
                        "test": f"context_awareness_{section_name}",
                        "success": True,
                        "processing_time": processing_time,
                        "context_score": context_score if response.reasoning else 0
                    })
                else:
                    print(f"   ❌ Failed: {response.message}")
                    self.test_results.append({
                        "test": f"context_awareness_{section_name}",
                        "success": False,
                        "error": response.message
                    })
                    
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            self.test_results.append({
                "test": "blueprint_context_awareness",
                "success": False,
                "error": str(e)
            })
    
    async def test_input_conversion_to_blocknote(self):
        """Test input conversion to BlockNote format."""
        print("\n🔍 Test 5: Input Conversion to BlockNote Format")
        print("-" * 50)
        
        try:
            test_content = """
            Machine Learning Fundamentals
            
            Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.
            
            Key Concepts:
            - Supervised Learning: Learning from labeled examples
            - Unsupervised Learning: Finding patterns in unlabeled data  
            - Reinforcement Learning: Learning through trial and error
            
            Applications include image recognition, natural language processing, and recommendation systems.
            """
            
            request = InputConversionRequest(
                input_content=test_content,
                input_format=ContentFormat.PLAIN_TEXT,
                preserve_structure=True,
                include_metadata=True
            )
            
            print("📝 Testing conversion of plain text to BlockNote format...")
            
            start_time = time.time()
            response = await self.orchestrator.convert_input_to_blocknote(request)
            processing_time = time.time() - start_time
            
            print(f"⏱️  Processing Time: {processing_time:.2f}s")
            print(f"✅ Success: {response.success}")
            print(f"📄 Message: {response.message}")
            
            if response.success:
                print(f"📝 Converted Content Length: {len(response.converted_content)} chars")
                print(f"📄 Plain Text Length: {len(response.plain_text)} chars")
                
                # Check if BlockNote format is valid JSON
                try:
                    blocknote_data = json.loads(response.converted_content)
                    print(f"✅ BlockNote JSON Valid: Yes")
                    print(f"📊 BlockNote Structure: {type(blocknote_data).__name__}")
                except json.JSONDecodeError:
                    print(f"⚠️  BlockNote JSON Valid: No (not valid JSON)")
                
                self.test_results.append({
                    "test": "input_conversion_blocknote",
                    "success": True,
                    "processing_time": processing_time,
                    "converted_length": len(response.converted_content),
                    "blocknote_valid": "valid_json" in locals()
                })
                
            else:
                print(f"❌ Conversion failed: {response.message}")
                self.test_results.append({
                    "test": "input_conversion_blocknote",
                    "success": False,
                    "error": response.message
                })
                
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            self.test_results.append({
                "test": "input_conversion_blocknote",
                "success": False,
                "error": str(e)
            })
    
    async def run_all_tests(self):
        """Run all test cases."""
        print("🚀 Starting Comprehensive Note Editing System Tests")
        print("=" * 70)
        
        # Setup
        if not await self.setup():
            print("❌ Setup failed, cannot run tests")
            return
        
        # Run all tests
        await self.test_basic_note_editing()
        await self.test_editing_suggestions()
        await self.test_different_edit_types()
        await self.test_blueprint_context_awareness()
        await self.test_input_conversion_to_blocknote()
        
        # Generate test summary
        self.generate_test_summary()
    
    def generate_test_summary(self):
        """Generate and display test results summary."""
        print("\n" + "=" * 70)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r["success"]])
        failed_tests = total_tests - successful_tests
        
        print(f"📈 Total Tests: {total_tests}")
        print(f"✅ Successful: {successful_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📊 Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        if successful_tests > 0:
            avg_processing_time = sum(r.get("processing_time", 0) for r in self.test_results if r.get("processing_time")) / successful_tests
            print(f"⏱️  Average Processing Time: {avg_processing_time:.2f}s")
        
        # Show failed tests
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"   - {result['test']}: {result.get('error', 'Unknown error')}")
        
        # Show successful test details
        if successful_tests > 0:
            print(f"\n✅ Successful Tests:")
            for result in self.test_results:
                if result["success"]:
                    details = []
                    if "processing_time" in result:
                        details.append(f"Time: {result['processing_time']:.2f}s")
                    if "content_version" in result:
                        details.append(f"Version: {result['content_version']}")
                    if "total_suggestions" in result:
                        details.append(f"Suggestions: {result['total_suggestions']}")
                    
                    print(f"   - {result['test']}: {', '.join(details)}")
        
        print("\n" + "=" * 70)
        
        if failed_tests == 0:
            print("🎉 All tests passed! The updated note editing system is working correctly.")
        else:
            print(f"⚠️  {failed_tests} test(s) failed. Please review the errors above.")


async def main():
    """Main test execution function."""
    tester = NoteEditingSystemTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
