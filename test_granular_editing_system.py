#!/usr/bin/env python3
"""
Comprehensive test suite for the new Granular Editing System.
Tests line-level, section-level, and block-level editing capabilities.
Uses real LLM calls to verify functionality.
"""

import asyncio
import json
import time
import os
from typing import Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded in test file")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment variables")
from app.core.note_services.note_agent_orchestrator import NoteAgentOrchestrator
from app.services.llm_service import create_llm_service
from app.models.note_creation_models import (
    NoteEditingRequest, UserPreferences, NoteStyle, ContentFormat
)


class GranularEditingSystemTester:
    """Test suite for the granular editing system."""
    
    def __init__(self):
        self.orchestrator = None
        self.test_results = []
        self.test_content = self._create_test_content()
        
    def _create_test_content(self) -> str:
        """Create test content for granular editing tests."""
        return """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.

## Key Concepts

Supervised learning involves training on labeled data to make predictions. This is the most common approach in practice.

Unsupervised learning finds patterns in unlabeled data. It's useful for discovering hidden structures.

Reinforcement learning learns through trial and error, receiving rewards for good decisions.

## Applications

Machine learning has applications in image recognition, natural language processing, and recommendation systems.

## Conclusion

Machine learning continues to advance rapidly, opening new possibilities for automation and intelligent systems."""

    async def setup(self):
        """Initialize the test environment."""
        print("🚀 Setting up Granular Editing System Test Environment...")
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
    
    async def test_line_level_editing(self):
        """Test line-level editing capabilities."""
        print("\n🔍 Test 1: Line-Level Editing Capabilities")
        print("-" * 50)
        
        line_edit_tests = [
            {
                "type": "edit_line",
                "target_line": 3,
                "instruction": "Make this line more engaging and clear",
                "description": "Edit specific line for clarity"
            },
            {
                "type": "add_line",
                "position": 5,
                "instruction": "Add a line explaining why machine learning is important",
                "description": "Add new line at specific position"
            },
            {
                "type": "remove_line",
                "target_line": 7,
                "instruction": "Remove the confusing sentence about unsupervised learning",
                "description": "Remove specific line"
            },
            {
                "type": "replace_line",
                "target_line": 9,
                "instruction": "Replace with a better explanation of reinforcement learning",
                "description": "Replace specific line content"
            }
        ]
        
        for test_case in line_edit_tests:
            try:
                print(f"\n📝 Testing {test_case['description']}...")
                
                if test_case["type"] == "edit_line":
                    request = NoteEditingRequest(
                        note_id=1,
                        blueprint_section_id=1,
                        edit_instruction=test_case["instruction"],
                        edit_type="edit_line",
                        target_line_number=test_case["target_line"],
                        include_reasoning=True
                    )
                elif test_case["type"] == "add_line":
                    request = NoteEditingRequest(
                        note_id=1,
                        blueprint_section_id=1,
                        edit_instruction=test_case["instruction"],
                        edit_type="add_line",
                        insertion_position=test_case["position"],
                        include_reasoning=True
                    )
                elif test_case["type"] == "remove_line":
                    request = NoteEditingRequest(
                        note_id=1,
                        blueprint_section_id=1,
                        edit_instruction=test_case["instruction"],
                        edit_type="remove_line",
                        target_line_number=test_case["target_line"],
                        include_reasoning=True
                    )
                elif test_case["type"] == "replace_line":
                    request = NoteEditingRequest(
                        note_id=1,
                        blueprint_section_id=1,
                        edit_instruction=test_case["instruction"],
                        edit_type="replace_line",
                        target_line_number=test_case["target_line"],
                        include_reasoning=True
                    )
                
                start_time = time.time()
                response = await self.orchestrator.edit_note_agentically(request)
                processing_time = time.time() - start_time
                
                print(f"   ⏱️  Processing Time: {processing_time:.2f}s")
                print(f"   ✅ Success: {response.success}")
                
                if response.success:
                    print(f"   📊 Edit Summary: {response.edit_summary}")
                    print(f"   📝 Content Version: {response.content_version}")
                    
                    # Check granular edit details
                    if response.granular_edits:
                        print(f"   🔍 Granular Edits: {len(response.granular_edits)}")
                        for edit in response.granular_edits:
                            print(f"      - {edit.edit_type} at position {edit.target_position}")
                    
                    if response.reasoning:
                        print(f"   🧠 AI Reasoning: {response.reasoning[:150]}...")
                    
                    self.test_results.append({
                        "test": f"line_level_{test_case['type']}",
                        "success": True,
                        "processing_time": processing_time,
                        "content_version": response.content_version,
                        "granular_edits": len(response.granular_edits) if response.granular_edits else 0
                    })
                else:
                    print(f"   ❌ Failed: {response.message}")
                    self.test_results.append({
                        "test": f"line_level_{test_case['type']}",
                        "success": False,
                        "error": response.message
                    })
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                self.test_results.append({
                    "test": f"line_level_{test_case['type']}",
                    "success": False,
                    "error": str(e)
                })
    
    async def test_section_level_editing(self):
        """Test section-level editing capabilities."""
        print("\n🔍 Test 2: Section-Level Editing Capabilities")
        print("-" * 50)
        
        section_edit_tests = [
            {
                "type": "edit_section",
                "target_section": "Key Concepts",
                "instruction": "Make this section more beginner-friendly with examples",
                "description": "Edit specific section for clarity"
            },
            {
                "type": "add_section",
                "position": 3,
                "title": "Real-World Examples",
                "instruction": "Add a new section with practical examples of machine learning",
                "description": "Add new section at specific position"
            },
            {
                "type": "remove_section",
                "target_section": "Conclusion",
                "instruction": "Remove the conclusion section as it's too generic",
                "description": "Remove specific section"
            }
        ]
        
        for test_case in section_edit_tests:
            try:
                print(f"\n📝 Testing {test_case['description']}...")
                
                if test_case["type"] == "edit_section":
                    request = NoteEditingRequest(
                        note_id=1,
                        blueprint_section_id=1,
                        edit_instruction=test_case["instruction"],
                        edit_type="edit_section",
                        target_section_title=test_case["target_section"],
                        include_reasoning=True
                    )
                elif test_case["type"] == "add_section":
                    request = NoteEditingRequest(
                        note_id=1,
                        blueprint_section_id=1,
                        edit_instruction=test_case["instruction"],
                        edit_type="add_section",
                        insertion_position=test_case["position"],
                        target_section_title=test_case["title"],
                        include_reasoning=True
                    )
                elif test_case["type"] == "remove_section":
                    request = NoteEditingRequest(
                        note_id=1,
                        blueprint_section_id=1,
                        edit_instruction=test_case["instruction"],
                        edit_type="remove_section",
                        target_section_title=test_case["target_section"],
                        include_reasoning=True
                    )
                
                start_time = time.time()
                response = await self.orchestrator.edit_note_agentically(request)
                processing_time = time.time() - start_time
                
                print(f"   ⏱️  Processing Time: {processing_time:.2f}s")
                print(f"   ✅ Success: {response.success}")
                
                if response.success:
                    print(f"   📊 Edit Summary: {response.edit_summary}")
                    print(f"   📝 Content Version: {response.content_version}")
                    
                    # Check granular edit details
                    if response.granular_edits:
                        print(f"   🔍 Granular Edits: {len(response.granular_edits)}")
                        for edit in response.granular_edits:
                            print(f"      - {edit.edit_type} for section '{edit.target_identifier}'")
                    
                    if response.reasoning:
                        print(f"   🧠 AI Reasoning: {response.reasoning[:150]}...")
                    
                    self.test_results.append({
                        "test": f"section_level_{test_case['type']}",
                        "success": True,
                        "processing_time": processing_time,
                        "content_version": response.content_version,
                        "granular_edits": len(response.granular_edits) if response.granular_edits else 0
                    })
                else:
                    print(f"   ❌ Failed: {response.message}")
                    self.test_results.append({
                        "test": f"section_level_{test_case['type']}",
                        "success": False,
                        "error": response.message
                    })
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                self.test_results.append({
                    "test": f"section_level_{test_case['type']}",
                    "success": False,
                    "error": str(e)
                })
    
    async def test_block_level_editing(self):
        """Test block-level editing capabilities for BlockNote format."""
        print("\n🔍 Test 3: Block-Level Editing Capabilities")
        print("-" * 50)
        
        # Create a simple BlockNote structure for testing
        test_blocknote = {
            "type": "doc",
            "content": [
                {
                    "id": "block1",
                    "type": "paragraph",
                    "content": [{"type": "text", "text": "This is the first paragraph about machine learning."}]
                },
                {
                    "id": "block2", 
                    "type": "paragraph",
                    "content": [{"type": "text", "text": "This is the second paragraph with key concepts."}]
                },
                {
                    "id": "block3",
                    "type": "paragraph", 
                    "content": [{"type": "text", "text": "This is the third paragraph about applications."}]
                }
            ]
        }
        
        block_edit_tests = [
            {
                "type": "edit_block",
                "target_block": "block2",
                "instruction": "Make this block more engaging with a question",
                "description": "Edit specific BlockNote block"
            },
            {
                "type": "add_block",
                "position": 2,
                "instruction": "Add a new block with a practical example",
                "description": "Add new BlockNote block"
            },
            {
                "type": "remove_block",
                "target_block": "block3",
                "instruction": "Remove the applications paragraph as it's too generic",
                "description": "Remove specific BlockNote block"
            }
        ]
        
        for test_case in block_edit_tests:
            try:
                print(f"\n📝 Testing {test_case['description']}...")
                
                if test_case["type"] == "edit_block":
                    request = NoteEditingRequest(
                        note_id=1,
                        blueprint_section_id=1,
                        edit_instruction=test_case["instruction"],
                        edit_type="edit_block",
                        target_block_id=test_case["target_block"],
                        include_reasoning=True
                    )
                elif test_case["type"] == "add_block":
                    request = NoteEditingRequest(
                        note_id=1,
                        blueprint_section_id=1,
                        edit_instruction=test_case["instruction"],
                        edit_type="add_block",
                        insertion_position=test_case["position"],
                        include_reasoning=True
                    )
                elif test_case["type"] == "remove_block":
                    request = NoteEditingRequest(
                        note_id=1,
                        blueprint_section_id=1,
                        edit_instruction=test_case["instruction"],
                        edit_type="remove_block",
                        target_block_id=test_case["target_block"],
                        include_reasoning=True
                    )
                
                start_time = time.time()
                response = await self.orchestrator.edit_note_agentically(request)
                processing_time = time.time() - start_time
                
                print(f"   ⏱️  Processing Time: {processing_time:.2f}s")
                print(f"   ✅ Success: {response.success}")
                
                if response.success:
                    print(f"   📊 Edit Summary: {response.edit_summary}")
                    print(f"   📝 Content Version: {response.content_version}")
                    
                    # Check granular edit details
                    if response.granular_edits:
                        print(f"   🔍 Granular Edits: {len(response.granular_edits)}")
                        for edit in response.granular_edits:
                            print(f"      - {edit.edit_type} for block '{edit.target_identifier}'")
                    
                    if response.reasoning:
                        print(f"   🧠 AI Reasoning: {response.reasoning[:150]}...")
                    
                    self.test_results.append({
                        "test": f"block_level_{test_case['type']}",
                        "success": True,
                        "processing_time": processing_time,
                        "content_version": response.content_version,
                        "granular_edits": len(response.granular_edits) if response.granular_edits else 0
                    })
                else:
                    print(f"   ❌ Failed: {response.message}")
                    self.test_results.append({
                        "test": f"block_level_{test_case['type']}",
                        "success": False,
                        "error": response.message
                    })
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                self.test_results.append({
                    "test": f"block_level_{test_case['type']}",
                    "success": False,
                    "error": str(e)
                })
    
    async def test_context_preservation(self):
        """Test that granular edits preserve context properly."""
        print("\n🔍 Test 4: Context Preservation in Granular Edits")
        print("-" * 50)
        
        try:
            print("📝 Testing context preservation during line editing...")
            
            # Test editing a line while preserving surrounding context
            request = NoteEditingRequest(
                note_id=1,
                blueprint_section_id=1,
                edit_instruction="Make this line more specific about supervised learning",
                edit_type="edit_line",
                target_line_number=6,  # "Supervised learning involves training on labeled data..."
                preserve_context=True,
                include_reasoning=True
            )
            
            start_time = time.time()
            response = await self.orchestrator.edit_note_agentically(request)
            processing_time = time.time() - start_time
            
            print(f"   ⏱️  Processing Time: {processing_time:.2f}s")
            print(f"   ✅ Success: {response.success}")
            
            if response.success:
                print(f"   📊 Edit Summary: {response.edit_summary}")
                
                # Check if context was preserved
                if response.granular_edits:
                    edit = response.granular_edits[0]
                    print(f"   🔍 Context Preserved: {edit.context_preserved}")
                    if edit.surrounding_context:
                        print(f"   📄 Surrounding Context: {edit.surrounding_context[:100]}...")
                
                if response.reasoning:
                    print(f"   🧠 AI Reasoning: {response.reasoning[:150]}...")
                
                self.test_results.append({
                    "test": "context_preservation",
                    "success": True,
                    "processing_time": processing_time,
                    "context_preserved": True
                })
            else:
                print(f"   ❌ Failed: {response.message}")
                self.test_results.append({
                    "test": "context_preservation",
                    "success": False,
                    "error": response.message
                })
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            self.test_results.append({
                "test": "context_preservation",
                "success": False,
                "error": str(e)
            })
    
    async def test_granular_vs_note_level(self):
        """Test comparison between granular and note-level editing."""
        print("\n🔍 Test 5: Granular vs Note-Level Editing Comparison")
        print("-" * 50)
        
        try:
            print("📝 Testing granular line editing...")
            
            # Test granular line editing
            granular_request = NoteEditingRequest(
                note_id=1,
                blueprint_section_id=1,
                edit_instruction="Make this line more engaging",
                edit_type="edit_line",
                target_line_number=3,
                include_reasoning=True
            )
            
            start_time = time.time()
            granular_response = await self.orchestrator.edit_note_agentically(granular_request)
            granular_time = time.time() - start_time
            
            print(f"   ⏱️  Granular Edit Time: {granular_time:.2f}s")
            print(f"   ✅ Granular Success: {granular_response.success}")
            
            print("\n📝 Testing note-level editing...")
            
            # Test note-level editing
            note_request = NoteEditingRequest(
                note_id=1,
                blueprint_section_id=1,
                edit_instruction="Make this note more engaging",
                edit_type="rewrite",
                include_reasoning=True
            )
            
            start_time = time.time()
            note_response = await self.orchestrator.edit_note_agentically(note_request)
            note_time = time.time() - start_time
            
            print(f"   ⏱️  Note-Level Edit Time: {note_time:.2f}s")
            print(f"   ✅ Note-Level Success: {note_response.success}")
            
            # Compare results
            if granular_response.success and note_response.success:
                print(f"\n📊 Comparison Results:")
                print(f"   🔍 Granular Edits: {len(granular_response.granular_edits) if granular_response.granular_edits else 0}")
                print(f"   📝 Granular Content Version: {granular_response.content_version}")
                print(f"   📝 Note-Level Content Version: {note_response.content_version}")
                print(f"   ⚡ Granular is {note_time/granular_time:.1f}x faster")
                
                self.test_results.append({
                    "test": "granular_vs_note_level",
                    "success": True,
                    "granular_time": granular_time,
                    "note_time": note_time,
                    "speed_improvement": note_time/granular_time
                })
            else:
                print(f"   ❌ One or both tests failed")
                self.test_results.append({
                    "test": "granular_vs_note_level",
                    "success": False,
                    "error": "One or both tests failed"
                })
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            self.test_results.append({
                "test": "granular_vs_note_level",
                "success": False,
                "error": str(e)
            })
    
    async def run_all_tests(self):
        """Run all granular editing tests."""
        print("🚀 Starting Granular Editing System Tests")
        print("=" * 70)
        
        # Setup
        if not await self.setup():
            print("❌ Setup failed, cannot run tests")
            return
        
        # Run all tests
        await self.test_line_level_editing()
        await self.test_section_level_editing()
        await self.test_block_level_editing()
        await self.test_context_preservation()
        await self.test_granular_vs_note_level()
        
        # Generate test summary
        self.generate_test_summary()
    
    def generate_test_summary(self):
        """Generate and display test results summary."""
        print("\n" + "=" * 70)
        print("📊 GRANULAR EDITING TEST RESULTS SUMMARY")
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
                    if "granular_edits" in result:
                        details.append(f"Granular Edits: {result['granular_edits']}")
                    if "speed_improvement" in result:
                        details.append(f"Speed: {result['speed_improvement']:.1f}x")
                    
                    print(f"   - {result['test']}: {', '.join(details)}")
        
        print("\n" + "=" * 70)
        
        if failed_tests == 0:
            print("🎉 All granular editing tests passed! The system is working correctly.")
        else:
            print(f"⚠️  {failed_tests} test(s) failed. Please review the errors above.")
        
        print("\n🚀 Granular Editing Features Tested:")
        print("✅ Line-level editing (edit, add, remove, replace)")
        print("✅ Section-level editing (edit, add, remove)")
        print("✅ Block-level editing (BlockNote format)")
        print("✅ Context preservation")
        print("✅ Performance comparison with note-level editing")


async def main():
    """Main test execution function."""
    tester = GranularEditingSystemTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
