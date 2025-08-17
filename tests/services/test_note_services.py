#!/usr/bin/env python3
"""
Test module for Note Services with REAL API calls.
Tests note creation, editing, and granular editing capabilities.
"""

import asyncio
import os
import time
from typing import Dict, Any, List

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded")
except ImportError:
    print("⚠️  python-dotenv not available")

from app.core.note_services.note_editing_service import NoteEditingService
from app.core.note_services.granular_editing_service import GranularEditingService
from app.core.note_services.note_agent_orchestrator import NoteAgentOrchestrator
from app.services.llm_service import create_llm_service
from app.models.note_creation_models import NoteEditingRequest, NoteSectionContext


class NoteServicesTester:
    """Test suite for note services with real API calls."""
    
    def __init__(self):
        self.test_results = []
        self.test_content = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.

## Supervised Learning

Supervised learning uses labeled training data to learn a mapping from inputs to outputs.

### Classification
Classification algorithms predict discrete categories or classes.

### Regression
Regression algorithms predict continuous numerical values.

## Unsupervised Learning

Unsupervised learning finds hidden patterns in unlabeled data.

### Clustering
Clustering groups similar data points together.

### Dimensionality Reduction
Dimensionality reduction reduces the number of features while preserving important information."""
        
    async def test_note_editing_service(self):
        """Test note editing service with real API calls."""
        print("\n🔍 Testing Note Editing Service")
        print("-" * 50)
        
        try:
            # Initialize service
            llm_service = create_llm_service(provider="gemini")
            editing_service = NoteEditingService(llm_service)
            
            # Test context analysis
            print("   🧠 Testing context analysis...")
            start_time = time.time()
            context = await editing_service._get_note_section_context(
                note_id=1,
                blueprint_section_id=1,
                current_content=self.test_content
            )
            end_time = time.time()
            
            print(f"   ✅ Context analysis successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📏 Context length: {len(str(context))} characters")
            
            # Test content analysis
            print("   🔍 Testing content analysis...")
            start_time = time.time()
            analysis = await editing_service._analyze_note_content_with_context(
                self.test_content,
                context
            )
            end_time = time.time()
            
            print(f"   ✅ Content analysis successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📝 Analysis: {str(analysis)[:150]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Note editing service test failed: {e}")
            return False
    
    async def test_granular_editing_service(self):
        """Test granular editing service with real API calls."""
        print("\n🔍 Testing Granular Editing Service")
        print("-" * 50)
        
        try:
            # Initialize service
            llm_service = create_llm_service(provider="gemini")
            granular_service = GranularEditingService(llm_service)
            
            # Test line-level editing
            print("   ✏️  Testing line-level editing...")
            start_time = time.time()
            edited_content, edits = await granular_service._execute_line_level_edit(
                request=NoteEditingRequest(
                    note_id=1,
                    blueprint_section_id=1,
                    edit_instruction="Change the first line to be more engaging",
                    target_line_number=1
                ),
                current_content=self.test_content,
                context=NoteSectionContext(
                    note_id=1,
                    blueprint_section_id=1,
                    section_title="Introduction to Machine Learning",
                    related_notes=[],
                    knowledge_primitives=[]
                )
            )
            end_time = time.time()
            
            print(f"   ✅ Line-level editing successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📝 Edits: {len(edits)} changes made")
            
            # Test section-level editing
            print("   📑 Testing section-level editing...")
            start_time = time.time()
            edited_content, edits = await granular_service._execute_section_level_edit(
                request=NoteEditingRequest(
                    note_id=1,
                    blueprint_section_id=1,
                    edit_instruction="Add a new section about reinforcement learning",
                    target_section_title="Reinforcement Learning"
                ),
                current_content=self.test_content,
                context=NoteSectionContext(
                    note_id=1,
                    blueprint_section_id=1,
                    section_title="Introduction to Machine Learning",
                    related_notes=[],
                    knowledge_primitives=[]
                )
            )
            end_time = time.time()
            
            print(f"   ✅ Section-level editing successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📝 Edits: {len(edits)} changes made")
            
            return True
            
        except Exception as e:
            print(f"❌ Granular editing service test failed: {e}")
            return False
    
    async def test_note_agent_orchestrator(self):
        """Test note agent orchestrator with real API calls."""
        print("\n🔍 Testing Note Agent Orchestrator")
        print("-" * 50)
        
        try:
            # Initialize orchestrator
            orchestrator = NoteAgentOrchestrator()
            
            # Test editing suggestions
            print("   💡 Testing editing suggestions...")
            start_time = time.time()
            suggestions = await orchestrator.get_editing_suggestions(
                note_id=1,
                blueprint_section_id=1,
                current_content=self.test_content,
                edit_instruction="Make this more engaging for beginners"
            )
            end_time = time.time()
            
            print(f"   ✅ Editing suggestions successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📝 Suggestions: {len(suggestions.suggestions)} generated")
            
            # Test agentic editing
            print("   🤖 Testing agentic editing...")
            start_time = time.time()
            edit_result = await orchestrator.edit_note_agentically(
                request=NoteEditingRequest(
                    note_id=1,
                    blueprint_section_id=1,
                    edit_instruction="Improve the introduction paragraph",
                    edit_type="enhancement"
                ),
                current_content=self.test_content
            )
            end_time = time.time()
            
            print(f"   ✅ Agentic editing successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📝 Result: {len(edit_result.edited_content)} characters")
            
            return True
            
        except Exception as e:
            print(f"❌ Note agent orchestrator test failed: {e}")
            return False
    
    async def test_content_conversion(self):
        """Test content conversion between formats."""
        print("\n🔍 Testing Content Conversion")
        print("-" * 50)
        
        try:
            from app.core.note_services.content_conversion_service import ContentConversionService
            
            conversion_service = ContentConversionService()
            
            # Test markdown to BlockNote conversion
            print("   🔄 Testing markdown to BlockNote conversion...")
            start_time = time.time()
            blocknote_content = await conversion_service.markdown_to_blocknote(self.test_content)
            end_time = time.time()
            
            print(f"   ✅ Markdown to BlockNote conversion successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📝 BlockNote content: {len(str(blocknote_content))} characters")
            
            # Test BlockNote to markdown conversion
            print("   🔄 Testing BlockNote to markdown conversion...")
            start_time = time.time()
            markdown_content = await conversion_service.blocknote_to_markdown(blocknote_content)
            end_time = time.time()
            
            print(f"   ✅ BlockNote to markdown conversion successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📝 Markdown content: {len(markdown_content)} characters")
            
            return True
            
        except Exception as e:
            print(f"❌ Content conversion test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all note service tests."""
        print("🚀 Note Services Test Suite")
        print("=" * 60)
        
        tests = [
            ("Note Editing Service", self.test_note_editing_service),
            ("Granular Editing Service", self.test_granular_editing_service),
            ("Note Agent Orchestrator", self.test_note_agent_orchestrator),
            ("Content Conversion", self.test_content_conversion)
        ]
        
        for test_name, test_func in tests:
            try:
                success = await test_func()
                self.test_results.append((test_name, success))
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                self.test_results.append((test_name, False))
        
        # Print summary
        print("\n📊 Note Services Test Results")
        print("-" * 40)
        passed = sum(1 for _, success in self.test_results if success)
        total = len(self.test_results)
        
        for test_name, success in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {status} {test_name}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        return passed == total


async def main():
    """Run note services tests."""
    tester = NoteServicesTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())





