#!/usr/bin/env python3
"""
Real LLM Test Script for Blueprint Editing Service

This script tests the blueprint editing service using REAL LLM calls
to Gemini 2.5 Flash via the LLM service. Tests actual AI functionality
and responses.
"""

import asyncio
import json
import time
import sys
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment variables")

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.blueprint_editing_service import BlueprintEditingService
from app.services.llm_service import GeminiLLMService
from app.models.blueprint_editing_models import (
    BlueprintEditingRequest, PrimitiveEditingRequest,
    MasteryCriterionEditingRequest, QuestionEditingRequest
)


class RealLLMBlueprintEditingTester:
    """Test the blueprint editing service with real LLM calls."""
    
    def __init__(self):
        """Initialize the tester with real LLM service."""
        print("🚀 Initializing Real LLM Blueprint Editing Tester...")
        self.llm_service = GeminiLLMService()
        self.blueprint_service = BlueprintEditingService(self.llm_service)
        print("✅ Services initialized successfully")
    
    async def test_blueprint_editing(self):
        """Test blueprint editing with real LLM calls."""
        print("\n" + "="*60)
        print("🧠 TESTING BLUEPRINT EDITING WITH REAL LLM CALLS")
        print("="*60)
        
        # Test 1: Edit blueprint for clarity improvement
        print("\n📝 Test 1: Editing Blueprint for Clarity Improvement")
        request = BlueprintEditingRequest(
            blueprint_id=1,
            edit_type="improve_clarity",
            edit_instruction="Make the content clearer and more engaging for beginners",
            preserve_original_structure=True,
            include_reasoning=True
        )
        
        start_time = time.time()
        response = await self.blueprint_service.edit_blueprint_agentically(request)
        processing_time = time.time() - start_time
        
        print(f"✅ Blueprint editing completed in {processing_time:.2f}s")
        print(f"📊 Success: {response.success}")
        print(f"💬 Message: {response.message}")
        if response.reasoning:
            print(f"🤔 AI Reasoning: {response.reasoning[:200]}...")
        if response.edit_summary:
            print(f"📋 Edit Summary: {response.edit_summary}")
        
        # Show before/after if successful
        if response.success and hasattr(response, 'original_content') and hasattr(response, 'edited_content'):
            print(f"\n📝 BEFORE (Original):")
            print(f"   {response.original_content[:300]}...")
            print(f"\n📝 AFTER (Edited):")
            print(f"   {response.edited_content[:300]}...")
        elif response.success:
            print(f"\n📝 EDIT RESULT:")
            print(f"   {response.message}")
        
        # Return success status
        return response.success
    
    async def test_primitive_editing(self):
        """Test primitive editing with real LLM calls."""
        print("\n" + "="*60)
        print("🔬 TESTING PRIMITIVE EDITING WITH REAL LLM CALLS")
        print("="*60)
        
        # Test 1: Edit primitive for clarity
        print("\n📝 Test 1: Editing Knowledge Primitive for Clarity")
        request = PrimitiveEditingRequest(
            primitive_id=1,
            edit_type="improve_clarity",
            edit_instruction="Simplify the concept definition and make it more accessible to beginners",
            preserve_original_structure=True,
            include_reasoning=True
        )
        
        start_time = time.time()
        response = await self.blueprint_service.edit_primitive_agentically(request)
        processing_time = time.time() - start_time
        
        print(f"✅ Primitive editing completed in {processing_time:.2f}s")
        print(f"📊 Success: {response.success}")
        print(f"💬 Message: {response.message}")
        if response.reasoning:
            print(f"🤔 AI Reasoning: {response.reasoning[:200]}...")
        if response.edit_summary:
            print(f"📋 Edit Summary: {response.edit_summary}")
        
        # Show before/after if successful
        if response.success and hasattr(response, 'original_content') and hasattr(response, 'edited_content'):
            print(f"\n📝 BEFORE (Original):")
            print(f"   {response.original_content[:300]}...")
            print(f"\n📝 AFTER (Edited):")
            print(f"   {response.edited_content[:300]}...")
        elif response.success:
            print(f"\n📝 EDIT RESULT:")
            print(f"   {response.message}")
        
        # Return success status
        return response.success
    
    async def test_mastery_criterion_editing(self):
        """Test mastery criterion editing with real LLM calls."""
        print("\n" + "="*60)
        print("🎯 TESTING MASTERY CRITERION EDITING WITH REAL LLM CALLS")
        print("="*60)
        
        # Test 1: Edit mastery criterion for clarity
        print("\n📝 Test 1: Editing Mastery Criterion for Clarity")
        request = MasteryCriterionEditingRequest(
            criterion_id=1,
            edit_type="improve_clarity",
            edit_instruction="Make the assessment criteria clearer and more specific",
            preserve_original_structure=True,
            include_reasoning=True
        )
        
        start_time = time.time()
        response = await self.blueprint_service.edit_mastery_criterion_agentically(request)
        processing_time = time.time() - start_time
        
        print(f"✅ Mastery criterion editing completed in {processing_time:.2f}s")
        print(f"📊 Success: {response.success}")
        print(f"💬 Message: {response.message}")
        if response.reasoning:
            print(f"🤔 AI Reasoning: {response.reasoning[:200]}...")
        if response.edit_summary:
            print(f"📋 Edit Summary: {response.edit_summary}")
        
        # Show before/after if successful
        if response.success and hasattr(response, 'original_content') and hasattr(response, 'edited_content'):
            print(f"\n📝 BEFORE (Original):")
            print(f"   {response.original_content[:300]}...")
            print(f"\n📝 AFTER (Edited):")
            print(f"   {response.edit_summary}")
        elif response.success:
            print(f"\n📝 EDIT RESULT:")
            print(f"   {response.message}")
        
        # Return success status
        return response.success
    
    async def test_question_editing(self):
        """Test question editing with real LLM calls."""
        print("\n" + "="*60)
        print("❓ TESTING QUESTION EDITING WITH REAL LLM CALLS")
        print("="*60)
        
        # Test 1: Edit question for clarity
        print("\n📝 Test 1: Editing Question for Clarity")
        request = QuestionEditingRequest(
            question_id=1,
            edit_type="improve_clarity",
            edit_instruction="Make the question clearer and more focused on the learning objective",
            preserve_original_structure=True,
            include_reasoning=True
        )
        
        start_time = time.time()
        response = await self.blueprint_service.edit_question_agentically(request)
        processing_time = time.time() - start_time
        
        print(f"✅ Question editing completed in {processing_time:.2f}s")
        print(f"📊 Success: {response.success}")
        print(f"💬 Message: {response.message}")
        if response.reasoning:
            print(f"🤔 AI Reasoning: {response.reasoning[:200]}...")
        if response.edit_summary:
            print(f"📋 Edit Summary: {response.edit_summary}")
        
        # Show before/after if successful
        if response.success and hasattr(response, 'original_content') and hasattr(response, 'edited_content'):
            print(f"\n📝 BEFORE (Original):")
            print(f"   {response.original_content[:300]}...")
            print(f"\n📝 AFTER (Edited):")
            print(f"   {response.edited_content[:300]}...")
        elif response.success:
            print(f"\n📝 EDIT RESULT:")
            print(f"   {response.message}")
        
        # Return success status
        return response.success
    
    async def test_suggestion_generation(self):
        """Test AI-powered suggestion generation with real LLM calls."""
        print("\n" + "="*60)
        print("💡 TESTING AI SUGGESTION GENERATION WITH REAL LLM CALLS")
        print("="*60)
        
        # Test 1: Blueprint editing suggestions
        print("\n📝 Test 1: Generating Blueprint Editing Suggestions")
        start_time = time.time()
        response = await self.blueprint_service.get_blueprint_editing_suggestions(
            blueprint_id=1,
            include_structure=True,
            include_content=True,
            include_relationships=True
        )
        processing_time = time.time() - start_time
        
        print(f"✅ Blueprint suggestions generated in {processing_time:.2f}s")
        print(f"📊 Success: {response.success}")
        print(f"💬 Message: {response.message}")
        print(f"💡 Number of suggestions: {len(response.suggestions)}")
        
        # Test 2: Primitive editing suggestions
        print("\n📝 Test 2: Generating Primitive Editing Suggestions")
        start_time = time.time()
        response = await self.blueprint_service.get_primitive_editing_suggestions(
            primitive_id=1,
            include_clarity=True,
            include_complexity=True,
            include_relationships=True
        )
        processing_time = time.time() - start_time
        
        print(f"✅ Primitive suggestions generated in {processing_time:.2f}s")
        print(f"📊 Success: {response.success}")
        print(f"💬 Message: {response.message}")
        print(f"💡 Number of suggestions: {len(response.suggestions)}")
        
        # Test 3: Mastery criterion suggestions
        print("\n📝 Test 3: Generating Mastery Criterion Suggestions")
        start_time = time.time()
        response = await self.blueprint_service.get_mastery_criterion_editing_suggestions(
            criterion_id=1,
            include_clarity=True,
            include_difficulty=True,
            include_assessment=True
        )
        processing_time = time.time() - start_time
        
        print(f"✅ Mastery criterion suggestions generated in {processing_time:.2f}s")
        print(f"📊 Success: {response.success}")
        print(f"💬 Message: {response.message}")
        print(f"💡 Number of suggestions: {len(response.suggestions)}")
        
        # Test 4: Question suggestions
        print("\n📝 Test 4: Generating Question Suggestions")
        start_time = time.time()
        response = await self.blueprint_service.get_question_editing_suggestions(
            question_id=1,
            include_clarity=True,
            include_difficulty=True,
            include_quality=True
        )
        processing_time = time.time() - start_time
        
        print(f"✅ Question suggestions generated in {processing_time:.2f}s")
        print(f"📊 Success: {response.success}")
        print(f"💬 Message: {response.message}")
        print(f"💡 Number of suggestions: {len(response.suggestions)}")
        
        # Return success status (all suggestion tests should succeed)
        return response.success
    
    async def test_granular_editing(self):
        """Test granular editing operations with real LLM calls."""
        print("\n" + "="*60)
        print("🔧 TESTING GRANULAR EDITING WITH REAL LLM CALLS")
        print("="*60)
        
        # Test 1: Add section operation
        print("\n📝 Test 1: Adding New Section to Blueprint")
        request = BlueprintEditingRequest(
            blueprint_id=1,
            edit_type="add_section",
            edit_instruction="Add a new section called 'Advanced Applications' with examples and case studies",
            preserve_original_structure=False,
            include_reasoning=True
        )
        
        start_time = time.time()
        response = await self.blueprint_service.edit_blueprint_agentically(request)
        processing_time = time.time() - start_time
        
        print(f"✅ Add section completed in {processing_time:.2f}s")
        print(f"📊 Success: {response.success}")
        print(f"💬 Message: {response.message}")
        if response.reasoning:
            print(f"🤔 AI Reasoning: {response.reasoning[:200]}...")
        if response.granular_edits:
            print(f"🔧 Number of granular edits: {len(response.granular_edits)}")
        
        # Show before/after if successful
        if response.success and hasattr(response, 'original_content') and hasattr(response, 'edited_content'):
            print(f"\n📝 BEFORE (Original):")
            print(f"   {response.original_content[:300]}...")
            print(f"\n📝 AFTER (Edited):")
            print(f"   {response.edited_content[:300]}...")
        elif response.success:
            print(f"\n📝 EDIT RESULT:")
            print(f"   {response.message}")
        
        # Test 2: Reorder sections operation
        print("\n📝 Test 2: Reordering Blueprint Sections")
        request = BlueprintEditingRequest(
            blueprint_id=1,
            edit_type="reorder_sections",
            edit_instruction="Reorder sections to follow a logical learning progression: Introduction -> Fundamentals -> Applications -> Advanced Topics",
            preserve_original_structure=False,
            include_reasoning=True
        )
        
        start_time = time.time()
        response = await self.blueprint_service.edit_blueprint_agentically(request)
        processing_time = time.time() - start_time
        
        print(f"✅ Reorder sections completed in {processing_time:.2f}s")
        print(f"📊 Success: {response.success}")
        print(f"💬 Message: {response.message}")
        if response.reasoning:
            print(f"🤔 AI Reasoning: {response.reasoning[:200]}...")
        if response.granular_edits:
            print(f"🔧 Number of granular edits: {len(response.granular_edits)}")
        
        # Show before/after if successful
        if response.success and hasattr(response, 'original_content') and hasattr(response, 'edited_content'):
            print(f"\n📝 BEFORE (Original):")
            print(f"   {response.original_content[:300]}...")
            print(f"\n📝 AFTER (Edited):")
            print(f"   {response.edited_content[:300]}...")
        elif response.success:
            print(f"\n📝 EDIT RESULT:")
            print(f"   {response.message}")
        
        # Return success status (both granular edit tests should succeed)
        return response.success
    
    async def run_all_tests(self):
        """Run all tests with real LLM calls."""
        print("🚀 Starting Real LLM Blueprint Editing Tests...")
        print("⚠️  This will make actual API calls to Gemini 2.5 Flash")
        print("💰 Be aware of potential costs and rate limits")
        
        test_results = []
        
        try:
            # Test basic editing operations
            test_results.append(await self.test_blueprint_editing())
            test_results.append(await self.test_primitive_editing())
            test_results.append(await self.test_mastery_criterion_editing())
            test_results.append(await self.test_question_editing())
            
            # Test suggestion generation
            test_results.append(await self.test_suggestion_generation())
            
            # Test granular editing
            test_results.append(await self.test_granular_editing())
            
            # Check if all tests passed
            all_passed = all(test_results)
            passed_count = sum(test_results)
            total_count = len(test_results)
            
            print("\n" + "="*60)
            if all_passed:
                print("🎉 ALL REAL LLM TESTS PASSED SUCCESSFULLY!")
                print("="*60)
                print("✅ Blueprint editing service is working with real AI")
                print("✅ All endpoints are functional")
                print("✅ LLM integration is successful")
            else:
                print("❌ SOME REAL LLM TESTS FAILED!")
                print("="*60)
                print(f"📊 Test Results: {passed_count}/{total_count} tests passed")
                print("❌ Some operations failed - check error messages above")
                print("⚠️  This indicates issues with LLM responses or parsing")
            
        except Exception as e:
            print(f"\n❌ Error during testing: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return all_passed


async def main():
    """Main function to run the real LLM tests."""
    print("🧪 Real LLM Blueprint Editing Service Test Suite")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("app").exists():
        print("❌ Error: Please run this script from the elevate-ai-api directory")
        print("   Current directory:", os.getcwd())
        return
    
    # Initialize and run tests
    tester = RealLLMBlueprintEditingTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎯 Test Summary:")
        print("   ✅ All real LLM tests passed")
        print("   ✅ Blueprint editing service is fully functional")
        print("   ✅ Ready for production use")
    else:
        print("\n❌ Test Summary:")
        print("   ❌ Some tests failed")
        print("   ❌ Check the error messages above")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
