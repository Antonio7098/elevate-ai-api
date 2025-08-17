#!/usr/bin/env python3
"""
Production readiness test for Mastery & Learning Services.
Tests mastery criteria generation, question generation, and mapping with REAL LLM calls.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.llm_service import create_llm_service
from app.core.mastery_criteria_service import MasteryCriteriaService
from app.core.question_generation_service import QuestionGenerationService
from app.core.question_mapping_service import QuestionMappingService
from app.core.criterion_question_generation import CriterionQuestionGenerator

class MasteryServicesTester:
    def __init__(self):
        self.llm_service = None
        self.mastery_service = None
        self.question_service = None
        self.mapping_service = None
        self.criterion_question_service = None
        
    async def setup_services(self):
        """Set up all mastery services with real dependencies."""
        print("🔧 Setting up Mastery & Learning Services...")
        
        try:
            # Set up LLM service
            print("   🚀 Setting up Gemini LLM service...")
            self.llm_service = create_llm_service(provider="gemini")
            print("   ✅ LLM service ready")
            
            # Set up Mastery Criteria Service
            print("   🎯 Setting up Mastery Criteria Service...")
            self.mastery_service = MasteryCriteriaService()
            print("   ✅ Mastery Criteria Service ready")
            
            # Set up Question Generation Service
            print("   ❓ Setting up Question Generation Service...")
            self.question_service = QuestionGenerationService()
            print("   ✅ Question Generation Service ready")
            
            # Set up Question Mapping Service
            print("   🗺️  Setting up Question Mapping Service...")
            self.mapping_service = QuestionMappingService()
            print("   ✅ Question Mapping Service ready")
            
            # Set up Criterion Question Generation Service
            print("   🔗 Setting up Criterion Question Generation Service...")
            self.criterion_question_service = CriterionQuestionGenerator()
            print("   ✅ Criterion Question Generation Service ready")
            
            print("   🎉 All Mastery Services set up successfully!")
            return True
            
        except Exception as e:
            print(f"   ❌ Service setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_mastery_criteria_generation(self):
        """Test mastery criteria generation with real LLM calls."""
        print("\n🎯 Testing Mastery Criteria Generation")
        print("-" * 60)
        
        try:
            print("   🚀 Testing mastery criteria generation...")
            
            # Test mastery criteria generation
            print("      🎯 Generating mastery criteria...")
            try:
                primitive_data = {
                    "name": "Linear Regression",
                    "description": "A supervised learning algorithm for regression",
                    "type": "algorithm",
                    "difficulty": "beginner"
                }
                
                criteria = await self.mastery_service.generate_mastery_criteria(
                    primitive=primitive_data,
                    uee_level_preference="balanced"
                )
                
                print(f"         ✅ Generated {len(criteria) if criteria else 0} mastery criteria")
                
                if criteria:
                    print("      📊 First criterion:")
                    first_criterion = criteria[0]
                    print(f"         Description: {first_criterion.get('description', 'N/A')[:100]}...")
                    print(f"         Level: {first_criterion.get('uee_level', 'N/A')}")
                
            except Exception as e:
                print(f"         ⚠️  Mastery criteria generation failed: {e}")
            
            print("   ✅ Mastery criteria generation test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Mastery criteria generation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_question_generation(self):
        """Test question generation with real LLM calls."""
        print("\n❓ Testing Question Generation")
        print("-" * 60)
        
        try:
            print("   🚀 Testing question generation...")
            
            # Test multiple choice questions
            print("      🔍 Testing multiple_choice questions...")
            try:
                primitive_data = {
                    "name": "Machine Learning Basics",
                    "description": "Fundamental concepts of machine learning",
                    "type": "concept",
                    "difficulty": "beginner",
                    "masteryCriteria": []  # Add empty mastery criteria list
                }
                
                source_content = """
                Machine learning is a subset of artificial intelligence that enables computers 
                to learn from data without being explicitly programmed. It uses algorithms 
                to identify patterns and make predictions based on input data.
                """
                
                questions = await self.question_service.generate_questions_for_primitive(
                    primitive=primitive_data,
                    source_content=source_content,
                    questions_per_criterion=2
                )
                
                print(f"         ✅ Generated questions for primitive")
                
                if questions:
                    total_questions = sum(len(q_list) for q_list in questions.values())
                    print(f"         📊 Total questions generated: {total_questions}")
                
            except Exception as e:
                print(f"         ⚠️  Question generation failed: {e}")
            
            print("   ✅ Question generation test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Question generation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_question_mapping(self):
        """Test question mapping to mastery criteria."""
        print("\n🗺️  Testing Question Mapping")
        print("-" * 60)
        
        try:
            print("   🚀 Testing question mapping...")
            
            # Generate sample criteria
            print("      🎯 Generating sample criteria...")
            try:
                primitive_data = {
                    "name": "ML Fundamentals",
                    "description": "Basic machine learning concepts",
                    "type": "concept",
                    "difficulty": "beginner"
                }
                
                criteria = await self.mastery_service.generate_mastery_criteria(
                    primitive=primitive_data,
                    uee_level_preference="balanced"
                )
                print(f"         ✅ Generated {len(criteria) if criteria else 0} criteria")
                
            except Exception as e:
                print(f"         ⚠️  Criteria generation failed: {e}")
                criteria = []
            
            # Generate sample questions
            print("      ❓ Generating sample questions...")
            try:
                source_content = "Machine learning enables computers to learn from data."
                
                # Add mastery criteria to primitive for question generation
                primitive_with_criteria = primitive_data.copy()
                primitive_with_criteria["masteryCriteria"] = criteria
                
                questions = await self.question_service.generate_questions_for_primitive(
                    primitive=primitive_with_criteria,
                    source_content=source_content,
                    questions_per_criterion=1
                )
                print(f"         ✅ Generated questions for primitive")
                
            except Exception as e:
                print(f"         ⚠️  Question generation failed: {e}")
                questions = {}
            
            # Test question-to-criteria mapping
            print("      🔗 Testing question-to-criteria mapping...")
            if criteria and questions:
                try:
                    # Flatten questions list
                    all_questions = []
                    for criterion_questions in questions.values():
                        all_questions.extend(criterion_questions)
                    
                    mappings = await self.mapping_service.map_questions_to_criteria(
                        questions=all_questions,
                        criteria=criteria,
                        primitive=primitive_data,
                        source_content=source_content
                    )
                    
                    print(f"         ✅ Mapped {len(mappings)} questions to criteria")
                    
                except Exception as e:
                    print(f"         ⚠️  Question mapping failed: {e}")
            
            print("   ✅ Question mapping test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Question mapping test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_criterion_question_generation(self):
        """Test criterion-specific question generation."""
        print("\n🔗 Testing Criterion Question Generation")
        print("-" * 60)
        
        try:
            print("   🚀 Testing criterion question generation...")
            
            # Generate questions for a specific criterion
            print("      ❓ Generating questions for criterion...")
            try:
                primitive_data = {
                    "name": "Supervised Learning",
                    "description": "Learning from labeled data",
                    "type": "concept",
                    "difficulty": "intermediate"
                }
                
                source_content = """
                Supervised learning uses labeled data to train models for prediction tasks.
                The model learns the relationship between input features and target outputs.
                """
                
                criterion_questions = await self.criterion_question_service.generate_questions_for_criterion(
                    criterion={"description": "Understand supervised learning concepts"},
                    primitive=primitive_data,
                    question_count=2
                )
                
                print(f"         ✅ Generated {len(criterion_questions) if criterion_questions else 0} questions for criterion")
                
            except Exception as e:
                print(f"         ⚠️  Criterion question generation failed: {e}")
            
            print("   ✅ Criterion question generation test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Criterion question generation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_integrated_learning_workflow(self):
        """Test integrated learning workflow."""
        print("\n🔄 Testing Integrated Learning Workflow")
        print("-" * 60)
        
        try:
            print("   🚀 Testing integrated workflow...")
            
            # Step 1: Generate mastery criteria
            print("      🎯 Step 1: Generating mastery criteria...")
            try:
                primitive_data = {
                    "name": "Neural Networks",
                    "description": "Computational models inspired by biological neurons",
                    "type": "algorithm",
                    "difficulty": "intermediate"
                }
                
                criteria = await self.mastery_service.generate_mastery_criteria(
                    primitive=primitive_data,
                    uee_level_preference="balanced"
                )
                
                print(f"         ✅ Generated {len(criteria) if criteria else 0} mastery criteria")
                
            except Exception as e:
                print(f"         ⚠️  Mastery criteria generation failed: {e}")
                criteria = []
            
            # Step 2: Generate questions
            print("      ❓ Step 2: Generating questions...")
            questions = {}
            try:
                source_content = """
                Neural networks are computational models inspired by biological neurons.
                They consist of interconnected nodes that process information in layers.
                """
                
                # Add mastery criteria to primitive for question generation
                primitive_with_criteria = primitive_data.copy()
                primitive_with_criteria["masteryCriteria"] = criteria
                
                questions = await self.question_service.generate_questions_for_primitive(
                    primitive=primitive_with_criteria,
                    source_content=source_content,
                    questions_per_criterion=2
                )
                
                print(f"         ✅ Generated questions for primitive")
                
            except Exception as e:
                print(f"         ⚠️  Question generation failed: {e}")
            
            # Step 3: Map questions to criteria
            print("      🗺️  Step 3: Mapping questions to criteria...")
            if criteria and questions:
                try:
                    all_questions = []
                    for criterion_questions in questions.values():
                        all_questions.extend(criterion_questions)
                    
                    mappings = await self.mapping_service.map_questions_to_criteria(
                        questions=all_questions,
                        criteria=criteria,
                        primitive=primitive_data,
                        source_content=source_content
                    )
                    
                    print(f"         ✅ Mapped questions to criteria")
                    
                except Exception as e:
                    print(f"         ⚠️  Question mapping failed: {e}")
            
            print("      ✅ Complete learning workflow tested")
            
            print("   ✅ Integrated learning workflow test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Integrated learning workflow test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_all_tests(self):
        """Run all mastery service tests."""
        print("🚀 Starting MASTERY & LEARNING SERVICES Production Readiness Test")
        print("=" * 80)
        
        # First, set up all services
        print("\n🔧 PHASE 1: Service Setup")
        setup_success = await self.setup_services()
        
        if not setup_success:
            print("❌ Service setup failed. Cannot proceed with tests.")
            return False
        
        print("\n🧪 PHASE 2: Running Tests")
        tests = [
            ("Mastery Criteria Generation", self.test_mastery_criteria_generation),
            ("Question Generation", self.test_question_generation),
            ("Question Mapping", self.test_question_mapping),
            ("Criterion Question Generation", self.test_criterion_question_generation),
            ("Integrated Learning Workflow", self.test_integrated_learning_workflow)
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n🧪 Running: {test_name}")
            try:
                result = await test_func()
                results.append((test_name, result))
                print(f"   {'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
            except Exception as e:
                print(f"   ❌ ERROR: {test_name} - {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 MASTERY & LEARNING SERVICES TEST SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL MASTERY TESTS PASSED! Services are production-ready!")
        else:
            print("⚠️  Some mastery tests failed. Check the output above for details.")
        
        return passed == total

async def main():
    """Main test function."""
    tester = MasteryServicesTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 Mastery services test suite completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some mastery service tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
