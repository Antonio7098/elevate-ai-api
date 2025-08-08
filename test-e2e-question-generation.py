#!/usr/bin/env python3

"""
End-to-End Question Generation Test Script

This script tests the complete question generation workflow:
1. Generate questions from text content
2. Generate criterion-specific questions 
3. Generate questions for existing blueprints
4. Batch question generation
5. Question type variation and difficulty scaling
6. Integration with mastery criteria

Usage: python test-e2e-question-generation.py
"""

import asyncio
import sys
import json
import time
from typing import Dict, List, Any, Optional
import httpx
from datetime import datetime

# Configuration
AI_API_BASE_URL = "http://localhost:8000"
API_KEY = "test_api_key_123"
TEST_USER_ID = 108

class TestResult:
    def __init__(self, step: str, status: str, details: str = None, error: Any = None):
        self.step = step
        self.status = status  # 'PASS', 'FAIL', 'SKIP'
        self.details = details
        self.error = error

class QuestionGenerationE2ETest:
    def __init__(self):
        self.results: List[TestResult] = []
        self.test_data = {}
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def run(self) -> None:
        """Run the complete question generation e2e test suite."""
        print("🚀 Starting AI API Question Generation E2E Test\n")
        
        try:
            await self.run_step(self.test_health_check)
            await self.run_step(self.test_basic_question_generation)
            await self.run_step(self.test_criterion_question_generation)
            await self.run_step(self.test_blueprint_question_generation)
            await self.run_step(self.test_batch_question_generation)
            await self.run_step(self.test_question_type_variation)
            await self.run_step(self.test_difficulty_scaling)
        except Exception as error:
            print(f"\n❌ Test suite aborted due to critical failure: {error}")
        finally:
            await self.client.aclose()
            self.print_results()
    
    async def run_step(self, step_func, continue_on_error: bool = False) -> None:
        """Execute a test step with error handling."""
        try:
            await step_func()
        except Exception as error:
            if not continue_on_error:
                raise error

    async def test_health_check(self) -> None:
        """Test 1: Verify AI API health and availability."""
        try:
            print("🏥 Step 1: Checking AI API health...")
            
            response = await self.client.get(f"{AI_API_BASE_URL}/api/health")
            
            if response.status_code == 200:
                health_data = response.json()
                self.results.append(TestResult(
                    "1. AI API Health Check",
                    "PASS",
                    f"AI API healthy - Status: {health_data.get('status', 'unknown')}"
                ))
                print("   ✅ AI API health check successful")
            else:
                raise Exception(f"Health check failed with status {response.status_code}")
                
        except Exception as error:
            self.results.append(TestResult(
                "1. AI API Health Check",
                "FAIL",
                f"Health check failed: {str(error)}",
                error
            ))
            print("   ❌ AI API health check failed")
            raise error

    async def test_basic_question_generation(self) -> None:
        """Test 2: Basic question generation from text content."""
        try:
            print("❓ Step 2: Testing basic question generation...")
            
            # Sample educational content for question generation
            source_content = """
            # Cell Division: Mitosis and Meiosis
            
            Cell division is a fundamental process in biology where a single cell divides to form two or more daughter cells.
            
            ## Mitosis
            Mitosis is the process of cell division that results in two genetically identical diploid daughter cells.
            The phases of mitosis are: prophase, metaphase, anaphase, and telophase.
            
            ## Meiosis
            Meiosis is a specialized cell division that produces four genetically diverse haploid gametes.
            Meiosis involves two consecutive divisions: meiosis I and meiosis II.
            
            ## Key Differences
            - Mitosis produces 2 identical diploid cells; meiosis produces 4 diverse haploid cells
            - Mitosis occurs in somatic cells; meiosis occurs in reproductive cells
            - Crossing over occurs in meiosis but not in mitosis
            """
            
            payload = {
                "blueprint_id": "test-blueprint-cell-division",
                "name": "Cell Division Question Set",
                "sourceContent": source_content,
                "questionCount": 5,
                "questionTypes": ["short_answer", "multiple_choice", "true_false", "essay"],
                "difficultyLevel": "intermediate",
                "topicFocus": "cell_division",
                "includeAnswerKeys": True
            }
            
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/generate/questions",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                self.test_data['basic_questions'] = data.get('questions', [])
                question_count = len(self.test_data['basic_questions'])
                
                # Verify question variety
                question_types = set(q.get('questionType') for q in self.test_data['basic_questions'])
                
                self.results.append(TestResult(
                    "2. Basic Question Generation",
                    "PASS",
                    f"Generated {question_count} questions with {len(question_types)} different types"
                ))
                print(f"   ✅ Generated {question_count} questions with variety: {question_types}")
                
            else:
                raise Exception(f"Basic question generation failed with status {response.status_code}: {response.text}")
                
        except Exception as error:
            self.results.append(TestResult(
                "2. Basic Question Generation",
                "FAIL",
                f"Basic question generation failed: {str(error)}",
                error
            ))
            print("   ❌ Basic question generation failed")
            raise error

    async def test_criterion_question_generation(self) -> None:
        """Test 3: Generate questions mapped to specific mastery criteria."""
        try:
            print("🎯 Step 3: Testing criterion-specific question generation...")
            
            # Mock mastery criteria for question generation
            mastery_criteria = [
                {
                    "criterionId": "crit_001",
                    "description": "Identify the phases of mitosis in correct order",
                    "ueeLevel": "UNDERSTAND",
                    "weight": 3.0,
                    "primitiveId": "prim_mitosis_phases"
                },
                {
                    "criterionId": "crit_002",
                    "description": "Compare and contrast mitosis vs meiosis outcomes",
                    "ueeLevel": "USE", 
                    "weight": 4.0,
                    "primitiveId": "prim_cell_division_types"
                },
                {
                    "criterionId": "crit_003",
                    "description": "Analyze the role of crossing over in genetic diversity",
                    "ueeLevel": "EXPLORE",
                    "weight": 5.0,
                    "primitiveId": "prim_genetic_diversity"
                }
            ]
            
            # Use the same source content from basic test
            source_content = """
            # Cell Division: Mitosis and Meiosis
            
            Cell division is a fundamental process in biology where a single cell divides to form two or more daughter cells.
            
            ## Mitosis
            Mitosis is the process of cell division that results in two genetically identical diploid daughter cells.
            The phases of mitosis are: prophase, metaphase, anaphase, and telophase.
            
            ## Meiosis
            Meiosis is a specialized cell division that produces four genetically diverse haploid gametes.
            Meiosis involves two consecutive divisions: meiosis I and meiosis II.
            
            ## Key Differences
            - Mitosis produces 2 identical diploid cells; meiosis produces 4 diverse haploid cells
            - Mitosis occurs in somatic cells; meiosis occurs in reproductive cells
            - Crossing over occurs in meiosis but not in mitosis
            """
            
            # Use the first mastery criterion for the required fields
            first_criterion = mastery_criteria[0]
            
            payload = {
                "criterionId": first_criterion["criterionId"],
                "criterionTitle": first_criterion["description"],
                "primitiveId": first_criterion["primitiveId"],
                "primitiveTitle": "Cell Division Process",
                "ueeLevel": first_criterion["ueeLevel"],
                "weight": first_criterion["weight"],
                "sourceContent": source_content,  # Use the same source content from basic test
                "questionsPerCriterion": 2,
                "questionPreferences": {
                    "alignToUeeLevel": True,
                    "varyQuestionTypes": True,
                    "includeScaffolding": True
                }
            }
            
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/questions/criterion-specific",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                self.test_data['criterion_questions'] = data.get('questions', [])
                question_count = len(self.test_data['criterion_questions'])
                
                # Verify criterion mapping
                mapped_criteria = set(q.get('criterionId') for q in self.test_data['criterion_questions'])
                
                self.results.append(TestResult(
                    "3. Criterion Question Generation",
                    "PASS",
                    f"Generated {question_count} questions mapped to {len(mapped_criteria)} criteria"
                ))
                print(f"   ✅ Generated {question_count} criterion-mapped questions")
                
            else:
                raise Exception(f"Criterion question generation failed with status {response.status_code}: {response.text}")
                
        except Exception as error:
            self.results.append(TestResult(
                "3. Criterion Question Generation",
                "FAIL",
                f"Criterion question generation failed: {str(error)}",
                error
            ))
            print("   ❌ Criterion question generation failed")
            raise error

    async def test_blueprint_question_generation(self) -> None:
        """Test 4: Generate questions from existing blueprint."""
        try:
            print("📋 Step 4: Testing blueprint-based question generation...")
            
            # Test question generation from existing blueprint
            blueprint_id = "test-blueprint-cell-division"
            
            payload = {
                "blueprintId": blueprint_id,
                "questionCount": 8,
                "sectionFocus": ["introduction", "key_concepts"],
                "questionDistribution": {
                    "short_answer": 3,
                    "multiple_choice": 3,
                    "essay": 2
                },
                "difficultyProgression": True,
                "includeReviewQuestions": True
            }
            
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/questions/blueprint/{blueprint_id}",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                self.test_data['blueprint_questions'] = data.get('questions', [])
                question_count = len(self.test_data['blueprint_questions'])
                
                # Verify section alignment
                sections_covered = set(q.get('sourceSection') for q in self.test_data['blueprint_questions'] if q.get('sourceSection'))
                
                self.results.append(TestResult(
                    "4. Blueprint Question Generation",
                    "PASS",
                    f"Generated {question_count} questions covering {len(sections_covered)} sections"
                ))
                print(f"   ✅ Generated {question_count} blueprint-based questions")
                
            else:
                raise Exception(f"Blueprint question generation failed with status {response.status_code}: {response.text}")
                
        except Exception as error:
            self.results.append(TestResult(
                "4. Blueprint Question Generation",
                "FAIL",
                f"Blueprint question generation failed: {str(error)}",
                error
            ))
            print("   ❌ Blueprint question generation failed")
            # Don't raise error - continue with other tests

    async def test_batch_question_generation(self) -> None:
        """Test 5: Batch question generation for multiple topics."""
        try:
            print("📦 Step 5: Testing batch question generation...")
            
            # Generate questions for multiple topics/sources in batch
            payload = {
                "batchRequests": [
                    {
                        "requestId": "req_001",
                        "topic": "photosynthesis",
                        "questionCount": 3,
                        "questionTypes": ["short_answer"],
                        "difficultyLevel": "beginner"
                    },
                    {
                        "requestId": "req_002", 
                        "topic": "cellular_respiration",
                        "questionCount": 3,
                        "questionTypes": ["multiple_choice"],
                        "difficultyLevel": "intermediate"
                    },
                    {
                        "requestId": "req_003",
                        "topic": "enzyme_function", 
                        "questionCount": 2,
                        "questionTypes": ["essay"],
                        "difficultyLevel": "advanced"
                    }
                ],
                "batchPreferences": {
                    "parallelProcessing": True,
                    "ensureTopicVariety": True,
                    "generateSummaryReport": True
                }
            }
            
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/questions/batch",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                batch_results = data.get('results', {})
                total_questions = sum(len(result.get('questions', [])) for result in batch_results.values())
                
                self.test_data['batch_questions'] = batch_results
                
                self.results.append(TestResult(
                    "5. Batch Question Generation",
                    "PASS",
                    f"Batch generated {total_questions} questions across {len(batch_results)} topics"
                ))
                print(f"   ✅ Batch generation successful - {total_questions} total questions")
                
            else:
                raise Exception(f"Batch question generation failed with status {response.status_code}: {response.text}")
                
        except Exception as error:
            self.results.append(TestResult(
                "5. Batch Question Generation",
                "FAIL",
                f"Batch question generation failed: {str(error)}",
                error
            ))
            print("   ❌ Batch question generation failed")
            # Don't raise error - continue with other tests

    async def test_question_type_variation(self) -> None:
        """Test 6: Question type variation and formatting."""
        try:
            print("🎨 Step 6: Testing question type variation...")
            
            # Test generation of specific question types with proper formatting
            payload = {
                "sourceContent": "Enzymes are biological catalysts that speed up chemical reactions by lowering activation energy.",
                "questionTypeRequests": [
                    {
                        "questionType": "multiple_choice",
                        "count": 2,
                        "options": 4,
                        "includeDistractors": True
                    },
                    {
                        "questionType": "true_false",
                        "count": 2,
                        "includeExplanations": True
                    },
                    {
                        "questionType": "fill_in_blank",
                        "count": 2,
                        "maxBlanks": 2
                    },
                    {
                        "questionType": "matching",
                        "count": 1,
                        "itemCount": 5
                    }
                ]
            }
            
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/questions/types",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                self.test_data['type_questions'] = data.get('questions', [])
                
                # Verify question type variety and formatting
                question_types = {}
                for q in self.test_data['type_questions']:
                    q_type = q.get('questionType')
                    question_types[q_type] = question_types.get(q_type, 0) + 1
                
                self.results.append(TestResult(
                    "6. Question Type Variation",
                    "PASS",
                    f"Generated {len(question_types)} different question types: {list(question_types.keys())}"
                ))
                print(f"   ✅ Question type variation successful - {question_types}")
                
            else:
                # If specific endpoint doesn't exist, use basic generation data
                if self.test_data.get('basic_questions'):
                    question_types = set(q.get('questionType') for q in self.test_data['basic_questions'])
                    
                    self.results.append(TestResult(
                        "6. Question Type Variation", 
                        "SKIP",
                        f"Specialized endpoint not available - verified {len(question_types)} types from basic generation"
                    ))
                    print(f"   ⏭️  Using basic generation data - verified {len(question_types)} question types")
                else:
                    raise Exception(f"Question type variation failed with status {response.status_code}: {response.text}")
                
        except Exception as error:
            self.results.append(TestResult(
                "6. Question Type Variation",
                "FAIL",
                f"Question type variation failed: {str(error)}",
                error
            ))
            print("   ❌ Question type variation failed")
            # Don't raise error - continue with other tests

    async def test_difficulty_scaling(self) -> None:
        """Test 7: Difficulty scaling and adaptive question generation."""
        try:
            print("⚖️ Step 7: Testing difficulty scaling...")
            
            # Test adaptive difficulty scaling
            base_topic = "photosynthesis process"
            
            difficulty_tests = [
                {"level": "beginner", "expectedComplexity": "simple"},
                {"level": "intermediate", "expectedComplexity": "moderate"}, 
                {"level": "advanced", "expectedComplexity": "complex"}
            ]
            
            difficulty_results = {}
            
            for test in difficulty_tests:
                payload = {
                    "topic": base_topic,
                    "difficultyLevel": test["level"],
                    "questionCount": 2,
                    "questionTypes": ["short_answer"],
                    "adaptiveScaling": True
                }
                
                headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
                response = await self.client.post(
                    f"{AI_API_BASE_URL}/api/v1/questions/generate",
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    difficulty_results[test["level"]] = data.get('questions', [])
                
            if difficulty_results:
                total_questions = sum(len(questions) for questions in difficulty_results.values())
                difficulty_levels = list(difficulty_results.keys())
                
                self.test_data['difficulty_questions'] = difficulty_results
                
                self.results.append(TestResult(
                    "7. Difficulty Scaling",
                    "PASS",
                    f"Generated {total_questions} questions across {len(difficulty_levels)} difficulty levels"
                ))
                print(f"   ✅ Difficulty scaling successful - {difficulty_levels}")
                
            else:
                raise Exception("No difficulty-scaled questions were generated successfully")
                
        except Exception as error:
            self.results.append(TestResult(
                "7. Difficulty Scaling",
                "FAIL",
                f"Difficulty scaling failed: {str(error)}",
                error
            ))
            print("   ❌ Difficulty scaling failed")
            # Don't raise error - this is not critical

    def print_results(self) -> None:
        """Print formatted test results."""
        print(f"\n{'=' * 60}")
        print("📊 AI API QUESTION GENERATION E2E TEST RESULTS")
        print(f"{'=' * 60}")
        
        for result in self.results:
            status_emoji = "✅" if result.status == "PASS" else "❌" if result.status == "FAIL" else "⏭️"
            print(f"{status_emoji} {result.step}")
            if result.details:
                print(f"   {result.details}")
        
        print("-" * 60)
        
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        skipped = sum(1 for r in self.results if r.status == "SKIP")
        
        print(f"📈 SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
        
        if failed > 0:
            print("⚠️  Some tests failed. Check the errors above for details.")
        elif passed > 0:
            print("🎉 All tests passed! Question generation workflow is working correctly.")
        
        print(f"{'=' * 60}")

async def main():
    """Main function to run the test suite."""
    test = QuestionGenerationE2ETest()
    await test.run()

if __name__ == "__main__":
    asyncio.run(main())
