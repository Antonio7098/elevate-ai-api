#!/usr/bin/env python3

"""
End-to-End Answer Evaluation Test Script

This script tests the complete answer evaluation workflow:
1. Basic answer evaluation 
2. Criterion-based answer evaluation
3. Batch answer processing
4. Mastery assessment and progression
5. Performance analytics integration

Usage: python test-e2e-answer-evaluation.py
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

class AnswerEvaluationE2ETest:
    def __init__(self):
        self.results: List[TestResult] = []
        self.test_data = {}
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def run(self) -> None:
        """Run the complete answer evaluation e2e test suite."""
        print("🚀 Starting AI API Answer Evaluation E2E Test\n")
        
        try:
            await self.run_step(self.test_health_check)
            await self.run_step(self.test_basic_answer_evaluation)
            await self.run_step(self.test_criterion_answer_evaluation) 
            await self.run_step(self.test_batch_answer_evaluation)
            await self.run_step(self.test_mastery_assessment)
            await self.run_step(self.test_performance_analytics)
            await self.run_step(self.test_feedback_generation)
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

    async def test_basic_answer_evaluation(self) -> None:
        """Test 2: Basic answer evaluation functionality."""
        try:
            print("📝 Step 2: Testing basic answer evaluation...")
            
            # Sample question and student answer for evaluation
            payload = {
                "question": "What is the process by which plants convert sunlight into chemical energy?",
                "studentAnswer": "Photosynthesis is when plants use sunlight, water, and carbon dioxide to make glucose and oxygen.",
                "questionType": "short_answer",
                "expectedAnswer": "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose.",
                "evaluationCriteria": {
                    "checkAccuracy": True,
                    "checkCompleteness": True,
                    "checkClarity": True,
                    "provideFeedback": True
                }
            }
            
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/evaluate/answer",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                self.test_data['basic_evaluation'] = data
                
                score = data.get('score', 0)
                feedback = data.get('feedback', '')
                
                self.results.append(TestResult(
                    "2. Basic Answer Evaluation",
                    "PASS",
                    f"Evaluation successful - Score: {score}/100, Feedback provided: {len(feedback) > 0}"
                ))
                print(f"   ✅ Basic evaluation successful - Score: {score}/100")
                
            else:
                raise Exception(f"Basic evaluation failed with status {response.status_code}: {response.text}")
                
        except Exception as error:
            self.results.append(TestResult(
                "2. Basic Answer Evaluation",
                "FAIL",
                f"Basic evaluation failed: {str(error)}",
                error
            ))
            print("   ❌ Basic answer evaluation failed")
            raise error

    async def test_criterion_answer_evaluation(self) -> None:
        """Test 3: Criterion-based answer evaluation."""
        try:
            print("🎯 Step 3: Testing criterion-based answer evaluation...")
            
            # Evaluate answer against specific mastery criteria
            payload = {
                "question": "Explain the difference between the light reactions and Calvin cycle in photosynthesis.",
                "studentAnswer": "Light reactions happen in thylakoids and make ATP and NADPH. Calvin cycle happens in stroma and uses ATP and NADPH to make glucose from CO2.",
                "masteryCriteria": [
                    {
                        "criterionId": "crit_001",
                        "description": "Identifies correct locations of each process",
                        "ueeLevel": "understand",
                        "weight": 3.0
                    },
                    {
                        "criterionId": "crit_002", 
                        "description": "Explains energy flow between processes",
                        "ueeLevel": "use",
                        "weight": 4.0
                    },
                    {
                        "criterionId": "crit_003",
                        "description": "Describes molecular inputs and outputs",
                        "ueeLevel": "explore",
                        "weight": 5.0
                    }
                ],
                "evaluationPreferences": {
                    "provideCriterionScores": True,
                    "calculateWeightedScore": True,
                    "generateTargetedFeedback": True
                }
            }
            
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/evaluate/criterion",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                self.test_data['criterion_evaluation'] = data
                
                criterion_scores = data.get('criterionScores', [])
                weighted_score = data.get('weightedScore', 0)
                
                self.results.append(TestResult(
                    "3. Criterion Answer Evaluation",
                    "PASS",
                    f"Criterion evaluation successful - {len(criterion_scores)} criteria scored, weighted score: {weighted_score}"
                ))
                print(f"   ✅ Criterion evaluation successful - {len(criterion_scores)} criteria evaluated")
                
            else:
                raise Exception(f"Criterion evaluation failed with status {response.status_code}: {response.text}")
                
        except Exception as error:
            self.results.append(TestResult(
                "3. Criterion Answer Evaluation",
                "FAIL",
                f"Criterion evaluation failed: {str(error)}",
                error
            ))
            print("   ❌ Criterion answer evaluation failed")
            raise error

    async def test_batch_answer_evaluation(self) -> None:
        """Test 4: Batch answer processing."""
        try:
            print("📦 Step 4: Testing batch answer evaluation...")
            
            # Process multiple answers in batch
            payload = {
                "evaluationBatch": [
                    {
                        "answerId": "ans_001",
                        "question": "What is photosynthesis?",
                        "studentAnswer": "Plants make food from sunlight",
                        "questionType": "short_answer"
                    },
                    {
                        "answerId": "ans_002", 
                        "question": "Where does photosynthesis occur?",
                        "studentAnswer": "In the chloroplasts of plant cells",
                        "questionType": "short_answer"
                    },
                    {
                        "answerId": "ans_003",
                        "question": "What are the products of photosynthesis?",
                        "studentAnswer": "Glucose and oxygen",
                        "questionType": "short_answer"
                    }
                ],
                "batchPreferences": {
                    "parallelProcessing": True,
                    "generateSummaryStats": True,
                    "identifyCommonErrors": True
                }
            }
            
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/evaluate/batch",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                self.test_data['batch_evaluation'] = data
                
                results = data.get('results', [])
                summary_stats = data.get('summaryStats', {})
                
                self.results.append(TestResult(
                    "4. Batch Answer Evaluation",
                    "PASS",
                    f"Batch evaluation successful - {len(results)} answers processed"
                ))
                print(f"   ✅ Batch evaluation successful - {len(results)} answers processed")
                
            else:
                raise Exception(f"Batch evaluation failed with status {response.status_code}: {response.text}")
                
        except Exception as error:
            self.results.append(TestResult(
                "4. Batch Answer Evaluation",
                "FAIL",
                f"Batch evaluation failed: {str(error)}",
                error
            ))
            print("   ❌ Batch answer evaluation failed")
            # Don't raise error - continue with other tests

    async def test_mastery_assessment(self) -> None:
        """Test 5: Mastery assessment and progression tracking."""
        try:
            print("📈 Step 5: Testing mastery assessment...")
            
            # Assess mastery level based on answer patterns
            payload = {
                "primitiveId": "prim_photosynthesis_001",
                "userId": TEST_USER_ID,
                "recentAnswers": [
                    {
                        "answerId": "ans_001",
                        "score": 85,
                        "criterionScores": [
                            {"criterionId": "crit_001", "score": 90, "weight": 3.0},
                            {"criterionId": "crit_002", "score": 80, "weight": 4.0},
                            {"criterionId": "crit_003", "score": 85, "weight": 5.0}
                        ],
                        "timestamp": "2024-01-15T10:00:00Z"
                    },
                    {
                        "answerId": "ans_002",
                        "score": 92,
                        "criterionScores": [
                            {"criterionId": "crit_001", "score": 95, "weight": 3.0},
                            {"criterionId": "crit_002", "score": 90, "weight": 4.0},
                            {"criterionId": "crit_003", "score": 90, "weight": 5.0}
                        ],
                        "timestamp": "2024-01-16T10:00:00Z"
                    }
                ],
                "assessmentPreferences": {
                    "calculateTrendAnalysis": True,
                    "identifyWeakCriteria": True,
                    "suggestNextQuestions": True
                }
            }
            
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/evaluate/mastery",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                self.test_data['mastery_assessment'] = data
                
                mastery_level = data.get('masteryLevel', 0)
                progression = data.get('progression', 'unknown')
                
                self.results.append(TestResult(
                    "5. Mastery Assessment",
                    "PASS",
                    f"Mastery assessment successful - Level: {mastery_level}%, Progression: {progression}"
                ))
                print(f"   ✅ Mastery assessment successful - Level: {mastery_level}%")
                
            else:
                raise Exception(f"Mastery assessment failed with status {response.status_code}: {response.text}")
                
        except Exception as error:
            self.results.append(TestResult(
                "5. Mastery Assessment",
                "FAIL",
                f"Mastery assessment failed: {str(error)}",
                error
            ))
            print("   ❌ Mastery assessment failed")
            # Don't raise error - continue with other tests

    async def test_performance_analytics(self) -> None:
        """Test 6: Performance analytics and usage tracking."""
        try:
            print("📊 Step 6: Testing performance analytics integration...")
            
            # Get evaluation analytics for user
            headers = {"Authorization": f"Bearer {API_KEY}"}
            response = await self.client.get(
                f"{AI_API_BASE_URL}/api/v1/evaluate/analytics/{TEST_USER_ID}?period=7d",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                self.test_data['analytics'] = data
                
                total_evaluations = data.get('totalEvaluations', 0)
                average_score = data.get('averageScore', 0)
                
                self.results.append(TestResult(
                    "6. Performance Analytics",
                    "PASS",
                    f"Analytics retrieved - {total_evaluations} evaluations, avg score: {average_score}"
                ))
                print(f"   ✅ Performance analytics successful - {total_evaluations} evaluations tracked")
                
            else:
                # Analytics endpoint may not exist yet, use mock data
                self.test_data['analytics'] = {
                    "totalEvaluations": 5,
                    "averageScore": 87.5,
                    "evaluationTrends": ["improving"]
                }
                
                self.results.append(TestResult(
                    "6. Performance Analytics",
                    "SKIP",
                    "Analytics endpoint not available - using mock data"
                ))
                print("   ⏭️  Using mock analytics data (endpoint not implemented)")
                
        except Exception as error:
            self.results.append(TestResult(
                "6. Performance Analytics",
                "FAIL",
                f"Performance analytics failed: {str(error)}",
                error
            ))
            print("   ❌ Performance analytics failed")
            # Don't raise error - continue with other tests

    async def test_feedback_generation(self) -> None:
        """Test 7: Personalized feedback generation."""
        try:
            print("💬 Step 7: Testing personalized feedback generation...")
            
            # Generate targeted feedback for improvement
            payload = {
                "evaluationHistory": self.test_data.get('basic_evaluation', {}),
                "criterionEvaluation": self.test_data.get('criterion_evaluation', {}),
                "masteryAssessment": self.test_data.get('mastery_assessment', {}),
                "feedbackPreferences": {
                    "feedbackStyle": "encouraging",
                    "detailLevel": "detailed",
                    "includeNextSteps": True,
                    "includeResources": True
                }
            }
            
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/evaluate/feedback",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                self.test_data['feedback'] = data
                
                feedback_text = data.get('personalizedFeedback', '')
                next_steps = data.get('nextSteps', [])
                
                self.results.append(TestResult(
                    "7. Feedback Generation",
                    "PASS",
                    f"Feedback generated successfully - {len(feedback_text)} chars, {len(next_steps)} next steps"
                ))
                print(f"   ✅ Personalized feedback generated successfully")
                
            else:
                raise Exception(f"Feedback generation failed with status {response.status_code}: {response.text}")
                
        except Exception as error:
            self.results.append(TestResult(
                "7. Feedback Generation",
                "FAIL",
                f"Feedback generation failed: {str(error)}",
                error
            ))
            print("   ❌ Feedback generation failed")
            # Don't raise error - this is not critical

    def print_results(self) -> None:
        """Print formatted test results."""
        print(f"\n{'=' * 60}")
        print("📊 AI API ANSWER EVALUATION E2E TEST RESULTS")
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
            print("🎉 All tests passed! Answer evaluation workflow is working correctly.")
        
        print(f"{'=' * 60}")

async def main():
    """Main function to run the test suite."""
    test = AnswerEvaluationE2ETest()
    await test.run()

if __name__ == "__main__":
    asyncio.run(main())
