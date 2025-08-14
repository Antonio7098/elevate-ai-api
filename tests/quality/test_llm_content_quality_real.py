#!/usr/bin/env python3
"""
Real LLM Content Quality Testing Framework
Makes actual API calls to LLM services and validates real generated content.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import statistics
import httpx
from openai import AsyncOpenAI
import google.generativeai as genai

@dataclass
class ContentQualityMetrics:
    """Metrics for content quality assessment"""
    completeness_score: float  # 0.0 - 1.0
    coherence_score: float     # 0.0 - 1.0
    relevance_score: float     # 0.0 - 1.0
    accuracy_score: float      # 0.0 - 1.0
    overall_score: float       # 0.0 - 1.0

@dataclass
class CoverageMetrics:
    """Metrics for content coverage assessment"""
    primitive_types_covered: Set[str]
    uue_stages_covered: Set[str]
    assessment_types_covered: Set[str]
    difficulty_levels_covered: Set[str]
    coverage_score: float  # 0.0 - 1.0

@dataclass
class QualityTestResult:
    """Result of a quality test"""
    test_name: str
    success: bool
    quality_metrics: Optional[ContentQualityMetrics] = None
    coverage_metrics: Optional[CoverageMetrics] = None
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class RealLLMContentQualityTester:
    """Real LLM tester that makes actual API calls"""
    
    def __init__(self):
        self.test_results: List[QualityTestResult] = []
        
        # Initialize LLM clients
        self.openai_client = None
        self.gemini_client = None
        
        # Load API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY")
        
        if self.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_client = genai.GenerativeModel('gemini-1.5-pro')
        
        # Expected primitive types
        self.expected_primitive_types = {"entity", "proposition", "process"}
        
        # Expected UUE stages
        self.expected_uue_stages = {"UNDERSTAND", "USE", "EXPLORE"}
        
        # Expected assessment types
        self.expected_assessment_types = {
            "QUESTION_BASED", "EXPLANATION_BASED", 
            "APPLICATION_BASED", "MULTIMODAL"
        }
        
        # Expected difficulty levels
        self.expected_difficulty_levels = {"BEGINNER", "INTERMEDIATE", "ADVANCED"}
    
    async def generate_content_with_openai(self, prompt: str) -> Dict[str, Any]:
        """Generate content using OpenAI API"""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert educational content creator. Generate high-quality learning content in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            # Parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise Exception("Could not parse JSON response from OpenAI")
                    
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {e}")
    
    async def generate_content_with_gemini(self, prompt: str) -> Dict[str, Any]:
        """Generate content using Gemini API"""
        if not self.gemini_client:
            raise Exception("Gemini client not initialized")
        
        try:
            response = self.gemini_client.generate_content(prompt)
            content = response.text
            
            # Parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise Exception("Could not parse JSON response from Gemini")
                    
        except Exception as e:
            raise Exception(f"Gemini API call failed: {e}")
    
    async def test_real_llm_primitive_generation(self, llm_provider: str = "openai") -> QualityTestResult:
        """Test real LLM generation of primitive types"""
        print(f"🧪 Testing Real LLM ({llm_provider.upper()}) Primitive Generation...")
        
        try:
            # Create a prompt for generating primitives
            prompt = """
            Generate educational content about "Machine Learning Fundamentals" in the following JSON format:
            {
                "primitives": [
                    {"type": "entity", "title": "Machine Learning", "description": "..."},
                    {"type": "proposition", "title": "Supervised Learning", "description": "..."},
                    {"type": "process", "title": "Model Training", "description": "..."}
                ],
                "mastery_criteria": [
                    {"title": "Understand ML Basics", "uue_stage": "UNDERSTAND", "difficulty": "BEGINNER"},
                    {"title": "Apply ML Concepts", "uue_stage": "USE", "difficulty": "INTERMEDIATE"}
                ],
                "questions": [
                    {"text": "What is machine learning?", "type": "definition"},
                    {"text": "How does supervised learning work?", "type": "explanation"}
                ]
            }
            
            Ensure you generate exactly 3 primitives covering all three types: entity, proposition, and process.
            """
            
            # Generate content using the specified LLM
            if llm_provider.lower() == "openai":
                content = await self.generate_content_with_openai(prompt)
            elif llm_provider.lower() == "gemini":
                content = await self.generate_content_with_gemini(prompt)
            else:
                raise Exception(f"Unsupported LLM provider: {llm_provider}")
            
            # Validate the generated content
            primitives = content.get("primitives", [])
            if not primitives:
                return QualityTestResult(
                    test_name=f"real_llm_primitive_generation_{llm_provider}",
                    success=False,
                    error_message="No primitives generated by LLM"
                )
            
            # Extract primitive types
            generated_types = set()
            for primitive in primitives:
                if isinstance(primitive, dict) and "type" in primitive:
                    generated_types.add(primitive["type"])
            
            # Check coverage
            missing_types = self.expected_primitive_types - generated_types
            coverage_score = len(generated_types) / len(self.expected_primitive_types)
            
            success = len(missing_types) == 0
            
            if success:
                print(f"  ✅ {llm_provider.upper()} generated all primitive types: {generated_types}")
            else:
                print(f"  ❌ {llm_provider.upper()} missing primitive types: {missing_types}")
            
            return QualityTestResult(
                test_name=f"real_llm_primitive_generation_{llm_provider}",
                success=success,
                coverage_metrics=CoverageMetrics(
                    primitive_types_covered=generated_types,
                    uue_stages_covered=set(),
                    assessment_types_covered=set(),
                    difficulty_levels_covered=set(),
                    coverage_score=coverage_score
                ),
                details={
                    "llm_provider": llm_provider,
                    "expected_types": list(self.expected_primitive_types),
                    "generated_types": list(generated_types),
                    "missing_types": list(missing_types),
                    "coverage_score": coverage_score,
                    "raw_content": content
                }
            )
            
        except Exception as e:
            return QualityTestResult(
                test_name=f"real_llm_primitive_generation_{llm_provider}",
                success=False,
                error_message=str(e)
            )
    
    async def test_real_llm_content_completeness(self, llm_provider: str = "openai") -> QualityTestResult:
        """Test real LLM content completeness"""
        print(f"🧪 Testing Real LLM ({llm_provider.upper()}) Content Completeness...")
        
        try:
            # Create a prompt for comprehensive content
            prompt = """
            Generate comprehensive educational content about "Neural Networks" in the following JSON format:
            {
                "primitives": [
                    {"type": "entity", "title": "Neural Network", "description": "..."},
                    {"type": "proposition", "title": "Backpropagation", "description": "..."},
                    {"type": "process", "title": "Training Process", "description": "..."}
                ],
                "mastery_criteria": [
                    {"title": "Understand Neural Networks", "uue_stage": "UNDERSTAND", "difficulty": "BEGINNER"},
                    {"title": "Implement Basic NN", "uue_stage": "USE", "difficulty": "INTERMEDIATE"},
                    {"title": "Design Advanced Architectures", "uue_stage": "EXPLORE", "difficulty": "ADVANCED"}
                ],
                "questions": [
                    {"text": "What is a neural network?", "type": "definition"},
                    {"text": "How does backpropagation work?", "type": "explanation"},
                    {"text": "What are the key components?", "type": "analysis"}
                ]
            }
            
            Ensure you generate at least 3 primitives, 3 mastery criteria covering all UUE stages, and 3 questions.
            """
            
            # Generate content
            if llm_provider.lower() == "openai":
                content = await self.generate_content_with_openai(prompt)
            elif llm_provider.lower() == "gemini":
                content = await self.generate_content_with_gemini(prompt)
            else:
                raise Exception(f"Unsupported LLM provider: {llm_provider}")
            
            # Validate completeness
            required_sections = ["primitives", "mastery_criteria", "questions"]
            present_sections = []
            
            for section in required_sections:
                if section in content and content[section]:
                    present_sections.append(section)
            
            completeness_score = len(present_sections) / len(required_sections)
            
            # Check content counts
            primitives_count = len(content.get("primitives", []))
            criteria_count = len(content.get("mastery_criteria", []))
            questions_count = len(content.get("questions", []))
            
            # Minimum content requirements
            min_primitives = 3
            min_criteria = 3
            min_questions = 3
            
            content_scores = []
            if primitives_count >= min_primitives:
                content_scores.append(1.0)
            else:
                content_scores.append(primitives_count / min_primitives)
            
            if criteria_count >= min_criteria:
                content_scores.append(1.0)
            else:
                content_scores.append(criteria_count / min_criteria)
            
            if questions_count >= min_questions:
                content_scores.append(1.0)
            else:
                content_scores.append(questions_count / min_questions)
            
            overall_completeness = statistics.mean(content_scores)
            
            success = overall_completeness >= 0.8  # 80% completeness threshold
            
            if success:
                print(f"  ✅ {llm_provider.upper()} content completeness: {overall_completeness*100:.1f}%")
            else:
                print(f"  ❌ {llm_provider.upper()} content completeness below threshold: {overall_completeness*100:.1f}%")
            
            return QualityTestResult(
                test_name=f"real_llm_content_completeness_{llm_provider}",
                success=success,
                quality_metrics=ContentQualityMetrics(
                    completeness_score=overall_completeness,
                    coherence_score=0.0,
                    relevance_score=0.0,
                    accuracy_score=0.0,
                    overall_score=overall_completeness
                ),
                details={
                    "llm_provider": llm_provider,
                    "present_sections": present_sections,
                    "primitives_count": primitives_count,
                    "criteria_count": criteria_count,
                    "questions_count": questions_count,
                    "completeness_score": overall_completeness,
                    "raw_content": content
                }
            )
            
        except Exception as e:
            return QualityTestResult(
                test_name=f"real_llm_content_completeness_{llm_provider}",
                success=False,
                error_message=str(e)
            )
    
    async def run_real_llm_tests(self) -> List[QualityTestResult]:
        """Run all real LLM quality tests"""
        print("🚀 Starting Real LLM Content Quality Testing")
        print("=" * 70)
        
        tests = []
        
        # Test OpenAI if available
        if self.openai_client:
            print("🔌 Testing OpenAI GPT-4...")
            tests.extend([
                self.test_real_llm_primitive_generation("openai"),
                self.test_real_llm_content_completeness("openai")
            ])
        else:
            print("⚠️  OpenAI not available (missing OPENAI_API_KEY)")
        
        # Test Gemini if available
        if self.gemini_client:
            print("🔌 Testing Google Gemini...")
            tests.extend([
                self.test_real_llm_primitive_generation("gemini"),
                self.test_real_llm_content_completeness("gemini")
            ])
        else:
            print("⚠️  Gemini not available (missing GOOGLE_API_KEY)")
        
        if not tests:
            print("❌ No LLM providers available for testing")
            return []
        
        # Run all tests
        for test in tests:
            result = await test
            self.test_results.append(result)
            
            if result.success:
                print(f"✅ {result.test_name}: PASSED")
            else:
                print(f"❌ {result.test_name}: FAILED - {result.error_message}")
        
        return self.test_results
    
    def print_real_llm_summary(self):
        """Print comprehensive real LLM test summary"""
        print("\n📊 REAL LLM CONTENT QUALITY TEST SUMMARY")
        print("=" * 70)
        
        if not self.test_results:
            print("No test results available.")
            return
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.success)
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Group results by LLM provider
        openai_results = [r for r in self.test_results if "openai" in r.test_name]
        gemini_results = [r for r in self.test_results if "gemini" in r.test_name]
        
        if openai_results:
            openai_passed = sum(1 for r in openai_results if r.success)
            print(f"\n🤖 OpenAI Results: {openai_passed}/{len(openai_results)} passed")
        
        if gemini_results:
            gemini_passed = sum(1 for r in gemini_results if r.success)
            print(f"🤖 Gemini Results: {gemini_passed}/{len(gemini_results)} passed")
        
        # Overall assessment
        print("\n🎯 OVERALL REAL LLM ASSESSMENT:")
        
        if passed_tests == total_tests:
            print("🎉 All real LLM tests passed! Content generation quality is excellent.")
        elif passed_tests >= total_tests * 0.8:
            print("✅ Most real LLM tests passed. Content generation quality is good.")
        elif passed_tests >= total_tests * 0.6:
            print("⚠️  Some real LLM tests passed. Content generation quality needs improvement.")
        else:
            print("❌ Multiple real LLM tests failed. Content generation quality needs significant attention.")

async def main():
    """Main function to run real LLM content quality tests"""
    print("🚀 Real LLM Content Quality Testing")
    print("=" * 70)
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("❌ No LLM API keys found!")
        print("Please set one or both of:")
        print("  - OPENAI_API_KEY for OpenAI GPT-4")
        print("  - GOOGLE_API_KEY for Google Gemini")
        return 1
    
    tester = RealLLMContentQualityTester()
    
    try:
        # Run all real LLM quality tests
        results = await tester.run_real_llm_tests()
        
        if results:
            # Print comprehensive summary
            tester.print_real_llm_summary()
            
            print("\n🎉 Real LLM content quality testing completed successfully!")
        else:
            print("\n❌ No tests were run due to missing API keys.")
            return 1
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
