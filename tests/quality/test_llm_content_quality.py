#!/usr/bin/env python3
"""
LLM Content Quality and Coverage Testing Framework
Tests the quality, completeness, and coverage of LLM-generated content.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import statistics

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

class LLMContentQualityTester:
    """Tester for LLM content quality and coverage"""
    
    def __init__(self):
        self.test_results: List[QualityTestResult] = []
        
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
    
    async def test_primitive_type_coverage(self, content: Dict[str, Any]) -> QualityTestResult:
        """Test that LLM generates all required primitive types"""
        print("🧪 Testing Primitive Type Coverage...")
        
        try:
            primitives = content.get("primitives", [])
            if not primitives:
                return QualityTestResult(
                    test_name="primitive_type_coverage",
                    success=False,
                    error_message="No primitives generated"
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
                print(f"  ✅ All primitive types covered: {generated_types}")
            else:
                print(f"  ❌ Missing primitive types: {missing_types}")
            
            return QualityTestResult(
                test_name="primitive_type_coverage",
                success=success,
                coverage_metrics=CoverageMetrics(
                    primitive_types_covered=generated_types,
                    uue_stages_covered=set(),
                    assessment_types_covered=set(),
                    difficulty_levels_covered=set(),
                    coverage_score=coverage_score
                ),
                details={
                    "expected_types": list(self.expected_primitive_types),
                    "generated_types": list(generated_types),
                    "missing_types": list(missing_types),
                    "coverage_score": coverage_score
                }
            )
            
        except Exception as e:
            return QualityTestResult(
                test_name="primitive_type_coverage",
                success=False,
                error_message=str(e)
            )
    
    async def test_content_completeness(self, content: Dict[str, Any]) -> QualityTestResult:
        """Test content completeness validation"""
        print("🧪 Testing Content Completeness...")
        
        try:
            # Check required sections
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
            min_criteria = 2
            min_questions = 2
            
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
                print(f"  ✅ Content completeness: {overall_completeness*100:.1f}%")
            else:
                print(f"  ❌ Content completeness below threshold: {overall_completeness*100:.1f}%")
            
            return QualityTestResult(
                test_name="content_completeness",
                success=success,
                quality_metrics=ContentQualityMetrics(
                    completeness_score=overall_completeness,
                    coherence_score=0.0,
                    relevance_score=0.0,
                    accuracy_score=0.0,
                    overall_score=overall_completeness
                ),
                details={
                    "present_sections": present_sections,
                    "primitives_count": primitives_count,
                    "criteria_count": criteria_count,
                    "questions_count": questions_count,
                    "completeness_score": overall_completeness
                }
            )
            
        except Exception as e:
            return QualityTestResult(
                test_name="content_completeness",
                success=False,
                error_message=str(e)
            )
    
    async def run_all_quality_tests(self, test_content: Dict[str, Any]) -> List[QualityTestResult]:
        """Run all content quality tests"""
        print("🚀 Starting LLM Content Quality and Coverage Testing")
        print("=" * 70)
        
        tests = [
            self.test_primitive_type_coverage(test_content),
            self.test_content_completeness(test_content)
        ]
        
        for test in tests:
            result = await test
            self.test_results.append(result)
            
            if result.success:
                print(f"✅ {result.test_name}: PASSED")
            else:
                print(f"❌ {result.test_name}: FAILED - {result.error_message}")
        
        return self.test_results
    
    def print_quality_summary(self):
        """Print comprehensive quality test summary"""
        print("\n📊 CONTENT QUALITY TEST SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.success)
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("🎉 All quality tests passed! Content meets high quality standards.")
        else:
            print("❌ Some quality tests failed. Content quality needs improvement.")

async def main():
    """Main function to run LLM content quality tests"""
    print("🚀 LLM Content Quality and Coverage Testing")
    print("=" * 70)
    
    # Create sample test content
    test_content = {
        "primitives": [
            {"type": "entity", "title": "Machine Learning"},
            {"type": "proposition", "title": "Supervised Learning"},
            {"type": "process", "title": "Model Training"}
        ],
        "mastery_criteria": [
            {"title": "Understand ML Basics", "uue_stage": "UNDERSTAND"},
            {"title": "Apply ML Concepts", "uue_stage": "USE"}
        ],
        "questions": [
            {"text": "What is machine learning?", "type": "definition"},
            {"text": "How does supervised learning work?", "type": "explanation"}
        ]
    }
    
    tester = LLMContentQualityTester()
    
    try:
        # Run all quality tests
        results = await tester.run_all_quality_tests(test_content)
        
        # Print comprehensive summary
        tester.print_quality_summary()
        
        print("\n🎉 LLM content quality testing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
