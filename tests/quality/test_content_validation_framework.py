#!/usr/bin/env python3
"""
Content Validation and Quality Assurance Framework
Comprehensive framework for validating content quality and ensuring consistency.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import statistics

@dataclass
class ValidationRule:
    """A validation rule for content quality"""
    name: str
    description: str
    validator: Callable[[Dict[str, Any]], bool]
    severity: str  # "ERROR", "WARNING", "INFO"
    weight: float  # 0.0 - 1.0

@dataclass
class ValidationResult:
    """Result of a validation check"""
    rule_name: str
    passed: bool
    severity: str
    weight: float
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class QualityScore:
    """Overall quality score for content"""
    total_score: float      # 0.0 - 100.0
    error_count: int
    warning_count: int
    info_count: int
    weighted_score: float   # 0.0 - 1.0

class ContentValidationFramework:
    """Framework for content validation and quality assurance"""
    
    def __init__(self):
        self.validation_rules: List[ValidationRule] = []
        self.validation_results: List[ValidationResult] = []
        self.quality_scores: List[QualityScore] = []
        
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default validation rules"""
        
        # Content completeness rules
        self.add_validation_rule(
            ValidationRule(
                name="has_primitives",
                description="Content must have at least 3 knowledge primitives",
                validator=self._validate_has_primitives,
                severity="ERROR",
                weight=0.2
            )
        )
        
        self.add_validation_rule(
            ValidationRule(
                name="has_mastery_criteria",
                description="Content must have at least 2 mastery criteria",
                validator=self._validate_has_mastery_criteria,
                severity="ERROR",
                weight=0.2
            )
        )
        
        self.add_validation_rule(
            ValidationRule(
                name="has_questions",
                description="Content must have at least 2 questions",
                validator=self._validate_has_questions,
                severity="WARNING",
                weight=0.15
            )
        )
        
        # Content quality rules
        self.add_validation_rule(
            ValidationRule(
                name="primitive_types_coverage",
                description="Content must cover all three primitive types",
                validator=self._validate_primitive_types_coverage,
                severity="WARNING",
                weight=0.15
            )
        )
        
        self.add_validation_rule(
            ValidationRule(
                name="uue_stages_coverage",
                description="Content must cover multiple UUE stages",
                validator=self._validate_uue_stages_coverage,
                severity="WARNING",
                weight=0.15
            )
        )
        
        self.add_validation_rule(
            ValidationRule(
                name="content_coherence",
                description="Content must be coherent and logically structured",
                validator=self._validate_content_coherence,
                severity="INFO",
                weight=0.1
            )
        )
        
        self.add_validation_rule(
            ValidationRule(
                name="difficulty_distribution",
                description="Content must have appropriate difficulty distribution",
                validator=self._validate_difficulty_distribution,
                severity="INFO",
                weight=0.05
            )
        )
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add a custom validation rule"""
        self.validation_rules.append(rule)
    
    def _validate_has_primitives(self, content: Dict[str, Any]) -> bool:
        """Validate that content has sufficient primitives"""
        primitives = content.get("primitives", [])
        return len(primitives) >= 3
    
    def _validate_has_mastery_criteria(self, content: Dict[str, Any]) -> bool:
        """Validate that content has sufficient mastery criteria"""
        criteria = content.get("mastery_criteria", [])
        return len(criteria) >= 2
    
    def _validate_has_questions(self, content: Dict[str, Any]) -> bool:
        """Validate that content has sufficient questions"""
        questions = content.get("questions", [])
        return len(questions) >= 2
    
    def _validate_primitive_types_coverage(self, content: Dict[str, Any]) -> bool:
        """Validate that content covers all primitive types"""
        primitives = content.get("primitives", [])
        if len(primitives) < 3:
            return False
        
        types = set()
        for primitive in primitives:
            if isinstance(primitive, dict) and "type" in primitive:
                types.add(primitive["type"])
        
        expected_types = {"entity", "proposition", "process"}
        return len(types.intersection(expected_types)) >= 2  # At least 2 types
    
    def _validate_uue_stages_coverage(self, content: Dict[str, Any]) -> bool:
        """Validate that content covers multiple UUE stages"""
        criteria = content.get("mastery_criteria", [])
        if len(criteria) < 2:
            return False
        
        stages = set()
        for criterion in criteria:
            if isinstance(criterion, dict) and "uue_stage" in criterion:
                stages.add(criterion["uue_stage"])
        
        return len(stages) >= 2  # At least 2 stages
    
    def _validate_content_coherence(self, content: Dict[str, Any]) -> bool:
        """Validate content coherence and logical structure"""
        # Check if content has logical structure
        has_structure = (
            "primitives" in content and
            "mastery_criteria" in content and
            "questions" in content
        )
        
        if not has_structure:
            return False
        
        # Check if primitives and criteria are related
        primitives = content.get("primitives", [])
        criteria = content.get("mastery_criteria", [])
        
        if not primitives or not criteria:
            return False
        
        # Simple coherence check: ensure we have content in each section
        return len(primitives) > 0 and len(criteria) > 0
    
    def _validate_difficulty_distribution(self, content: Dict[str, Any]) -> bool:
        """Validate appropriate difficulty distribution"""
        criteria = content.get("mastery_criteria", [])
        if len(criteria) < 2:
            return False
        
        difficulties = []
        for criterion in criteria:
            if isinstance(criterion, dict) and "difficulty" in criterion:
                difficulties.append(criterion["difficulty"])
        
        if len(difficulties) < 2:
            return False
        
        # Check if we have a reasonable spread of difficulties
        unique_difficulties = set(difficulties)
        return len(unique_difficulties) >= 2
    
    async def validate_content(self, content: Dict[str, Any]) -> List[ValidationResult]:
        """Run all validation rules on content"""
        print("🔍 Running Content Validation...")
        
        results = []
        
        for rule in self.validation_rules:
            try:
                passed = rule.validator(content)
                
                result = ValidationResult(
                    rule_name=rule.name,
                    passed=passed,
                    severity=rule.severity,
                    weight=rule.weight,
                    details={"description": rule.description}
                )
                
                results.append(result)
                
                if passed:
                    print(f"  ✅ {rule.name}: PASSED")
                else:
                    print(f"  ❌ {rule.name}: FAILED")
                
            except Exception as e:
                result = ValidationResult(
                    rule_name=rule.name,
                    passed=False,
                    severity=rule.severity,
                    weight=rule.weight,
                    error_message=str(e),
                    details={"description": rule.description}
                )
                
                results.append(result)
                print(f"  ❌ {rule.name}: ERROR - {e}")
        
        self.validation_results.extend(results)
        return results
    
    def calculate_quality_score(self, validation_results: List[ValidationResult]) -> QualityScore:
        """Calculate overall quality score from validation results"""
        if not validation_results:
            return QualityScore(0.0, 0, 0, 0, 0.0)
        
        # Count by severity
        error_count = sum(1 for r in validation_results if r.severity == "ERROR" and not r.passed)
        warning_count = sum(1 for r in validation_results if r.severity == "WARNING" and not r.passed)
        info_count = sum(1 for r in validation_results if r.severity == "INFO" and not r.passed)
        
        # Calculate weighted score
        total_weight = 0.0
        passed_weight = 0.0
        
        for result in validation_results:
            total_weight += result.weight
            if result.passed:
                passed_weight += result.weight
        
        weighted_score = passed_weight / total_weight if total_weight > 0 else 0.0
        
        # Convert to 0-100 scale
        total_score = weighted_score * 100.0
        
        return QualityScore(
            total_score=total_score,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            weighted_score=weighted_score
        )
    
    async def run_ab_testing(self, content_a: Dict[str, Any], content_b: Dict[str, Any]) -> Dict[str, Any]:
        """Run A/B testing to compare content quality"""
        print("🧪 Running A/B Content Quality Testing...")
        
        # Validate both content versions
        results_a = await self.validate_content(content_a)
        results_b = await self.validate_content(content_b)
        
        # Calculate quality scores
        score_a = self.calculate_quality_score(results_a)
        score_b = self.calculate_quality_score(results_b)
        
        # Determine winner
        winner = "A" if score_a.total_score > score_b.total_score else "B"
        if score_a.total_score == score_b.total_score:
            winner = "TIE"
        
        comparison = {
            "content_a_score": score_a.total_score,
            "content_b_score": score_b.total_score,
            "winner": winner,
            "difference": abs(score_a.total_score - score_b.total_score),
            "improvement_percentage": ((score_b.total_score - score_a.total_score) / score_a.total_score * 100) if score_a.total_score > 0 else 0
        }
        
        print(f"  📊 Content A Score: {score_a.total_score:.1f}/100")
        print(f"  📊 Content B Score: {score_b.total_score:.1f}/100")
        print(f"  🏆 Winner: {winner}")
        
        return comparison
    
    async def run_regression_testing(self, content_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run regression testing for content generation consistency"""
        print("🔄 Running Content Generation Regression Testing...")
        
        if len(content_samples) < 2:
            return {"error": "Need at least 2 content samples for regression testing"}
        
        all_scores = []
        consistency_scores = []
        
        # Validate each sample
        for i, content in enumerate(content_samples):
            results = await self.validate_content(content)
            score = self.calculate_quality_score(results)
            all_scores.append(score.total_score)
            
            print(f"  📊 Sample {i+1} Score: {score.total_score:.1f}/100")
        
        # Calculate consistency metrics
        if len(all_scores) > 1:
            mean_score = statistics.mean(all_scores)
            std_score = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
            
            # Consistency score: lower standard deviation = higher consistency
            max_std = 20.0  # Maximum acceptable standard deviation
            consistency_score = max(0.0, 1.0 - (std_score / max_std))
            
            consistency_scores.append(consistency_score)
            
            print(f"  📊 Mean Score: {mean_score:.1f}/100")
            print(f"  📊 Standard Deviation: {std_score:.1f}")
            print(f"  📊 Consistency Score: {consistency_score*100:.1f}%")
        
        return {
            "sample_scores": all_scores,
            "mean_score": statistics.mean(all_scores) if all_scores else 0.0,
            "std_score": statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0,
            "consistency_score": statistics.mean(consistency_scores) if consistency_scores else 0.0
        }
    
    def generate_quality_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for quality metrics dashboard"""
        if not self.validation_results:
            return {"error": "No validation results available"}
        
        # Aggregate results by severity
        error_results = [r for r in self.validation_results if r.severity == "ERROR"]
        warning_results = [r for r in self.validation_results if r.severity == "WARNING"]
        info_results = [r for r in self.validation_results if r.severity == "INFO"]
        
        # Calculate pass rates
        error_pass_rate = sum(1 for r in error_results if r.passed) / len(error_results) if error_results else 1.0
        warning_pass_rate = sum(1 for r in warning_results if r.passed) / len(warning_results) if warning_results else 1.0
        info_pass_rate = sum(1 for r in info_results if r.passed) / len(info_results) if info_results else 1.0
        
        # Overall pass rate
        overall_pass_rate = sum(1 for r in self.validation_results if r.passed) / len(self.validation_results)
        
        # Rule performance
        rule_performance = {}
        for rule in self.validation_rules:
            rule_results = [r for r in self.validation_results if r.rule_name == rule.name]
            if rule_results:
                pass_rate = sum(1 for r in rule_results if r.passed) / len(rule_results)
                rule_performance[rule.name] = {
                    "pass_rate": pass_rate,
                    "total_checks": len(rule_results),
                    "severity": rule.severity,
                    "weight": rule.weight
                }
        
        return {
            "overall_pass_rate": overall_pass_rate,
            "severity_breakdown": {
                "error": {
                    "pass_rate": error_pass_rate,
                    "total": len(error_results)
                },
                "warning": {
                    "pass_rate": warning_pass_rate,
                    "total": len(warning_results)
                },
                "info": {
                    "pass_rate": info_pass_rate,
                    "total": len(info_results)
                }
            },
            "rule_performance": rule_performance,
            "total_validations": len(self.validation_results),
            "quality_scores": [score.total_score for score in self.quality_scores]
        }
    
    def flag_low_quality_content(self, quality_threshold: float = 70.0) -> List[Dict[str, Any]]:
        """Flag content that falls below quality threshold"""
        flagged_content = []
        
        for i, score in enumerate(self.quality_scores):
            if score.total_score < quality_threshold:
                flagged_content.append({
                    "content_index": i,
                    "score": score.total_score,
                    "threshold": quality_threshold,
                    "error_count": score.error_count,
                    "warning_count": score.warning_count,
                    "severity": "HIGH" if score.error_count > 0 else "MEDIUM" if score.warning_count > 0 else "LOW"
                })
        
        return flagged_content
    
    def print_validation_summary(self):
        """Print comprehensive validation summary"""
        print("\n📊 CONTENT VALIDATION SUMMARY")
        print("=" * 70)
        
        if not self.validation_results:
            print("No validation results available.")
            return
        
        total_validations = len(self.validation_results)
        passed_validations = sum(1 for r in self.validation_results if r.passed)
        failed_validations = total_validations - passed_validations
        
        print(f"Total Validations: {total_validations}")
        print(f"Passed: {passed_validations}")
        print(f"Failed: {failed_validations}")
        print(f"Pass Rate: {(passed_validations/total_validations)*100:.1f}%")
        
        # Breakdown by severity
        print("\n🔍 SEVERITY BREAKDOWN:")
        for severity in ["ERROR", "WARNING", "INFO"]:
            severity_results = [r for r in self.validation_results if r.severity == severity]
            if severity_results:
                severity_passed = sum(1 for r in severity_results if r.passed)
                severity_total = len(severity_results)
                severity_pass_rate = (severity_passed / severity_total) * 100
                
                print(f"  {severity}: {severity_passed}/{severity_total} ({severity_pass_rate:.1f}%)")
        
        # Quality scores
        if self.quality_scores:
            print("\n📊 QUALITY SCORES:")
            for i, score in enumerate(self.quality_scores):
                print(f"  Content {i+1}: {score.total_score:.1f}/100")
            
            avg_score = statistics.mean([s.total_score for s in self.quality_scores])
            print(f"  Average Score: {avg_score:.1f}/100")
        
        # Flagged content
        flagged = self.flag_low_quality_content()
        if flagged:
            print(f"\n⚠️  FLAGGED CONTENT ({len(flagged)} items):")
            for item in flagged:
                print(f"  Content {item['content_index']+1}: {item['score']:.1f}/100 ({item['severity']})")

async def main():
    """Main function to run content validation framework"""
    print("🚀 Content Validation and Quality Assurance Framework")
    print("=" * 70)
    
    # Create sample content for testing
    sample_content = {
        "primitives": [
            {"type": "entity", "title": "Machine Learning"},
            {"type": "proposition", "title": "Supervised Learning"},
            {"type": "process", "title": "Model Training"}
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
    
    # Create alternative content for A/B testing
    alternative_content = {
        "primitives": [
            {"type": "entity", "title": "Artificial Intelligence"},
            {"type": "proposition", "title": "Deep Learning"},
            {"type": "process", "title": "Neural Network Training"}
        ],
        "mastery_criteria": [
            {"title": "AI Fundamentals", "uue_stage": "UNDERSTAND", "difficulty": "BEGINNER"},
            {"title": "Deep Learning Applications", "uue_stage": "EXPLORE", "difficulty": "ADVANCED"}
        ],
        "questions": [
            {"text": "What is artificial intelligence?", "type": "definition"},
            {"text": "How do neural networks learn?", "type": "explanation"}
        ]
    }
    
    framework = ContentValidationFramework()
    
    try:
        # Run content validation
        print("\n🔍 Validating Sample Content...")
        validation_results = await framework.validate_content(sample_content)
        
        # Calculate quality score
        quality_score = framework.calculate_quality_score(validation_results)
        framework.quality_scores.append(quality_score)
        
        print(f"\n📊 Quality Score: {quality_score.total_score:.1f}/100")
        
        # Run A/B testing
        print("\n🧪 Running A/B Testing...")
        ab_results = await framework.run_ab_testing(sample_content, alternative_content)
        
        # Run regression testing
        print("\n🔄 Running Regression Testing...")
        regression_results = await framework.run_regression_testing([sample_content, alternative_content])
        
        # Generate dashboard data
        dashboard_data = framework.generate_quality_dashboard_data()
        
        # Print comprehensive summary
        framework.print_validation_summary()
        
        print("\n🎉 Content validation framework testing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
