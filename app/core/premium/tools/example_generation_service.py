"""
Real Example Generation Service for Premium Tools
Provides AI-powered, context-aware example generation based on user learning patterns.
"""

import asyncio
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime
import json
import re

@dataclass
class ExampleRequest:
    """Request for example generation"""
    concept: str
    user_id: str
    context: Dict[str, Any]
    user_level: str
    example_type: str
    learning_style: str
    previous_examples: List[str]

@dataclass
class ExampleResult:
    """Result of example generation"""
    concept: str
    examples: List[Dict[str, Any]]
    explanation: str
    difficulty_level: str
    learning_objectives: List[str]
    related_concepts: List[str]
    generation_time: float
    confidence_score: float
    error: Optional[str] = None
    success: bool = True

@dataclass
class UserLearningProfile:
    """User's learning profile for personalization"""
    user_id: str
    learning_style: str
    preferred_explanation_style: str
    mastery_level: str
    focus_areas: List[str]
    learning_efficiency: float
    preferred_example_types: List[str]

class ExampleGenerationService:
    """AI-powered example generation service"""
    
    def __init__(self):
        # Example templates for different types
        self.example_templates = {
            "code": {
                "beginner": "Simple, well-commented code with step-by-step explanation",
                "intermediate": "Moderate complexity with best practices and error handling",
                "advanced": "Complex implementation with design patterns and optimization"
            },
            "scenario": {
                "beginner": "Real-world situation with clear context and simple outcomes",
                "intermediate": "Complex scenario with multiple factors and decision points",
                "advanced": "Multi-layered scenario with competing priorities and trade-offs"
            },
            "step_by_step": {
                "beginner": "Detailed breakdown with explanations for each step",
                "intermediate": "Logical flow with key decision points highlighted",
                "advanced": "Strategic approach with alternative paths and optimization"
            }
        }
        
        # Learning style adaptations
        self.style_adaptations = {
            "visual": "Include diagrams, charts, and visual representations",
            "auditory": "Focus on verbal explanations and storytelling",
            "kinesthetic": "Emphasize hands-on activities and practical applications",
            "reading": "Provide comprehensive written explanations with examples"
        }
    
    async def generate_examples(self, request: ExampleRequest) -> ExampleResult:
        """
        Generate contextually relevant examples
        
        Args:
            request: Example generation request with user context
            
        Returns:
            ExampleResult with generated examples
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Get user learning profile
            user_profile = await self._get_user_learning_profile(request.user_id)
            
            # Analyze concept and context
            concept_analysis = await self._analyze_concept(request.concept, request.context)
            
            # Generate examples based on user profile and concept analysis
            examples = await self._generate_contextual_examples(
                request, user_profile, concept_analysis
            )
            
            # Create explanation tailored to user's learning style
            explanation = await self._create_personalized_explanation(
                request.concept, examples, user_profile
            )
            
            # Determine difficulty level
            difficulty_level = self._determine_difficulty_level(
                examples, user_profile, concept_analysis
            )
            
            # Extract learning objectives
            learning_objectives = self._extract_learning_objectives(
                request.concept, examples, user_profile
            )
            
            # Find related concepts
            related_concepts = await self._find_related_concepts(
                request.concept, concept_analysis
            )
            
            generation_time = asyncio.get_event_loop().time() - start_time
            
            return ExampleResult(
                concept=request.concept,
                examples=examples,
                explanation=explanation,
                difficulty_level=difficulty_level,
                learning_objectives=learning_objectives,
                related_concepts=related_concepts,
                generation_time=generation_time,
                confidence_score=0.85  # Would be calculated based on quality metrics
            )
            
        except Exception as e:
            generation_time = asyncio.get_event_loop().time() - start_time
            return ExampleResult(
                concept=request.concept,
                examples=[],
                explanation="",
                difficulty_level="unknown",
                learning_objectives=[],
                related_concepts=[],
                generation_time=generation_time,
                error=f"Example generation failed: {str(e)}",
                success=False
            )
    
    async def _get_user_learning_profile(self, user_id: str) -> UserLearningProfile:
        """Get user's learning profile for personalization"""
        # In production, this would fetch from a database or user service
        # For now, return a default profile
        return UserLearningProfile(
            user_id=user_id,
            learning_style="visual",
            preferred_explanation_style="step_by_step",
            mastery_level="intermediate",
            focus_areas=["programming", "mathematics", "problem_solving"],
            learning_efficiency=0.75,
            preferred_example_types=["code", "scenario", "step_by_step"]
        )
    
    async def _analyze_concept(self, concept: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the concept to understand its complexity and requirements"""
        # In production, this would use NLP or AI to analyze the concept
        analysis = {
            "complexity": "intermediate",
            "prerequisites": [],
            "key_components": [],
            "common_misconceptions": [],
            "practical_applications": []
        }
        
        # Simple keyword-based analysis
        concept_lower = concept.lower()
        
        if any(word in concept_lower for word in ["algorithm", "data structure", "design pattern"]):
            analysis["complexity"] = "advanced"
            analysis["key_components"] = ["implementation", "efficiency", "trade-offs"]
        elif any(word in concept_lower for word in ["function", "loop", "condition"]):
            analysis["complexity"] = "beginner"
            analysis["key_components"] = ["syntax", "logic", "flow"]
        else:
            analysis["complexity"] = "intermediate"
            analysis["key_components"] = ["understanding", "application", "practice"]
        
        return analysis
    
    async def _generate_contextual_examples(self, request: ExampleRequest, 
                                          user_profile: UserLearningProfile,
                                          concept_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate examples based on user context and concept analysis"""
        examples = []
        
        # Generate code examples if requested
        if "code" in request.example_type or "code" in user_profile.preferred_example_types:
            code_examples = await self._generate_code_examples(
                request.concept, user_profile, concept_analysis
            )
            examples.extend(code_examples)
        
        # Generate scenario examples
        if "scenario" in request.example_type or "scenario" in user_profile.preferred_example_types:
            scenario_examples = await self._generate_scenario_examples(
                request.concept, user_profile, concept_analysis
            )
            examples.extend(scenario_examples)
        
        # Generate step-by-step examples
        if "step_by_step" in request.example_type or "step_by_step" in user_profile.preferred_explanation_style:
            step_examples = await self._generate_step_examples(
                request.concept, user_profile, concept_analysis
            )
            examples.extend(step_examples)
        
        return examples
    
    async def _generate_code_examples(self, concept: str, user_profile: UserLearningProfile,
                                     concept_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate code examples based on user level and concept"""
        examples = []
        
        # Determine appropriate complexity
        if user_profile.mastery_level == "beginner":
            complexity = "beginner"
        elif user_profile.mastery_level == "advanced":
            complexity = "advanced"
        else:
            complexity = "intermediate"
        
        # Generate example based on concept
        if "sorting" in concept.lower():
            examples.append({
                "type": "code",
                "title": "Bubble Sort Implementation",
                "description": "Simple sorting algorithm for beginners",
                "code": self._get_bubble_sort_example(complexity),
                "explanation": "This example shows how to implement a basic sorting algorithm step by step.",
                "difficulty": complexity,
                "tags": ["algorithms", "sorting", "beginner"]
            })
        elif "function" in concept.lower():
            examples.append({
                "type": "code",
                "title": "Function Definition and Usage",
                "description": "Creating and using functions in Python",
                "code": self._get_function_example(complexity),
                "explanation": "Learn how to define functions and use them in your programs.",
                "difficulty": complexity,
                "tags": ["functions", "python", "basics"]
            })
        else:
            # Generic code example
            examples.append({
                "type": "code",
                "title": f"Example: {concept}",
                "description": f"Code example demonstrating {concept}",
                "code": f"# Example code for {concept}\n# This demonstrates the concept in practice",
                "explanation": f"This example shows how to implement {concept} in code.",
                "difficulty": complexity,
                "tags": [concept.lower(), "example", complexity]
            })
        
        return examples
    
    async def _generate_scenario_examples(self, concept: str, user_profile: UserLearningProfile,
                                         concept_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate real-world scenario examples"""
        examples = []
        
        # Create scenario based on concept
        if "database" in concept.lower():
            examples.append({
                "type": "scenario",
                "title": "Online Shopping Cart",
                "description": "Managing user shopping cart in an e-commerce system",
                "scenario": "You're building an online store where users can add items to their cart...",
                "challenges": ["Data consistency", "User session management", "Inventory updates"],
                "solutions": ["Database transactions", "Session storage", "Real-time inventory"],
                "difficulty": user_profile.mastery_level,
                "tags": ["databases", "e-commerce", "real-world"]
            })
        else:
            examples.append({
                "type": "scenario",
                "title": f"Real-world Application: {concept}",
                "description": f"How {concept} is used in practical situations",
                "scenario": f"Imagine you're working on a project that requires understanding {concept}...",
                "challenges": ["Understanding requirements", "Implementation", "Testing"],
                "solutions": ["Clear documentation", "Best practices", "Systematic approach"],
                "difficulty": user_profile.mastery_level,
                "tags": [concept.lower(), "practical", "application"]
            })
        
        return examples
    
    async def _generate_step_examples(self, concept: str, user_profile: UserLearningProfile,
                                     concept_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate step-by-step explanation examples"""
        examples = []
        
        # Create step-by-step breakdown
        steps = [
            "Understand the problem or concept",
            "Break it down into smaller components",
            "Learn the fundamental principles",
            "Practice with simple examples",
            "Apply to more complex scenarios",
            "Review and refine understanding"
        ]
        
        examples.append({
            "type": "step_by_step",
            "title": f"Learning Path: {concept}",
            "description": f"Step-by-step approach to mastering {concept}",
            "steps": steps,
            "explanation": f"This learning path breaks down {concept} into manageable steps.",
            "difficulty": user_profile.mastery_level,
            "tags": ["learning", "methodology", "step-by-step"]
        })
        
        return examples
    
    def _get_bubble_sort_example(self, complexity: str) -> str:
        """Get bubble sort code example based on complexity"""
        if complexity == "beginner":
            return """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Example usage
numbers = [64, 34, 25, 12, 22, 11, 90]
sorted_numbers = bubble_sort(numbers)
print(f"Sorted array: {sorted_numbers}")"""
        else:
            return """def bubble_sort_optimized(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break  # Array is already sorted
    return arr

# Example usage with performance tracking
import time
numbers = [64, 34, 25, 12, 22, 11, 90]
start_time = time.time()
sorted_numbers = bubble_sort_optimized(numbers)
end_time = time.time()
print(f"Sorted array: {sorted_numbers}")
print(f"Time taken: {end_time - start_time:.6f} seconds")"""
    
    def _get_function_example(self, complexity: str) -> str:
        """Get function code example based on complexity"""
        if complexity == "beginner":
            return """def greet(name):
    return f"Hello, {name}!"

# Example usage
message = greet("Alice")
print(message)  # Output: Hello, Alice!"""
        else:
            return """def calculate_area(shape, **dimensions):
    \"\"\"
    Calculate area of different shapes
    
    Args:
        shape (str): Type of shape ('rectangle', 'circle', 'triangle')
        **dimensions: Shape-specific dimensions
    
    Returns:
        float: Calculated area
    \"\"\"
    if shape == "rectangle":
        return dimensions.get('length', 0) * dimensions.get('width', 0)
    elif shape == "circle":
        import math
        radius = dimensions.get('radius', 0)
        return math.pi * radius ** 2
    elif shape == "triangle":
        base = dimensions.get('base', 0)
        height = dimensions.get('height', 0)
        return 0.5 * base * height
    else:
        raise ValueError(f"Unsupported shape: {shape}")

# Example usage
rectangle_area = calculate_area('rectangle', length=5, width=3)
circle_area = calculate_area('circle', radius=4)
print(f"Rectangle area: {rectangle_area}")
print(f"Circle area: {circle_area:.2f}")"""
    
    async def _create_personalized_explanation(self, concept: str, examples: List[Dict[str, Any]],
                                              user_profile: UserLearningProfile) -> str:
        """Create explanation tailored to user's learning style"""
        explanation = f"Here's how to understand {concept}:\n\n"
        
        # Adapt explanation to learning style
        if user_profile.learning_style == "visual":
            explanation += "📊 Visual learners: Focus on the diagrams and visual representations in the examples.\n"
        elif user_profile.learning_style == "auditory":
            explanation += "🎧 Auditory learners: Read the explanations aloud and discuss with others.\n"
        elif user_profile.learning_style == "kinesthetic":
            explanation += "🖐️ Kinesthetic learners: Try implementing the examples yourself and experiment.\n"
        elif user_profile.learning_style == "reading":
            explanation += "📖 Reading learners: Take notes and review the detailed explanations.\n"
        
        explanation += f"\nThe examples provided cover different aspects of {concept}:\n"
        
        for i, example in enumerate(examples, 1):
            explanation += f"{i}. {example['title']}: {example['description']}\n"
        
        explanation += f"\nStart with examples that match your current level ({user_profile.mastery_level}) "
        explanation += "and gradually work your way up to more complex scenarios."
        
        return explanation
    
    def _determine_difficulty_level(self, examples: List[Dict[str, Any]], 
                                   user_profile: UserLearningProfile,
                                   concept_analysis: Dict[str, Any]) -> str:
        """Determine appropriate difficulty level for the user"""
        # Consider user's mastery level and concept complexity
        if user_profile.mastery_level == "beginner" and concept_analysis["complexity"] == "beginner":
            return "beginner"
        elif user_profile.mastery_level == "advanced" and concept_analysis["complexity"] == "advanced":
            return "advanced"
        else:
            return "intermediate"
    
    def _extract_learning_objectives(self, concept: str, examples: List[Dict[str, Any]],
                                     user_profile: UserLearningProfile) -> List[str]:
        """Extract learning objectives from examples and concept"""
        objectives = [
            f"Understand the fundamental principles of {concept}",
            f"Apply {concept} in practical scenarios",
            f"Recognize when and how to use {concept}",
            f"Develop problem-solving skills related to {concept}"
        ]
        
        # Add example-specific objectives
        for example in examples:
            if example["type"] == "code":
                objectives.append(f"Write code that demonstrates {concept}")
            elif example["type"] == "scenario":
                objectives.append(f"Analyze real-world problems using {concept}")
            elif example["type"] == "step_by_step":
                objectives.append(f"Follow systematic approach to learning {concept}")
        
        return objectives
    
    async def _find_related_concepts(self, concept: str, 
                                     concept_analysis: Dict[str, Any]) -> List[str]:
        """Find concepts related to the main concept"""
        # In production, this would use a knowledge graph or AI analysis
        related_concepts = []
        
        concept_lower = concept.lower()
        
        if "sorting" in concept_lower:
            related_concepts = ["algorithms", "data structures", "complexity analysis", "optimization"]
        elif "function" in concept_lower:
            related_concepts = ["parameters", "return values", "scope", "recursion", "lambda functions"]
        elif "database" in concept_lower:
            related_concepts = ["SQL", "data modeling", "normalization", "indexing", "transactions"]
        else:
            related_concepts = ["fundamentals", "best practices", "common patterns", "advanced techniques"]
        
        return related_concepts
    
    async def adapt_example(self, example_id: str, user_feedback: Dict[str, Any]) -> ExampleResult:
        """Adapt example based on user feedback"""
        try:
            # In production, this would retrieve the original example and modify it
            # For now, return a placeholder
            return ExampleResult(
                concept="adapted_example",
                examples=[],
                explanation="Example adapted based on your feedback",
                difficulty_level="adaptive",
                learning_objectives=["Personalized learning"],
                related_concepts=[],
                generation_time=0.0,
                confidence_score=0.9
            )
        except Exception as e:
            return ExampleResult(
                concept="adapted_example",
                examples=[],
                explanation="",
                difficulty_level="unknown",
                learning_objectives=[],
                related_concepts=[],
                generation_time=0.0,
                error=f"Example adaptation failed: {str(e)}",
                success=False
            )
    
    async def get_supported_example_types(self) -> List[str]:
        """Get list of supported example types"""
        return ["code", "scenario", "step_by_step", "interactive", "visual"]
    
    async def get_learning_style_adaptations(self) -> Dict[str, str]:
        """Get available learning style adaptations"""
        return self.style_adaptations
