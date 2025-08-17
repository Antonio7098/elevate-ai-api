"""
LangChain-based Blueprint Editing Agent for the premium multi-agent system.
Provides intelligent editing capabilities for blueprints, primitives, mastery criteria, and questions.
"""

from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import Dict, Any, List, Optional, Union
from ..langgraph_setup import PremiumAgentState
from ..gemini_service import GeminiService
from ..core_api_client import CoreAPIClient
import json
import asyncio

# Pydantic models for structured LLM output
class EditPlan(BaseModel):
    """Structured edit plan from LLM"""
    summary: str = Field(description="Brief overview of planned changes")
    context_alignment: str = Field(description="How changes align with context")
    changes: List[Dict[str, str]] = Field(description="List of specific changes to make")
    priority: str = Field(description="Priority level: low, medium, high, critical")
    estimated_impact: str = Field(description="Expected impact on learning outcomes")

class ContentAnalysis(BaseModel):
    """Structured content analysis from LLM"""
    strengths: List[str] = Field(description="Identified strengths")
    weaknesses: List[str] = Field(description="Areas for improvement")
    complexity_score: float = Field(description="Complexity rating 0.0-1.0")
    clarity_score: float = Field(description="Clarity rating 0.0-1.0")
    learning_objective_alignment: float = Field(description="Alignment with learning objectives 0.0-1.0")
    recommendations: List[str] = Field(description="Specific improvement recommendations")

class BlueprintEditingAgent:
    """LangChain-enhanced blueprint editing agent with structured output"""
    
    def __init__(self):
        self.llm = GeminiService()
        self.core_api_client = CoreAPIClient()
        self.edit_plan_parser = PydanticOutputParser(pydantic_object=EditPlan)
        self.analysis_parser = PydanticOutputParser(pydantic_object=ContentAnalysis)
        
        # Register all tools
        self.tools = [
            self.analyze_blueprint_content,
            self.analyze_primitive_content,
            self.create_edit_plan,
            self.apply_edits,
            self.get_editing_suggestions,
            self.execute_granular_edit
        ]
    
    async def analyze_blueprint_content(self, blueprint_id: str, user_id: str) -> str:
        """Analyze blueprint content for strengths, weaknesses, and improvement opportunities"""
        try:
            # Use mock blueprint data instead of Core API
            mock_blueprint = {
                "id": blueprint_id,
                "title": "Introduction to Machine Learning",
                "description": "A comprehensive guide to ML fundamentals",
                "sections": [
                    {"title": "What is ML?", "content": "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed."},
                    {"title": "Types of ML", "content": "Supervised, unsupervised, and reinforcement learning are the main categories."}
                ],
                "difficulty": "beginner",
                "estimated_time": "4 hours"
            }
            
            prompt = f"""
            Analyze this learning blueprint for strengths, weaknesses, and improvement opportunities.
            
            Blueprint: {json.dumps(mock_blueprint, indent=2)}
            
            Provide a structured analysis focusing on:
            1. Content quality and clarity
            2. Learning objective alignment
            3. Structural organization
            4. Difficulty progression
            5. Engagement factors
            
            Return the analysis in this exact JSON format:
            {{
                "strengths": ["list", "of", "strengths"],
                "weaknesses": ["list", "of", "weaknesses"],
                "complexity_score": 0.7,
                "clarity_score": 0.8,
                "learning_objective_alignment": 0.9,
                "recommendations": ["list", "of", "recommendations"]
            }}
            """
            
            response_text = await self.llm.generate(prompt, model="gemini_2_5_flash")
            
            # Parse with structured output parser
            try:
                analysis = self.analysis_parser.parse(response_text)
                return f"Blueprint analysis completed successfully. Clarity score: {analysis.clarity_score}, Complexity: {analysis.complexity_score}"
            except Exception as parse_error:
                # Fallback to manual parsing if structured output fails
                return f"Analysis completed but parsing failed: {parse_error}. Raw response: {response_text[:200]}..."
                
        except Exception as e:
            return f"Failed to analyze blueprint: {str(e)}"
    
    async def analyze_primitive_content(self, primitive_id: str, user_id: str) -> str:
        """Analyze knowledge primitive content for improvement opportunities"""
        try:
            # Use mock primitive data instead of Core API
            mock_primitive = {
                "id": primitive_id,
                "title": "Machine Learning Fundamentals",
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                "concepts": ["supervised learning", "unsupervised learning", "reinforcement learning"],
                "difficulty": "intermediate",
                "prerequisites": ["basic programming", "statistics"],
                "examples": ["spam detection", "recommendation systems", "image recognition"]
            }
            
            prompt = f"""
            Analyze this knowledge primitive for strengths, weaknesses, and improvement opportunities.
            
            Primitive: {json.dumps(mock_primitive, indent=2)}
            
            Focus on:
            1. Concept clarity and explanation quality
            2. Learning objective alignment
            3. Difficulty appropriateness
            4. Prerequisites and dependencies
            5. Engagement and retention factors
            
            Return the analysis in this exact JSON format:
            {{
                "strengths": ["list", "of", "strengths"],
                "weaknesses": ["list", "of", "weaknesses"],
                "complexity_score": 0.6,
                "clarity_score": 0.7,
                "learning_objective_alignment": 0.8,
                "recommendations": ["list", "of", "recommendations"]
            }}
            """
            
            response_text = await self.llm.generate(prompt, model="gemini_2_5_flash")
            
            try:
                analysis = self.analysis_parser.parse(response_text)
                return f"Primitive analysis completed. Clarity: {analysis.clarity_score}, Complexity: {analysis.complexity_score}"
            except Exception as parse_error:
                return f"Analysis completed but parsing failed: {parse_error}. Raw response: {response_text[:200]}..."
                
        except Exception as e:
            return f"Failed to analyze primitive: {str(e)}"
    
    async def create_edit_plan(self, content_analysis: str, edit_instruction: str, content_type: str) -> str:
        """Create a structured edit plan based on content analysis and edit instruction"""
        try:
            prompt = f"""
            Create a detailed edit plan based on this content analysis and edit instruction.
            
            Content Analysis: {content_analysis}
            Edit Instruction: {edit_instruction}
            Content Type: {content_type}
            
            Create a comprehensive edit plan that includes:
            1. Specific changes to make
            2. Priority levels for each change
            3. Expected impact on learning outcomes
            4. Context alignment considerations
            
            Return the plan in this exact JSON format:
            {{
                "summary": "Brief overview of planned changes",
                "context_alignment": "How changes align with context",
                "changes": [
                    {{
                        "type": "content|clarity|structure|complexity",
                        "description": "What to change",
                        "reason": "Why this change is needed",
                        "priority": "low|medium|high|critical"
                    }}
                ],
                "priority": "low|medium|high|critical",
                "estimated_impact": "Expected impact on learning outcomes"
            }}
            """
            
            response_text = await self.llm.generate(prompt, model="gemini_2_5_flash")
            
            try:
                edit_plan = self.edit_plan_parser.parse(response_text)
                return f"Edit plan created successfully. Priority: {edit_plan.priority}, Changes: {len(edit_plan.changes)}"
            except Exception as parse_error:
                return f"Edit plan created but parsing failed: {parse_error}. Raw response: {response_text[:200]}..."
                
        except Exception as e:
            return f"Failed to create edit plan: {str(e)}"
    
    async def apply_edits(self, content: str, edit_plan: str, content_type: str) -> str:
        """Apply edits to content based on the edit plan"""
        try:
            prompt = f"""
            Apply the following edits to this {content_type} content based on the edit plan.
            
            Original Content: {content}
            Edit Plan: {edit_plan}
            
            Apply all specified changes while maintaining:
            1. Content structure and flow
            2. Learning objectives
            3. Difficulty level appropriateness
            4. Contextual relevance
            
            Return the edited content in the same format as the original, with all changes applied.
            """
            
            response_text = await self.llm.generate(prompt, model="gemini_2_5_flash")
            
            return f"Edits applied successfully. New content length: {len(response_text)} characters"
            
        except Exception as e:
            return f"Failed to apply edits: {str(e)}"
    
    async def get_editing_suggestions(self, content: str, content_type: str, user_id: str) -> str:
        """Generate intelligent editing suggestions for content improvement"""
        try:
            prompt = f"""
            Analyze this {content_type} content and provide intelligent editing suggestions.
            
            Content: {content}
            
            Generate 3-5 specific, actionable suggestions for improvement in these areas:
            1. Content clarity and comprehension
            2. Learning objective alignment
            3. Engagement and retention
            4. Difficulty progression
            5. Structural organization
            
            Return suggestions in this format:
            {{
                "suggestions": [
                    {{
                        "type": "clarity|engagement|structure|difficulty",
                        "description": "Specific suggestion",
                        "reason": "Why this improvement is needed",
                        "impact": "Expected learning impact",
                        "priority": "low|medium|high"
                    }}
                ]
            }}
            """
            
            response_text = await self.llm.generate(prompt, model="gemini_2_5_flash")
            
            return f"Generated {len(response_text.split('suggestion'))} editing suggestions successfully"
            
        except Exception as e:
            return f"Failed to generate suggestions: {str(e)}"
    
    async def execute_granular_edit(self, edit_type: str, content: str, parameters: Dict[str, Any]) -> str:
        """Execute a specific granular edit operation (add, remove, reorder, etc.)"""
        try:
            edit_operations = {
                "add_section": "Add a new section to the content",
                "remove_section": "Remove a specified section",
                "reorder_sections": "Reorder sections for better flow",
                "add_primitive": "Add a new knowledge primitive",
                "update_difficulty": "Update difficulty level",
                "add_example": "Add clarifying examples",
                "restructure": "Restructure content organization"
            }
            
            operation = edit_operations.get(edit_type, "unknown operation")
            
            prompt = f"""
            Execute this granular edit operation: {operation}
            
            Content: {content}
            Parameters: {json.dumps(parameters, indent=2)}
            
            Perform the {edit_type} operation while maintaining:
            1. Content integrity and coherence
            2. Learning objective alignment
            3. Structural consistency
            4. Contextual relevance
            
            Return the modified content with the requested changes applied.
            """
            
            response_text = await self.llm.generate(prompt, model="gemini_2_5_flash")
            
            return f"Granular edit '{edit_type}' executed successfully. Modified content length: {len(response_text)} characters"
            
        except Exception as e:
            return f"Failed to execute granular edit: {str(e)}"
    
    async def edit_content_agentically(
        self,
        content_id: str,
        content_type: str,
        edit_instruction: str,
        user_id: str,
        edit_type: str = "improve_clarity",
        preserve_structure: bool = True
    ) -> Dict[str, Any]:
        """Main method for agentic content editing using LangChain tools"""
        try:
            # Step 1: Analyze content
            if content_type == "blueprint":
                analysis_result = await self.analyze_blueprint_content(content_id, user_id)
            elif content_type == "primitive":
                analysis_result = await self.analyze_primitive_content(content_id, user_id)
            else:
                analysis_result = await self.analyze_blueprint_content(content_id, user_id)  # Default
            
            # Step 2: Create edit plan
            edit_plan_result = await self.create_edit_plan(
                analysis_result, edit_instruction, content_type
            )
            
            # Step 3: Get original content (mock data)
            if content_type == "blueprint":
                original_content = {
                    "id": content_id,
                    "title": "Introduction to Machine Learning",
                    "description": "A comprehensive guide to ML fundamentals",
                    "sections": [
                        {"title": "What is ML?", "content": "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed."},
                        {"title": "Types of ML", "content": "Supervised, unsupervised, and reinforcement learning are the main categories."}
                    ],
                    "difficulty": "beginner",
                    "estimated_time": "4 hours"
                }
            elif content_type == "primitive":
                original_content = {
                    "id": content_id,
                    "title": "Machine Learning Fundamentals",
                    "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                    "concepts": ["supervised learning", "unsupervised learning", "reinforcement learning"],
                    "difficulty": "intermediate",
                    "prerequisites": ["basic programming", "statistics"],
                    "examples": ["spam detection", "recommendation systems", "image recognition"]
                }
            else:
                original_content = {
                    "id": content_id,
                    "title": "Default Content",
                    "content": "This is default content for testing purposes."
                }
            
            # Step 4: Apply edits
            edited_content_result = await self.apply_edits(
                json.dumps(original_content), edit_plan_result, content_type
            )
            
            return {
                "success": True,
                "content_type": content_type,
                "edit_type": edit_type,
                "original_content": original_content,
                "edited_content": edited_content_result,
                "edit_plan": edit_plan_result,
                "analysis": analysis_result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content_type": content_type,
                "edit_type": edit_type
            }
    
    async def get_editing_suggestions_agentically(
        self,
        content_id: str,
        content_type: str,
        user_id: str,
        suggestion_type: str = "general"
    ) -> Dict[str, Any]:
        """Get intelligent editing suggestions using LangChain tools"""
        try:
            # Get content for analysis (mock data)
            if content_type == "blueprint":
                content = {
                    "id": content_id,
                    "title": "Introduction to Machine Learning",
                    "description": "A comprehensive guide to ML fundamentals",
                    "sections": [
                        {"title": "What is ML?", "content": "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed."},
                        {"title": "Types of ML", "content": "Supervised, unsupervised, and reinforcement learning are the main categories."}
                    ],
                    "difficulty": "beginner",
                    "estimated_time": "4 hours"
                }
            elif content_type == "primitive":
                content = {
                    "id": content_id,
                    "title": "Machine Learning Fundamentals",
                    "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                    "concepts": ["supervised learning", "unsupervised learning", "reinforcement learning"],
                    "difficulty": "intermediate",
                    "prerequisites": ["basic programming", "statistics"],
                    "examples": ["spam detection", "recommendation systems", "image recognition"]
                }
            else:
                content = {
                    "id": content_id,
                    "title": "Default Content",
                    "content": "This is default content for testing purposes."
                }
            
            # Generate suggestions
            suggestions_result = await self.get_editing_suggestions(
                json.dumps(content), content_type, user_id
            )
            
            return {
                "success": True,
                "content_type": content_type,
                "suggestion_type": suggestion_type,
                "suggestions": suggestions_result,
                "content_id": content_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content_type": content_type,
                "suggestion_type": suggestion_type
            }
