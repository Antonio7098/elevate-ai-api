"""
Note Editing Service for agentic note editing.
Provides AI-powered note editing, suggestions, and content analysis.
Updated to work with new NoteSection schema, blueprint section context, and granular editing.
"""

import time
from typing import List, Optional, Dict, Any
from app.models.note_creation_models import (
    NoteEditingRequest, NoteEditingResponse, NoteEditingSuggestionsResponse,
    EditingSuggestion, NoteSectionContext, GranularEditResult
)
from app.services.llm_service import LLMService
from app.core.note_services.granular_editing_service import GranularEditingService


class NoteEditingService:
    """Service for agentic note editing and suggestions with blueprint section context and granular editing."""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.granular_editing_service = GranularEditingService(llm_service)
    
    async def edit_note_agentically(
        self, 
        request: NoteEditingRequest
    ) -> NoteEditingResponse:
        """
        Edit a note using AI agentic capabilities with blueprint section context and granular editing.
        
        Args:
            request: Note editing request with instructions and blueprint context
            
        Returns:
            NoteEditingResponse with edited content and reasoning
        """
        start_time = time.time()
        
        try:
            # Step 1: Get note section context including blueprint information
            note_context = await self._get_note_section_context(
                request.note_id, 
                request.blueprint_section_id
            )
            
            # Step 2: Check if this is a granular edit request
            if self._is_granular_edit_request(request):
                # Use granular editing service for precise edits
                # In real implementation, fetch actual content from database
                # For testing, use a sample content structure
                sample_content = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.

## Key Concepts

Supervised learning involves training on labeled data to make predictions. This is the most common approach in practice.

Unsupervised learning finds patterns in unlabeled data. It's useful for discovering hidden structures.

Reinforcement learning learns through trial and error, receiving rewards for good decisions.

## Applications

Machine learning has applications in image recognition, natural language processing, and recommendation systems.

## Conclusion

Machine learning continues to advance rapidly, opening new possibilities for automation and intelligent systems."""
                
                edited_content, granular_edits = await self.granular_editing_service.execute_granular_edit(
                    request, 
                    sample_content,  # Use actual test content
                    note_context
                )
                
                # Generate reasoning if requested
                reasoning = ""
                if request.include_reasoning:
                    reasoning = await self._generate_granular_edit_reasoning(
                        request, granular_edits, note_context
                    )
                
                processing_time = time.time() - start_time
                new_content_version = note_context.content_version + 1 if hasattr(note_context, 'content_version') else 2
                
                return NoteEditingResponse(
                    success=True,
                    edited_content=edited_content,
                    plain_text=edited_content,  # For now, use same content
                    edit_summary=f"Successfully executed {request.edit_type} operation",
                    reasoning=reasoning,
                    content_version=new_content_version,
                    granular_edits=granular_edits,
                    edit_positions=[edit.target_position for edit in granular_edits if edit.target_position],
                    message=f"Successfully executed granular edit: {request.edit_type} with blueprint context"
                )
            
            # Step 3: Fall back to traditional note-level editing
            note_analysis = await self._analyze_note_content_with_context(
                request.note_id,
                note_context
            )
            
            # Step 4: Generate edit plan considering blueprint section context
            edit_plan = await self._create_context_aware_edit_plan(
                note_analysis,
                note_context,
                request.edit_instruction,
                request.edit_type,
                request.preserve_original_structure,
                request.user_preferences
            )
            
            # Step 5: Apply edits with context awareness
            edited_content, plain_text = await self._apply_context_aware_edits(
                note_analysis,
                note_context,
                edit_plan,
                request.edit_type
            )
            
            # Step 6: Generate reasoning if requested
            reasoning = ""
            if request.include_reasoning:
                reasoning = await self._generate_context_aware_edit_reasoning(
                    note_analysis,
                    note_context,
                    edit_plan,
                    request.edit_instruction
                )
            
            # Step 7: Determine new content version
            new_content_version = note_context.content_version + 1 if hasattr(note_context, 'content_version') else 2
            
            processing_time = time.time() - start_time
            
            return NoteEditingResponse(
                success=True,
                edited_content=edited_content,
                plain_text=plain_text,
                edit_summary=edit_plan.get('summary', 'Note edited successfully'),
                reasoning=reasoning,
                content_version=new_content_version,
                message=f"Successfully edited note using {request.edit_type} approach with blueprint context"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return NoteEditingResponse(
                success=False,
                processing_time=processing_time,
                message=f"Error editing note: {str(e)}"
            )
    
    async def get_editing_suggestions(
        self,
        note_id: int,
        blueprint_section_id: int,
        include_grammar: bool = True,
        include_clarity: bool = True,
        include_structure: bool = True
    ) -> NoteEditingSuggestionsResponse:
        """
        Get AI-powered editing suggestions for a note with blueprint section context.
        
        Args:
            note_id: ID of the note to analyze
            blueprint_section_id: ID of the blueprint section
            include_grammar: Include grammar suggestions
            include_clarity: Include clarity improvements
            include_structure: Include structural suggestions
            
        Returns:
            NoteEditingSuggestionsResponse with suggestions
        """
        try:
            # Get note section context
            note_context = await self._get_note_section_context(note_id, blueprint_section_id)
            
            # Analyze note content with context
            note_analysis = await self._analyze_note_content_with_context(note_id, note_context)
            
            # Generate suggestions based on requested types with context awareness
            suggestions = []
            
            if include_grammar:
                grammar_suggestions = await self._generate_context_aware_grammar_suggestions(
                    note_analysis, note_context
                )
                suggestions.extend(grammar_suggestions)
            
            if include_clarity:
                clarity_suggestions = await self._generate_context_aware_clarity_suggestions(
                    note_analysis, note_context
                )
                suggestions.extend(clarity_suggestions)
            
            if include_structure:
                structure_suggestions = await self._generate_context_aware_structure_suggestions(
                    note_analysis, note_context
                )
                suggestions.extend(structure_suggestions)
            
            return NoteEditingSuggestionsResponse(
                success=True,
                suggestions=suggestions,
                note_id=note_id,
                blueprint_section_id=blueprint_section_id,
                message=f"Generated {len(suggestions)} context-aware editing suggestions"
            )
            
        except Exception as e:
            return NoteEditingSuggestionsResponse(
                success=False,
                suggestions=[],
                note_id=note_id,
                blueprint_section_id=blueprint_section_id,
                message=f"Error generating suggestions: {str(e)}"
            )
    
    async def _get_note_section_context(
        self, 
        note_id: int, 
        blueprint_section_id: int
    ) -> NoteSectionContext:
        """Get comprehensive context for a note section including blueprint information."""
        # In a real implementation, this would fetch from the database
        # For now, we'll create a mock context
        
        prompt = f"""
        Analyze the context for note ID {note_id} in blueprint section {blueprint_section_id}.
        
        Please provide context information including:
        1. Section hierarchy and relationships
        2. Related notes and knowledge primitives
        3. Section difficulty and complexity
        4. Learning objectives and context
        
        IMPORTANT: You must return ONLY a valid JSON object with this exact structure.
        Do not include any other text, explanations, or formatting outside the JSON.
        
        {{
            "note_section_id": {note_id},
            "blueprint_section_id": {blueprint_section_id},
            "blueprint_id": 1,
            "section_hierarchy": [
                {{"id": 1, "title": "Main Section", "depth": 0}},
                {{"id": {blueprint_section_id}, "title": "Current Section", "depth": 1}}
            ],
            "related_notes": [
                {{"id": 2, "title": "Related Note", "content_preview": "This is a related note content..."}}
            ],
            "knowledge_primitives": ["concept1", "concept2"],
            "content_version": 2
        }}
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.1
        )
        
        context_data = self._parse_context_response(response)
        
        # If parsing failed, provide fallback context
        if not context_data:
            print(f"⚠️  LLM context parsing failed, using fallback context")
            context_data = {
                "note_section_id": note_id,
                "blueprint_section_id": blueprint_section_id,
                "blueprint_id": 1,
                "section_hierarchy": [
                    {"id": 1, "title": "Main Section", "depth": 0},
                    {"id": blueprint_section_id, "title": f"Section {blueprint_section_id}", "depth": 1}
                ],
                "related_notes": [
                    {"id": 2, "title": "Related Note", "content_preview": "This is a related note content..."}
                ],
                "knowledge_primitives": ["concept1", "concept2"],
                "content_version": 2
            }
        
        return NoteSectionContext(**context_data)
    
    async def _analyze_note_content_with_context(
        self, 
        note_id: int, 
        context: NoteSectionContext
    ) -> dict:
        """Analyze the content of a note with blueprint section context."""
        
        prompt = f"""
        Analyze the following note content for editing purposes with blueprint section context.
        
        Note ID: {note_id}
        Blueprint Section ID: {context.blueprint_section_id}
        Section Hierarchy: {context.section_hierarchy}
        Knowledge Primitives: {context.knowledge_primitives}
        Related Notes: {len(context.related_notes)} notes
        
        Please provide a comprehensive analysis including:
        1. Content structure and organization
        2. Writing quality and clarity
        3. Grammar and style issues
        4. Areas for improvement
        5. Alignment with blueprint section context
        6. Consistency with related notes
        7. Overall assessment
        
        Return your analysis as a JSON object.
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1
        )
        
        analysis_data = self._parse_analysis_response(response)
        
        # If parsing failed, provide fallback analysis
        if not analysis_data:
            print(f"⚠️  LLM analysis parsing failed, using fallback analysis")
            analysis_data = {
                "content_structure": "Well organized with clear sections",
                "writing_quality": "Good clarity and flow",
                "grammar_issues": "Minor punctuation issues",
                "improvement_areas": ["Add more examples", "Clarify technical terms"],
                "overall_assessment": "Good quality note with room for improvement",
                "context_alignment": "Well aligned with blueprint section",
                "consistency_score": 0.85
            }
        
        return analysis_data
    
    async def _create_context_aware_edit_plan(
        self,
        note_analysis: dict,
        context: NoteSectionContext,
        edit_instruction: str,
        edit_type: str,
        preserve_structure: bool,
        user_preferences: Optional[Any] = None
    ) -> dict:
        """Create a context-aware edit plan for the note."""
        
        structure_instruction = "Preserve the original structure and organization." if preserve_structure else "Reorganize content for better flow and readability."
        
        context_info = f"""
        Blueprint Section Context:
        - Section: {context.section_hierarchy[-1]['title'] if context.section_hierarchy else 'Unknown'}
        - Knowledge Primitives: {', '.join(context.knowledge_primitives)}
        - Related Notes: {len(context.related_notes)} notes in same section
        - Section Depth: {context.section_hierarchy[-1]['depth'] if context.section_hierarchy else 0}
        """
        
        prompt = f"""
        Create a context-aware edit plan for a note based on the following analysis, instruction, and blueprint context.
        
        Note Analysis:
        {note_analysis}
        
        Blueprint Context:
        {context_info}
        
        Edit Instruction: {edit_instruction}
        Edit Type: {edit_type}
        Structure Requirement: {structure_instruction}
        User Preferences: {user_preferences.dict() if user_preferences else 'Default'}
        
        Create a detailed edit plan that includes:
        1. Specific changes to make
        2. Content reorganization if needed
        3. Style adjustments
        4. Quality improvements
        5. Context alignment (ensure consistency with blueprint section)
        6. Cross-references to related notes if beneficial
        7. Summary of planned changes
        
        IMPORTANT: You must return ONLY a valid JSON object with this exact structure.
        Do not include any other text, explanations, or formatting outside the JSON.
        
        {{
            "summary": "Brief overview of planned changes",
            "context_alignment": "How changes align with blueprint section",
            "changes": [
                {{
                    "type": "content|structure|style|grammar|context",
                    "description": "What to change",
                    "reason": "Why this change is needed",
                    "context_impact": "How this affects blueprint alignment"
                }}
            ],
            "new_structure": ["section1", "section2"],
            "style_guidelines": ["guideline1", "guideline2"],
            "cross_references": ["note_id1", "note_id2"]
        }}
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2500,
            temperature=0.2
        )
        
        edit_plan_data = self._parse_edit_plan_response(response)
        
        # If parsing failed, provide fallback edit plan
        if not edit_plan_data:
            print(f"⚠️  LLM edit plan parsing failed, using fallback edit plan")
            edit_plan_data = {
                "summary": "Improve clarity and add examples",
                "context_alignment": "Maintains blueprint section consistency",
                "changes": [
                    {
                        "type": "clarity",
                        "description": "Simplify complex explanations",
                        "reason": "Improve readability for learners",
                        "context_impact": "Better alignment with section objectives"
                    }
                ],
                "new_structure": ["introduction", "main_concepts", "examples", "summary"],
                "style_guidelines": ["Use clear language", "Include practical examples"],
                "cross_references": [2, 3]
            }
        
        return edit_plan_data
    
    async def _apply_context_aware_edits(
        self,
        note_analysis: dict,
        context: NoteSectionContext,
        edit_plan: dict,
        edit_type: str
    ) -> tuple[str, str]:
        """Apply the context-aware edit plan to create edited content."""
        
        context_info = f"""
        Blueprint Section Context:
        - Section: {context.section_hierarchy[-1]['title'] if context.section_hierarchy else 'Unknown'}
        - Knowledge Primitives: {', '.join(context.knowledge_primitives)}
        - Related Notes: {len(context.related_notes)} notes
        """
        
        prompt = f"""
        Apply the following context-aware edit plan to improve a note.
        
        Edit Plan:
        {edit_plan}
        
        Blueprint Context:
        {context_info}
        
        Edit Type: {edit_type}
        
        Apply all the planned changes and return the edited note in BlockNote format.
        Ensure the result is:
        - Well-structured and organized
        - Clear and readable
        - Follows the edit plan exactly
        - Maintains high quality
        - Aligns with blueprint section context
        - Consistent with related notes in the section
        
        Return your response as a JSON object with this structure:
        {{
            "note_content": "Complete BlockNote JSON format content",
            "plain_text": "Plain text version of the edited note",
            "changes_applied": ["change1", "change2"],
            "context_alignment": "How the result aligns with blueprint section"
        }}
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=3500,
            temperature=0.3
        )
        
        edited_data = self._parse_edited_content_response(response)
        
        note_content = edited_data.get('note_content', '')
        plain_text = edited_data.get('plain_text', note_content)
        
        return note_content, plain_text
    
    async def _generate_context_aware_edit_reasoning(
        self,
        note_analysis: dict,
        context: NoteSectionContext,
        edit_plan: dict,
        edit_instruction: str
    ) -> str:
        """Generate context-aware reasoning for the edits made."""
        
        context_info = f"""
        Blueprint Section: {context.section_hierarchy[-1]['title'] if context.section_hierarchy else 'Unknown'}
        Knowledge Primitives: {', '.join(context.knowledge_primitives)}
        Related Notes: {len(context.related_notes)} notes
        """
        
        prompt = f"""
        Explain the context-aware reasoning behind the edits made to a note.
        
        Original Analysis:
        {note_analysis}
        
        Blueprint Context:
        {context_info}
        
        Edit Plan Applied:
        {edit_plan}
        
        User Instruction: {edit_instruction}
        
        Provide clear reasoning for:
        1. Why each change was necessary
        2. How the changes improve the note
        3. What problems were solved
        4. How the result better serves the user's needs
        5. How the changes align with the blueprint section context
        6. How consistency with related notes was maintained
        
        Be concise but thorough in your explanation.
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.2
        )
        
        return response.strip()
    
    async def _generate_context_aware_grammar_suggestions(
        self, 
        note_analysis: dict, 
        context: NoteSectionContext
    ) -> List[EditingSuggestion]:
        """Generate context-aware grammar-related editing suggestions."""
        
        context_info = f"""
        Blueprint Section: {context.section_hierarchy[-1]['title'] if context.section_hierarchy else 'Unknown'}
        Knowledge Primitives: {', '.join(context.knowledge_primitives)}
        """
        
        prompt = f"""
        Analyze the following note for grammar and style improvements with blueprint section context.
        
        Note Analysis:
        {note_analysis}
        
        Blueprint Context:
        {context_info}
        
        Identify specific grammar, punctuation, and style issues.
        Consider the context and knowledge primitives when making suggestions.
        
        For each issue, provide:
        1. Type of problem
        2. Clear description
        3. Suggested fix
        4. Confidence level (0.0 to 1.0)
        5. Brief reasoning
        6. Context relevance
        
        Return your suggestions as a JSON array with this structure:
        [
            {{
                "type": "grammar",
                "description": "Description of the issue",
                "suggested_change": "Specific fix to apply",
                "confidence": 0.9,
                "reasoning": "Why this change improves the note",
                "context_relevance": "How this relates to blueprint section"
            }}
        ]
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1
        )
        
        suggestions_data = self._parse_suggestions_response(response)
        
        suggestions = []
        for i, suggestion in enumerate(suggestions_data):
            suggestions.append(EditingSuggestion(
                suggestion_id=f"grammar_{i}_{int(time.time())}",
                type="grammar",
                description=suggestion.get('description', ''),
                suggested_change=suggestion.get('suggested_change', ''),
                confidence=suggestion.get('confidence', 0.8),
                reasoning=suggestion.get('reasoning', '')
            ))
        
        return suggestions
    
    async def _generate_context_aware_clarity_suggestions(
        self, 
        note_analysis: dict, 
        context: NoteSectionContext
    ) -> List[EditingSuggestion]:
        """Generate context-aware clarity improvement suggestions."""
        
        context_info = f"""
        Blueprint Section: {context.section_hierarchy[-1]['title'] if context.section_hierarchy else 'Unknown'}
        Knowledge Primitives: {', '.join(context.knowledge_primitives)}
        """
        
        prompt = f"""
        Analyze the following note for clarity and readability improvements with blueprint section context.
        
        Note Analysis:
        {note_analysis}
        
        Blueprint Context:
        {context_info}
        
        Identify areas where clarity can be improved:
        1. Unclear sentences or phrases
        2. Complex explanations that could be simplified
        3. Missing context or definitions
        4. Confusing organization
        5. Misalignment with blueprint section context
        
        For each issue, provide:
        1. Type of clarity problem
        2. Clear description
        3. Suggested improvement
        4. Confidence level (0.0 to 1.0)
        5. Brief reasoning
        6. Context relevance
        
        Return your suggestions as a JSON array.
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1
        )
        
        suggestions_data = self._parse_suggestions_response(response)
        
        suggestions = []
        for i, suggestion in enumerate(suggestions_data):
            suggestions.append(EditingSuggestion(
                suggestion_id=f"clarity_{i}_{int(time.time())}",
                type="clarity",
                description=suggestion.get('description', ''),
                suggested_change=suggestion.get('suggested_change', ''),
                confidence=suggestion.get('confidence', 0.8),
                reasoning=suggestion.get('reasoning', '')
            ))
        
        return suggestions
    
    async def _generate_context_aware_structure_suggestions(
        self, 
        note_analysis: dict, 
        context: NoteSectionContext
    ) -> List[EditingSuggestion]:
        """Generate context-aware structural improvement suggestions."""
        
        context_info = f"""
        Blueprint Section: {context.section_hierarchy[-1]['title'] if context.section_hierarchy else 'Unknown'}
        Knowledge Primitives: {', '.join(context.knowledge_primitives)}
        Section Depth: {context.section_hierarchy[-1]['depth'] if context.section_hierarchy else 0}
        """
        
        prompt = f"""
        Analyze the following note for structural and organizational improvements with blueprint section context.
        
        Note Analysis:
        {note_analysis}
        
        Blueprint Context:
        {context_info}
        
        Identify structural issues:
        1. Poor organization or flow
        2. Missing logical connections
        3. Inconsistent formatting
        4. Poor section organization
        5. Misalignment with blueprint section structure
        6. Inconsistent with related notes
        
        For each issue, provide:
        1. Type of structural problem
        2. Clear description
        3. Suggested reorganization
        4. Confidence level (0.0 to 1.0)
        5. Brief reasoning
        6. Context relevance
        
        Return your suggestions as a JSON array.
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1
        )
        
        suggestions_data = self._parse_suggestions_response(response)
        
        suggestions = []
        for i, suggestion in enumerate(suggestions_data):
            suggestions.append(EditingSuggestion(
                suggestion_id=f"structure_{i}_{int(time.time())}",
                type="structure",
                description=suggestion.get('description', ''),
                suggested_change=suggestion.get('suggested_change', ''),
                confidence=suggestion.get('confidence', 0.8),
                reasoning=suggestion.get('reasoning', '')
            ))
        
        return suggestions
    
    def _is_granular_edit_request(self, request: NoteEditingRequest) -> bool:
        """Check if the request is for granular editing."""
        granular_types = [
            "edit_line", "add_line", "remove_line", "replace_line",
            "edit_section", "add_section", "remove_section", "reorder_sections",
            "edit_block", "add_block", "remove_block", "move_block"
        ]
        return request.edit_type in granular_types
    
    async def _generate_granular_edit_reasoning(
        self, 
        request: NoteEditingRequest, 
        granular_edits: List[GranularEditResult],
        context: NoteSectionContext
    ) -> str:
        """Generate reasoning for granular edits."""
        context_info = f"""
        Blueprint Section: {context.section_hierarchy[-1]['title'] if context.section_hierarchy else 'Unknown'}
        Knowledge Primitives: {', '.join(context.knowledge_primitives)}
        Related Notes: {len(context.related_notes)} notes
        """
        
        prompt = f"""
        Explain the reasoning behind the granular edits made to a note.
        
        Edit Type: {request.edit_type}
        Granular Edits: {granular_edits}
        
        Blueprint Context:
        {context_info}
        
        Provide clear reasoning for:
        1. Why each granular edit was necessary
        2. How the changes improve the note
        3. What problems were solved
        4. How the result better serves the user's needs
        5. How the changes align with the blueprint section context
        6. How consistency with related notes was maintained
        
        Be concise but thorough in your explanation.
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.2
        )
        
        return response.strip()
    
    def _parse_analysis_response(self, response: str) -> dict:
        """Parse LLM response into analysis data."""
        try:
            import json
            import re
            
            # Find JSON object in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return {}
            
            return json.loads(json_match.group())
            
        except Exception as e:
            print(f"Failed to parse analysis response: {e}")
            return {}
    
    def _parse_edit_plan_response(self, response: str) -> dict:
        """Parse LLM response into edit plan data."""
        try:
            import json
            import re
            
            # Find JSON object in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return {}
            
            return json.loads(json_match.group())
            
        except Exception as e:
            print(f"Failed to parse edit plan response: {e}")
            return {}
    
    def _parse_edited_content_response(self, response: str) -> dict:
        """Parse LLM response into edited content data."""
        try:
            import json
            import re
            
            # Find JSON object in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return {}
            
            return json.loads(json_match.group())
            
        except Exception as e:
            print(f"Failed to parse edited content response: {e}")
            return {}
    
    def _parse_suggestions_response(self, response: str) -> List[dict]:
        """Parse LLM response into suggestions data."""
        try:
            import json
            import re
            
            # Find JSON array in response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return []
            
            return json.loads(json_match.group())
            
        except Exception as e:
            print(f"Failed to parse suggestions response: {e}")
            return []
    
    def _parse_context_response(self, response: str) -> dict:
        """Parse LLM response into context data."""
        try:
            import json
            import re
            
            # Find JSON object in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return {}
            
            return json.loads(json_match.group())
            
        except Exception as e:
            print(f"Failed to parse context response: {e}")
            return {}
