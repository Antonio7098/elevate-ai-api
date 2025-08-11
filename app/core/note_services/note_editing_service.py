"""
Note Editing Service for agentic note editing.
Provides AI-powered note editing, suggestions, and content analysis.
"""

import time
from typing import List, Optional
from app.models.note_creation_models import (
    NoteEditingRequest, NoteEditingResponse, NoteEditingSuggestionsResponse,
    EditingSuggestion
)
from app.services.llm_service import LLMService


class NoteEditingService:
    """Service for agentic note editing and suggestions."""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def edit_note_agentically(
        self, 
        request: NoteEditingRequest
    ) -> NoteEditingResponse:
        """
        Edit a note using AI agentic capabilities.
        
        Args:
            request: Note editing request with instructions
            
        Returns:
            NoteEditingResponse with edited content and reasoning
        """
        start_time = time.time()
        
        try:
            # Step 1: Analyze current note content
            note_analysis = await self._analyze_note_content(request.note_id)
            
            # Step 2: Generate edit plan
            edit_plan = await self._create_edit_plan(
                note_analysis,
                request.edit_instruction,
                request.edit_type,
                request.preserve_original_structure
            )
            
            # Step 3: Apply edits
            edited_content, plain_text = await self._apply_edits(
                note_analysis,
                edit_plan,
                request.edit_type
            )
            
            # Step 4: Generate reasoning if requested
            reasoning = ""
            if request.include_reasoning:
                reasoning = await self._generate_edit_reasoning(
                    note_analysis,
                    edit_plan,
                    request.edit_instruction
                )
            
            processing_time = time.time() - start_time
            
            return NoteEditingResponse(
                success=True,
                edited_content=edited_content,
                plain_text=plain_text,
                edit_summary=edit_plan.get('summary', 'Note edited successfully'),
                reasoning=reasoning,
                message=f"Successfully edited note using {request.edit_type} approach"
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
        note_id: str,
        include_grammar: bool = True,
        include_clarity: bool = True,
        include_structure: bool = True
    ) -> NoteEditingSuggestionsResponse:
        """
        Get AI-powered editing suggestions for a note.
        
        Args:
            note_id: ID of the note to analyze
            include_grammar: Include grammar suggestions
            include_clarity: Include clarity improvements
            include_structure: Include structural suggestions
            
        Returns:
            NoteEditingSuggestionsResponse with suggestions
        """
        try:
            # Analyze note content
            note_analysis = await self._analyze_note_content(note_id)
            
            # Generate suggestions based on requested types
            suggestions = []
            
            if include_grammar:
                grammar_suggestions = await self._generate_grammar_suggestions(note_analysis)
                suggestions.extend(grammar_suggestions)
            
            if include_clarity:
                clarity_suggestions = await self._generate_clarity_suggestions(note_analysis)
                suggestions.extend(clarity_suggestions)
            
            if include_structure:
                structure_suggestions = await self._generate_structure_suggestions(note_analysis)
                suggestions.extend(structure_suggestions)
            
            return NoteEditingSuggestionsResponse(
                success=True,
                suggestions=suggestions,
                note_id=note_id,
                message=f"Generated {len(suggestions)} editing suggestions"
            )
            
        except Exception as e:
            return NoteEditingSuggestionsResponse(
                success=False,
                suggestions=[],
                note_id=note_id,
                message=f"Error generating suggestions: {str(e)}"
            )
    
    async def _analyze_note_content(self, note_id: str) -> dict:
        """Analyze the content of a note for editing purposes."""
        # In a real implementation, this would fetch the note from the database
        # For now, we'll create a mock analysis
        
        prompt = f"""
        Analyze the following note content for editing purposes.
        Note ID: {note_id}
        
        Please provide a comprehensive analysis including:
        1. Content structure and organization
        2. Writing quality and clarity
        3. Grammar and style issues
        4. Areas for improvement
        5. Overall assessment
        
        Return your analysis as a JSON object.
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1
        )
        
        return self._parse_analysis_response(response)
    
    async def _create_edit_plan(
        self,
        note_analysis: dict,
        edit_instruction: str,
        edit_type: str,
        preserve_structure: bool
    ) -> dict:
        """Create a plan for editing the note."""
        
        structure_instruction = "Preserve the original structure and organization." if preserve_structure else "Reorganize content for better flow and readability."
        
        prompt = f"""
        Create an edit plan for a note based on the following analysis and instruction.
        
        Note Analysis:
        {note_analysis}
        
        Edit Instruction: {edit_instruction}
        Edit Type: {edit_type}
        Structure Requirement: {structure_instruction}
        
        Create a detailed edit plan that includes:
        1. Specific changes to make
        2. Content reorganization if needed
        3. Style adjustments
        4. Quality improvements
        5. Summary of planned changes
        
        Return your plan as a JSON object with this structure:
        {{
            "summary": "Brief overview of planned changes",
            "changes": [
                {{
                    "type": "content|structure|style|grammar",
                    "description": "What to change",
                    "reason": "Why this change is needed"
                }}
            ],
            "new_structure": ["section1", "section2"],
            "style_guidelines": ["guideline1", "guideline2"]
        }}
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2500,
            temperature=0.2
        )
        
        return self._parse_edit_plan_response(response)
    
    async def _apply_edits(
        self,
        note_analysis: dict,
        edit_plan: dict,
        edit_type: str
    ) -> tuple[str, str]:
        """Apply the edit plan to create edited content."""
        
        prompt = f"""
        Apply the following edit plan to improve a note.
        
        Edit Plan:
        {edit_plan}
        
        Edit Type: {edit_type}
        
        Apply all the planned changes and return the edited note in BlockNote format.
        Ensure the result is:
        - Well-structured and organized
        - Clear and readable
        - Follows the edit plan exactly
        - Maintains high quality
        
        Return your response as a JSON object with this structure:
        {{
            "note_content": "Complete BlockNote JSON format content",
            "plain_text": "Plain text version of the edited note",
            "changes_applied": ["change1", "change2"]
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
    
    async def _generate_edit_reasoning(
        self,
        note_analysis: dict,
        edit_plan: dict,
        edit_instruction: str
    ) -> str:
        """Generate reasoning for the edits made."""
        
        prompt = f"""
        Explain the reasoning behind the edits made to a note.
        
        Original Analysis:
        {note_analysis}
        
        Edit Plan Applied:
        {edit_plan}
        
        User Instruction: {edit_instruction}
        
        Provide clear reasoning for:
        1. Why each change was necessary
        2. How the changes improve the note
        3. What problems were solved
        4. How the result better serves the user's needs
        
        Be concise but thorough in your explanation.
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.2
        )
        
        return response.strip()
    
    async def _generate_grammar_suggestions(self, note_analysis: dict) -> List[EditingSuggestion]:
        """Generate grammar-related editing suggestions."""
        
        prompt = f"""
        Analyze the following note for grammar and style improvements.
        
        Note Analysis:
        {note_analysis}
        
        Identify specific grammar, punctuation, and style issues.
        For each issue, provide:
        1. Type of problem
        2. Clear description
        3. Suggested fix
        4. Confidence level (0.0 to 1.0)
        5. Brief reasoning
        
        Return your suggestions as a JSON array with this structure:
        [
            {{
                "type": "grammar",
                "description": "Description of the issue",
                "suggested_change": "Specific fix to apply",
                "confidence": 0.9,
                "reasoning": "Why this change improves the note"
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
    
    async def _generate_clarity_suggestions(self, note_analysis: dict) -> List[EditingSuggestion]:
        """Generate clarity improvement suggestions."""
        
        prompt = f"""
        Analyze the following note for clarity and readability improvements.
        
        Note Analysis:
        {note_analysis}
        
        Identify areas where clarity can be improved:
        1. Unclear sentences or phrases
        2. Complex explanations that could be simplified
        3. Missing context or definitions
        4. Confusing organization
        
        For each issue, provide:
        1. Type of clarity problem
        2. Clear description
        3. Suggested improvement
        4. Confidence level (0.0 to 1.0)
        5. Brief reasoning
        
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
    
    async def _generate_structure_suggestions(self, note_analysis: dict) -> List[EditingSuggestion]:
        """Generate structural improvement suggestions."""
        
        prompt = f"""
        Analyze the following note for structural and organizational improvements.
        
        Note Analysis:
        {note_analysis}
        
        Identify structural issues:
        1. Poor organization or flow
        2. Missing logical connections
        3. Inconsistent formatting
        4. Poor section organization
        
        For each issue, provide:
        1. Type of structural problem
        2. Clear description
        3. Suggested reorganization
        4. Confidence level (0.0 to 1.0)
        5. Brief reasoning
        
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
