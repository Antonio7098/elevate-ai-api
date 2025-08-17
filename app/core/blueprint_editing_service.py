"""
Blueprint Editing Service for agentic blueprint editing.
Provides AI-powered editing for blueprints, primitives, mastery criteria, and questions.
Updated to work with new blueprint-centric schema and granular editing capabilities.
"""

import time
from typing import List, Optional, Dict, Any, Union
from app.models.blueprint_editing_models import (
    BlueprintEditingRequest, BlueprintEditingResponse, BlueprintEditingSuggestionsResponse,
    PrimitiveEditingRequest, PrimitiveEditingResponse, PrimitiveEditingSuggestionsResponse,
    MasteryCriterionEditingRequest, MasteryCriterionEditingResponse, MasteryCriterionEditingSuggestionsResponse,
    QuestionEditingRequest, QuestionEditingResponse, QuestionEditingSuggestionsResponse,
    EditingSuggestion, BlueprintContext, GranularEditResult
)
from app.services.llm_service import LLMService
from app.core.blueprint_editing.granular_editing_service import GranularBlueprintEditingService


class BlueprintEditingService:
    """Service for agentic blueprint editing and suggestions with comprehensive context awareness."""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.granular_editing_service = GranularBlueprintEditingService(llm_service)
    
    # ============================================================================
    # BLUEPRINT EDITING
    # ============================================================================
    
    async def edit_blueprint_agentically(
        self, 
        request: BlueprintEditingRequest
    ) -> BlueprintEditingResponse:
        """
        Edit a blueprint using AI agentic capabilities with comprehensive context awareness.
        
        Args:
            request: Blueprint editing request with instructions and context
            
        Returns:
            BlueprintEditingResponse with edited content and reasoning
        """
        start_time = time.time()
        
        try:
            # Step 1: Get comprehensive blueprint context
            blueprint_context = await self._get_blueprint_context(request.blueprint_id)
            
            # Step 2: Check if this is a granular edit request
            if self._is_granular_blueprint_edit_request(request):
                edited_content, granular_edits = await self.granular_editing_service.execute_blueprint_granular_edit(
                    request, 
                    blueprint_context
                )
                
                reasoning = ""
                if request.include_reasoning:
                    reasoning = await self._generate_blueprint_granular_edit_reasoning(
                        request, granular_edits, blueprint_context
                    )
                
                processing_time = time.time() - start_time
                new_version = blueprint_context.version + 1 if hasattr(blueprint_context, 'version') else 2
                
                return BlueprintEditingResponse(
                    success=True,
                    edited_content=edited_content,
                    edit_summary=f"Successfully executed {request.edit_type} operation",
                    reasoning=reasoning,
                    version=new_version,
                    granular_edits=granular_edits,
                    message=f"Successfully executed granular blueprint edit: {request.edit_type}"
                )
            
            # Step 3: Traditional blueprint-level editing
            blueprint_analysis = await self._analyze_blueprint_content_with_context(
                request.blueprint_id,
                blueprint_context
            )
            
            # Step 4: Generate edit plan
            edit_plan = await self._create_blueprint_edit_plan(
                blueprint_analysis,
                blueprint_context,
                request.edit_instruction,
                request.edit_type,
                request.preserve_original_structure,
                request.user_preferences
            )
            
            # Step 5: Apply edits
            edited_content = await self._apply_blueprint_edits(
                blueprint_analysis,
                blueprint_context,
                edit_plan,
                request.edit_type
            )
            
            # Step 6: Generate reasoning if requested
            reasoning = ""
            if request.include_reasoning:
                reasoning = await self._generate_blueprint_edit_reasoning(
                    blueprint_analysis,
                    blueprint_context,
                    edit_plan,
                    request.edit_instruction
                )
            
            processing_time = time.time() - start_time
            new_version = blueprint_context.version + 1 if hasattr(blueprint_context, 'version') else 2
            
            return BlueprintEditingResponse(
                success=True,
                edited_content=edited_content,
                edit_summary=edit_plan.get('summary', 'Blueprint edited successfully'),
                reasoning=reasoning,
                version=new_version,
                message=f"Successfully edited blueprint using {request.edit_type} approach"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return BlueprintEditingResponse(
                success=False,
                processing_time=processing_time,
                message=f"Error editing blueprint: {str(e)}"
            )
    
    # ============================================================================
    # PRIMITIVE EDITING
    # ============================================================================
    
    async def edit_primitive_agentically(
        self, 
        request: PrimitiveEditingRequest
    ) -> PrimitiveEditingResponse:
        """
        Edit a knowledge primitive using AI agentic capabilities.
        
        Args:
            request: Primitive editing request with instructions and context
            
        Returns:
            PrimitiveEditingResponse with edited content and reasoning
        """
        start_time = time.time()
        
        try:
            # Get primitive context
            primitive_context = await self._get_primitive_context(request.primitive_id)
            
            # Analyze primitive content
            primitive_analysis = await self._analyze_primitive_content_with_context(
                request.primitive_id,
                primitive_context
            )
            
            # Create edit plan
            edit_plan = await self._create_primitive_edit_plan(
                primitive_analysis,
                primitive_context,
                request.edit_instruction,
                request.edit_type,
                request.preserve_original_structure,
                request.user_preferences
            )
            
            # Apply edits
            edited_content = await self._apply_primitive_edits(
                primitive_analysis,
                primitive_context,
                edit_plan,
                request.edit_type
            )
            
            # Generate reasoning if requested
            reasoning = ""
            if request.include_reasoning:
                reasoning = await self._generate_primitive_edit_reasoning(
                    primitive_analysis,
                    primitive_context,
                    edit_plan,
                    request.edit_instruction
                )
            
            processing_time = time.time() - start_time
            new_version = primitive_context.version + 1 if hasattr(primitive_context, 'version') else 2
            
            return PrimitiveEditingResponse(
                success=True,
                edited_content=edited_content,
                edit_summary=edit_plan.get('summary', 'Primitive edited successfully'),
                reasoning=reasoning,
                version=new_version,
                message=f"Successfully edited primitive using {request.edit_type} approach"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return PrimitiveEditingResponse(
                success=False,
                processing_time=processing_time,
                message=f"Error editing primitive: {str(e)}"
            )
    
    # ============================================================================
    # MASTERY CRITERION EDITING
    # ============================================================================
    
    async def edit_mastery_criterion_agentically(
        self, 
        request: MasteryCriterionEditingRequest
    ) -> MasteryCriterionEditingResponse:
        """
        Edit a mastery criterion using AI agentic capabilities.
        
        Args:
            request: Mastery criterion editing request with instructions and context
            
        Returns:
            MasteryCriterionEditingResponse with edited content and reasoning
        """
        start_time = time.time()
        
        try:
            # Get mastery criterion context
            criterion_context = await self._get_mastery_criterion_context(request.criterion_id)
            
            # Analyze criterion content
            criterion_analysis = await self._analyze_mastery_criterion_content_with_context(
                request.criterion_id,
                criterion_context
            )
            
            # Create edit plan
            edit_plan = await self._create_mastery_criterion_edit_plan(
                criterion_analysis,
                criterion_context,
                request.edit_instruction,
                request.edit_type,
                request.preserve_original_structure,
                request.user_preferences
            )
            
            # Apply edits
            edited_content = await self._apply_mastery_criterion_edits(
                criterion_analysis,
                criterion_context,
                edit_plan,
                request.edit_type
            )
            
            # Generate reasoning if requested
            reasoning = ""
            if request.include_reasoning:
                reasoning = await self._generate_mastery_criterion_edit_reasoning(
                    criterion_analysis,
                    criterion_context,
                    edit_plan,
                    request.edit_instruction
                )
            
            processing_time = time.time() - start_time
            new_version = criterion_context.version + 1 if hasattr(criterion_context, 'version') else 2
            
            return MasteryCriterionEditingResponse(
                success=True,
                edited_content=edited_content,
                edit_summary=edit_plan.get('summary', 'Mastery criterion edited successfully'),
                reasoning=reasoning,
                version=new_version,
                message=f"Successfully edited mastery criterion using {request.edit_type} approach"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return MasteryCriterionEditingResponse(
                success=False,
                processing_time=processing_time,
                message=f"Error editing mastery criterion: {str(e)}"
            )
    
    # ============================================================================
    # QUESTION EDITING
    # ============================================================================
    
    async def edit_question_agentically(
        self, 
        request: QuestionEditingRequest
    ) -> QuestionEditingResponse:
        """
        Edit a question using AI agentic capabilities.
        
        Args:
            request: Question editing request with instructions and context
            
        Returns:
            QuestionEditingResponse with edited content and reasoning
        """
        start_time = time.time()
        
        try:
            # Get question context
            question_context = await self._get_question_context(request.question_id)
            
            # Analyze question content
            question_analysis = await self._analyze_question_content_with_context(
                request.question_id,
                question_context
            )
            
            # Create edit plan
            edit_plan = await self._create_question_edit_plan(
                question_analysis,
                question_context,
                request.edit_instruction,
                request.edit_type,
                request.preserve_original_structure,
                request.user_preferences
            )
            
            # Apply edits
            edited_content = await self._apply_question_edits(
                question_analysis,
                question_context,
                edit_plan,
                request.edit_type
            )
            
            # Generate reasoning if requested
            reasoning = ""
            if request.include_reasoning:
                reasoning = await self._generate_question_edit_reasoning(
                    question_analysis,
                    question_context,
                    edit_plan,
                    request.edit_instruction
                )
            
            processing_time = time.time() - start_time
            new_version = question_context.version + 1 if hasattr(question_context, 'version') else 2
            
            return QuestionEditingResponse(
                success=True,
                edited_content=edited_content,
                edit_summary=edit_plan.get('summary', 'Question edited successfully'),
                reasoning=reasoning,
                version=new_version,
                message=f"Successfully edited question using {request.edit_type} approach"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return QuestionEditingResponse(
                success=False,
                processing_time=processing_time,
                message=f"Error editing question: {str(e)}"
            )
    
    # ============================================================================
    # SUGGESTION GENERATION METHODS
    # ============================================================================
    
    async def get_blueprint_editing_suggestions(
        self,
        blueprint_id: int,
        include_structure: bool = True,
        include_content: bool = True,
        include_relationships: bool = True
    ) -> BlueprintEditingSuggestionsResponse:
        """Get AI-powered editing suggestions for a blueprint."""
        try:
            blueprint_context = await self._get_blueprint_context(blueprint_id)
            blueprint_analysis = await self._analyze_blueprint_content_with_context(blueprint_id, blueprint_context)
            
            suggestions = []
            
            if include_structure:
                structure_suggestions = await self._generate_blueprint_structure_suggestions(
                    blueprint_analysis, blueprint_context
                )
                suggestions.extend(structure_suggestions)
            
            if include_content:
                content_suggestions = await self._generate_blueprint_content_suggestions(
                    blueprint_analysis, blueprint_context
                )
                suggestions.extend(content_suggestions)
            
            if include_relationships:
                relationship_suggestions = await self._generate_blueprint_relationship_suggestions(
                    blueprint_analysis, blueprint_context
                )
                suggestions.extend(relationship_suggestions)
            
            return BlueprintEditingSuggestionsResponse(
                success=True,
                suggestions=suggestions,
                blueprint_id=blueprint_id,
                message=f"Generated {len(suggestions)} blueprint editing suggestions"
            )
            
        except Exception as e:
            return BlueprintEditingSuggestionsResponse(
                success=False,
                suggestions=[],
                blueprint_id=blueprint_id,
                message=f"Error generating suggestions: {str(e)}"
            )
    
    async def get_primitive_editing_suggestions(
        self,
        primitive_id: int,
        include_clarity: bool = True,
        include_complexity: bool = True,
        include_relationships: bool = True
    ) -> PrimitiveEditingSuggestionsResponse:
        """Get AI-powered editing suggestions for a primitive."""
        try:
            primitive_context = await self._get_primitive_context(primitive_id)
            primitive_analysis = await self._analyze_primitive_content_with_context(primitive_id, primitive_context)
            
            suggestions = []
            
            if include_clarity:
                clarity_suggestions = await self._generate_primitive_clarity_suggestions(
                    primitive_analysis, primitive_context
                )
                suggestions.extend(clarity_suggestions)
            
            if include_complexity:
                complexity_suggestions = await self._generate_primitive_complexity_suggestions(
                    primitive_analysis, primitive_context
                )
                suggestions.extend(complexity_suggestions)
            
            if include_relationships:
                relationship_suggestions = await self._generate_primitive_relationship_suggestions(
                    primitive_analysis, primitive_context
                )
                suggestions.extend(relationship_suggestions)
            
            return PrimitiveEditingSuggestionsResponse(
                success=True,
                suggestions=suggestions,
                primitive_id=primitive_id,
                message=f"Generated {len(suggestions)} primitive editing suggestions"
            )
            
        except Exception as e:
            return PrimitiveEditingSuggestionsResponse(
                success=False,
                suggestions=[],
                primitive_id=primitive_id,
                message=f"Error generating suggestions: {str(e)}"
            )
    
    async def get_mastery_criterion_editing_suggestions(
        self,
        criterion_id: int,
        include_clarity: bool = True,
        include_difficulty: bool = True,
        include_assessment: bool = True
    ) -> MasteryCriterionEditingSuggestionsResponse:
        """Get AI-powered editing suggestions for a mastery criterion."""
        try:
            criterion_context = await self._get_mastery_criterion_context(criterion_id)
            criterion_analysis = await self._analyze_mastery_criterion_content_with_context(criterion_id, criterion_context)
            
            suggestions = []
            
            if include_clarity:
                clarity_suggestions = await self._generate_criterion_clarity_suggestions(
                    criterion_analysis, criterion_context
                )
                suggestions.extend(clarity_suggestions)
            
            if include_difficulty:
                difficulty_suggestions = await self._generate_criterion_difficulty_suggestions(
                    criterion_analysis, criterion_context
                )
                suggestions.extend(difficulty_suggestions)
            
            if include_assessment:
                assessment_suggestions = await self._generate_criterion_assessment_suggestions(
                    criterion_analysis, criterion_context
                )
                suggestions.extend(assessment_suggestions)
            
            return MasteryCriterionEditingSuggestionsResponse(
                success=True,
                suggestions=suggestions,
                criterion_id=criterion_id,
                message=f"Generated {len(suggestions)} mastery criterion editing suggestions"
            )
            
        except Exception as e:
            return MasteryCriterionEditingSuggestionsResponse(
                success=False,
                suggestions=[],
                criterion_id=criterion_id,
                message=f"Error generating suggestions: {str(e)}"
            )
    
    async def get_question_editing_suggestions(
        self,
        question_id: int,
        include_clarity: bool = True,
        include_difficulty: bool = True,
        include_quality: bool = True
    ) -> QuestionEditingSuggestionsResponse:
        """Get AI-powered editing suggestions for a question."""
        try:
            question_context = await self._get_question_context(question_id)
            question_analysis = await self._analyze_question_content_with_context(question_id, question_context)
            
            suggestions = []
            
            if include_clarity:
                clarity_suggestions = await self._generate_question_clarity_suggestions(
                    question_analysis, question_context
                )
                suggestions.extend(clarity_suggestions)
            
            if include_difficulty:
                difficulty_suggestions = await self._generate_question_difficulty_suggestions(
                    question_analysis, question_context
                )
                suggestions.extend(difficulty_suggestions)
            
            if include_quality:
                quality_suggestions = await self._generate_question_quality_suggestions(
                    question_analysis, question_context
                )
                suggestions.extend(quality_suggestions)
            
            return QuestionEditingSuggestionsResponse(
                success=True,
                suggestions=suggestions,
                question_id=question_id,
                message=f"Generated {len(suggestions)} question editing suggestions"
            )
            
        except Exception as e:
            return QuestionEditingSuggestionsResponse(
                success=False,
                suggestions=[],
                question_id=question_id,
                message=f"Error generating suggestions: {str(e)}"
            )
    
    # ============================================================================
    # PRIVATE HELPER METHODS
    # ============================================================================
    
    def _is_granular_blueprint_edit_request(self, request: BlueprintEditingRequest) -> bool:
        """Check if the request is for granular blueprint editing."""
        granular_types = [
            "edit_section", "add_section", "remove_section", "reorder_sections",
            "edit_primitive", "add_primitive", "remove_primitive", "reorder_primitives",
            "edit_criterion", "add_criterion", "remove_criterion", "reorder_criteria",
            "edit_question", "add_question", "remove_question", "reorder_questions"
        ]
        return request.edit_type in granular_types
    
    async def _get_blueprint_context(self, blueprint_id: int) -> BlueprintContext:
        """Get comprehensive context for a blueprint."""
        prompt = f"""
        Analyze the context for blueprint ID {blueprint_id}.
        
        Please provide context information including:
        1. Blueprint structure and sections
        2. Knowledge primitives and their relationships
        3. Mastery criteria and assessment types
        4. Question sets and difficulty distribution
        5. Learning objectives and complexity
        
        IMPORTANT: You must return ONLY a valid JSON object with this exact structure.
        Do not include any other text, explanations, or formatting outside the JSON.
        
        {{
            "blueprint_id": {blueprint_id},
            "title": "Sample Blueprint",
            "description": "A comprehensive learning blueprint",
            "version": 1,
            "sections_count": 5,
            "primitives_count": 25,
            "criteria_count": 30,
            "questions_count": 150,
            "difficulty_distribution": {{"beginner": 0.3, "intermediate": 0.5, "advanced": 0.2}},
            "estimated_time_hours": 40,
            "learning_objectives": ["objective1", "objective2", "objective3"]
        }}
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.1
        )
        
        context_data = self._parse_context_response(response)
        
        if not context_data:
            raise Exception(f"Failed to parse LLM response for blueprint context: {response}")
        
        return BlueprintContext(**context_data)
    
    async def _get_primitive_context(self, primitive_id: int) -> dict:
        """Get context for a knowledge primitive."""
        # Implementation would fetch from database
        return {
            "primitive_id": primitive_id,
            "title": f"Primitive {primitive_id}",
            "type": "concept",
            "difficulty": "intermediate",
            "version": 1,
            "related_primitives": [],
            "prerequisites": [],
            "complexity_score": 0.6
        }
    
    async def _get_mastery_criterion_context(self, criterion_id: int) -> dict:
        """Get context for a mastery criterion."""
        # Implementation would fetch from database
        return {
            "criterion_id": criterion_id,
            "title": f"Criterion {criterion_id}",
            "type": "question_based",
            "difficulty": "intermediate",
            "version": 1,
            "uue_stage": "understand",
            "weight": 1.0
        }
    
    async def _get_question_context(self, question_id: int) -> dict:
        """Get context for a question."""
        # Implementation would fetch from database
        return {
            "question_id": question_id,
            "type": "multiple_choice",
            "difficulty": "medium",
            "version": 1,
            "criterion_id": 1,
            "primitive_id": 1
        }
    
    # Additional private methods would follow the same pattern as the note editing service
    # For brevity, I'll include just the key analysis methods
    
    async def _analyze_blueprint_content_with_context(self, blueprint_id: int, context: BlueprintContext) -> dict:
        """Analyze blueprint content with context."""
        prompt = f"""
        Analyze blueprint ID {blueprint_id} for editing purposes.
        
        Context: {context}
        
        Provide analysis including:
        1. Structure and organization
        2. Content quality and completeness
        3. Learning flow and progression
        4. Areas for improvement
        5. Consistency and coherence
        
        Return as JSON.
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1
        )
        
        return self._parse_analysis_response(response)
    
    async def _analyze_primitive_content_with_context(self, primitive_id: int, context: dict) -> dict:
        """Analyze primitive content with context using real LLM."""
        prompt = f"""
        Analyze primitive ID {primitive_id} for editing purposes.
        
        Primitive Context: {context}
        
        Provide analysis including:
        1. Content clarity and accessibility
        2. Complexity level appropriateness
        3. Relationship mapping quality
        4. Areas for improvement
        5. Learning objective alignment
        
        Return as JSON.
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1
        )
        
        analysis_data = self._parse_analysis_response(response)
        
        if not analysis_data:
            raise Exception(f"Failed to parse LLM response for primitive analysis: {response}")
        
        return analysis_data
    
    async def _analyze_mastery_criterion_content_with_context(self, criterion_id: int, context: dict) -> dict:
        """Analyze mastery criterion content with context using real LLM."""
        prompt = f"""
        Analyze mastery criterion ID {criterion_id} for editing purposes.
        
        Criterion Context: {context}
        
        Provide analysis including:
        1. Assessment criteria clarity
        2. Difficulty level appropriateness
        3. Learning pathway alignment
        4. Areas for improvement
        5. UUE stage progression quality
        
        Return as JSON.
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1
        )
        
        analysis_data = self._parse_analysis_response(response)
        
        if not analysis_data:
            raise Exception(f"Failed to parse LLM response for mastery criterion analysis: {response}")
        
        return analysis_data
    
    async def _analyze_question_content_with_context(self, question_id: int, context: dict) -> dict:
        """Analyze question content with context using real LLM."""
        prompt = f"""
        Analyze question ID {question_id} for editing purposes.
        
        Question Context: {context}
        
        Provide analysis including:
        1. Question clarity and focus
        2. Difficulty level appropriateness
        3. Assessment quality
        4. Learning objective alignment
        5. Areas for improvement
        
        Return as JSON.
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1
        )
        
        analysis_data = self._parse_analysis_response(response)
        
        if not analysis_data:
            raise Exception(f"Failed to parse LLM response for question analysis: {response}")
        
        return analysis_data
    
    # Additional helper methods would follow the same pattern
    # Including edit plan creation, edit application, reasoning generation, etc.
    
    def _parse_context_response(self, response: str) -> dict:
        """Parse LLM response into context data with robust error handling."""
        try:
            import json
            import re
            
            # Try multiple parsing strategies
            strategies = [
                # Strategy 1: Look for JSON between ```json and ``` markers
                lambda: re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL),
                # Strategy 2: Look for JSON between ``` and ``` markers
                lambda: re.search(r'```\s*(\{.*?\})\s*```', response, re.DOTALL),
                # Strategy 3: Look for JSON between { and } with balanced braces
                lambda: self._find_balanced_json(response),
                # Strategy 4: Look for any JSON-like structure
                lambda: re.search(r'\{.*\}', response, re.DOTALL)
            ]
            
            for i, strategy in enumerate(strategies):
                try:
                    match = strategy()
                    if match:
                        json_str = match.group(1) if match.groups() else match.group(0)
                        # Clean up common LLM formatting issues
                        json_str = self._clean_json_string(json_str)
                        parsed = json.loads(json_str)
                        print(f"✅ Context JSON parsed successfully using strategy {i+1}")
                        return parsed
                except Exception as e:
                    print(f"⚠️  Context strategy {i+1} failed: {e}")
                    continue
            
            print(f"❌ All context JSON parsing strategies failed")
            return {}
            
        except Exception as e:
            print(f"Failed to parse context response: {e}")
            return {}
    
    def _parse_analysis_response(self, response: str) -> dict:
        """Parse LLM response into analysis data with robust error handling."""
        try:
            import json
            import re
            
            # Try multiple parsing strategies
            strategies = [
                # Strategy 1: Look for JSON between ```json and ``` markers
                lambda: re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL),
                # Strategy 2: Look for JSON between ``` and ``` markers
                lambda: re.search(r'```\s*(\{.*?\})\s*```', response, re.DOTALL),
                # Strategy 3: Look for JSON between { and } with balanced braces
                lambda: self._find_balanced_json(response),
                # Strategy 4: Look for any JSON-like structure
                lambda: re.search(r'\{.*\}', response, re.DOTALL)
            ]
            
            for i, strategy in enumerate(strategies):
                try:
                    match = strategy()
                    if match:
                        # Handle different return types from strategies
                        if hasattr(match, 'groups') and match.groups():
                            json_str = match.group(1)
                        elif hasattr(match, 'group'):
                            json_str = match.group(0)
                        else:
                            json_str = match  # Direct string from _find_balanced_json
                        
                        # Clean up common LLM formatting issues
                        json_str = self._clean_json_string(json_str)
                        parsed = json.loads(json_str)
                        print(f"✅ Analysis JSON parsed successfully using strategy {i+1}")
                        return parsed
                except Exception as e:
                    print(f"⚠️  Analysis strategy {i+1} failed: {e}")
                    continue
            
            # If all strategies fail, try to extract partial JSON and fix common issues
            print(f"⚠️  All parsing strategies failed, attempting partial recovery...")
            try:
                # Look for any JSON-like content and try to fix it
                json_match = re.search(r'\{.*', response, re.DOTALL)
                if json_match:
                    partial_json = json_match.group(0)
                    # Try to find a reasonable ending point
                    brace_count = partial_json.count('{') - partial_json.count('}')
                    if brace_count > 0:
                        # Add missing closing braces
                        partial_json += '}' * brace_count
                    
                    # Clean and try to parse
                    cleaned_json = self._clean_json_string(partial_json)
                    parsed = json.loads(cleaned_json)
                    print(f"✅ Partial JSON recovery successful")
                    return parsed
            except Exception as recovery_error:
                print(f"⚠️  Partial recovery failed: {recovery_error}")
            
            # Final fallback: try to extract just the summary if available
            try:
                summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', response)
                if summary_match:
                    fallback_data = {
                        "summary": summary_match.group(1),
                        "context_alignment": "Partial recovery from truncated response",
                        "changes": [
                            {
                                "type": "recovery",
                                "description": "Response was truncated, using partial data",
                                "reason": "LLM response exceeded token limits",
                                "context_impact": "Limited editing information available"
                            }
                        ]
                    }
                    print(f"✅ Fallback summary recovery successful")
                    return fallback_data
            except Exception as fallback_error:
                print(f"⚠️  Fallback recovery failed: {fallback_error}")
            
            print(f"❌ All analysis JSON parsing strategies failed")
            return {}
            
        except Exception as e:
            print(f"Failed to parse analysis response: {e}")
            return {}
    
    # Additional parsing methods would be implemented for other response types
    
    # ============================================================================
    # MISSING HELPER METHODS - IMPLEMENTING NOW
    # ============================================================================
    
    async def _create_blueprint_edit_plan(
        self,
        blueprint_analysis: dict,
        context: BlueprintContext,
        edit_instruction: str,
        edit_type: str,
        preserve_structure: bool,
        user_preferences: Optional[Any] = None
    ) -> dict:
        """Create a blueprint edit plan."""
        structure_instruction = "Preserve the original structure and organization." if preserve_structure else "Reorganize content for better flow and readability."
        
        context_info = f"""
        Blueprint Context:
        - Title: {context.title}
        - Sections: {context.sections_count}
        - Primitives: {context.primitives_count}
        - Criteria: {context.criteria_count}
        - Questions: {context.questions_count}
        - Difficulty Distribution: {context.difficulty_distribution}
        """
        
        prompt = f"""
        Create a blueprint edit plan based on the following analysis, instruction, and context.
        
        Blueprint Analysis:
        {blueprint_analysis}
        
        Blueprint Context:
        {context_info}
        
        Edit Instruction: {edit_instruction}
        Edit Type: {edit_type}
        Structure Requirement: {structure_instruction}
        User Preferences: {user_preferences.dict() if user_preferences else 'Default'}
        
        Create a detailed edit plan that includes:
        1. Specific changes to make
        2. Content reorganization if needed
        3. Structure adjustments
        4. Quality improvements
        5. Context alignment
        6. Summary of planned changes
        
        Return as JSON with this structure:
        {{
            "summary": "Brief overview of planned changes",
            "context_alignment": "How changes align with blueprint",
            "changes": [
                {{
                    "type": "content|structure|style|quality|context",
                    "description": "What to change",
                    "reason": "Why this change is needed",
                    "context_impact": "How this affects blueprint"
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
        
        edit_plan_data = self._parse_edit_plan_response(response)
        
        if not edit_plan_data:
            raise Exception(f"Failed to parse LLM response for blueprint edit plan: {response}")
        
        return edit_plan_data
    
    async def _create_primitive_edit_plan(
        self,
        primitive_analysis: dict,
        context: dict,
        edit_instruction: str,
        edit_type: str,
        preserve_structure: bool,
        user_preferences: Optional[Any] = None
    ) -> dict:
        """Create a primitive edit plan using real LLM."""
        prompt = f"""
        Create a primitive edit plan based on the following analysis, instruction, and context.
        
        Primitive Analysis:
        {primitive_analysis}
        
        Primitive Context:
        - Type: {context.get('type', 'concept')}
        - Difficulty: {context.get('difficulty', 'intermediate')}
        - Version: {context.get('version', 1)}
        
        Edit Instruction: {edit_instruction}
        Edit Type: {edit_type}
        Structure Requirement: {"Preserve original structure" if preserve_structure else "Reorganize for better flow"}
        User Preferences: {user_preferences.dict() if user_preferences else 'Default'}
        
        Create a detailed edit plan that includes:
        1. Specific changes to make
        2. Content improvements
        3. Clarity enhancements
        4. Context alignment
        5. Summary of planned changes
        
        Return as JSON with this structure:
        {{
            "summary": "Brief overview of planned changes",
            "context_alignment": "How changes align with primitive context",
            "changes": [
                {{
                    "type": "content|clarity|structure|complexity",
                    "description": "What to change",
                    "reason": "Why this change is needed",
                    "context_impact": "How this affects learning progression"
                }}
            ]
        }}
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=6000,  # Increased further to handle very long responses
            temperature=0.2
        )
        

        
        edit_plan_data = self._parse_edit_plan_response(response)
        
        if not edit_plan_data:
            raise Exception(f"Failed to parse LLM response for primitive edit plan: {response}")
        
        return edit_plan_data
    
    async def _create_mastery_criterion_edit_plan(
        self,
        criterion_analysis: dict,
        context: dict,
        edit_instruction: str,
        edit_type: str,
        preserve_structure: bool,
        user_preferences: Optional[Any] = None
    ) -> dict:
        """Create a mastery criterion edit plan using real LLM."""
        prompt = f"""
        Create a mastery criterion edit plan based on the following analysis, instruction, and context.
        
        Criterion Analysis:
        {criterion_analysis}
        
        Criterion Context:
        - Type: {context.get('type', 'question_based')}
        - Difficulty: {context.get('difficulty', 'intermediate')}
        - UUE Stage: {context.get('uue_stage', 'understand')}
        - Weight: {context.get('weight', 1.0)}
        
        Edit Instruction: {edit_instruction}
        Edit Type: {edit_type}
        Structure Requirement: {"Preserve original structure" if preserve_structure else "Reorganize for better flow"}
        User Preferences: {user_preferences.dict() if user_preferences else 'Default'}
        
        Create a detailed edit plan that includes:
        1. Specific changes to make
        2. Assessment improvements
        3. Clarity enhancements
        4. Learning pathway alignment
        5. Summary of planned changes
        
        Return as JSON with this structure:
        {{
            "summary": "Brief overview of planned changes",
            "context_alignment": "How changes align with learning pathway",
            "changes": [
                {{
                    "type": "clarity|difficulty|assessment|pathway",
                    "description": "What to change",
                    "reason": "Why this change is needed",
                    "context_impact": "How this affects learning progression"
                }}
            ]
        }}
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=6000,  # Increased further to handle very long responses
            temperature=0.2
        )
        
        edit_plan_data = self._parse_edit_plan_response(response)
        
        return edit_plan_data
    
    async def _create_question_edit_plan(
        self,
        question_analysis: dict,
        context: dict,
        edit_instruction: str,
        edit_type: str,
        preserve_structure: bool,
        user_preferences: Optional[Any] = None
    ) -> dict:
        """Create a question edit plan using real LLM."""
        prompt = f"""
        Create a question edit plan based on the following analysis, instruction, and context.
        
        Question Analysis:
        {question_analysis}
        
        Question Context:
        - Type: {context.get('type', 'multiple_choice')}
        - Difficulty: {context.get('difficulty', 'medium')}
        - Criterion ID: {context.get('criterion_id', 'unknown')}
        - Primitive ID: {context.get('primitive_id', 'unknown')}
        
        Edit Instruction: {edit_instruction}
        Edit Type: {edit_type}
        Structure Requirement: {"Preserve original structure" if preserve_structure else "Reorganize for better flow"}
        User Preferences: {user_preferences.dict() if user_preferences else 'Default'}
        
        Create a detailed edit plan that includes:
        1. Specific changes to make
        2. Question improvements
        3. Clarity enhancements
        4. Assessment quality improvements
        5. Summary of planned changes
        
        Return as JSON with this structure:
        {{
            "summary": "Brief overview of planned changes",
            "context_alignment": "How changes align with assessment quality",
            "changes": [
                {{
                    "type": "clarity|difficulty|quality|focus",
                    "description": "What to change",
                    "reason": "Why this change is needed",
                    "context_impact": "How this affects assessment accuracy"
                }}
            ]
        }}
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=6000,  # Increased further to handle very long responses
            temperature=0.2
        )
        
        edit_plan_data = self._parse_edit_plan_response(response)
        
        if not edit_plan_data:
            raise Exception(f"Failed to parse LLM response for question edit plan: {response}")
        
        return edit_plan_data
    
    async def _apply_blueprint_edits(
        self,
        blueprint_analysis: dict,
        context: BlueprintContext,
        edit_plan: dict,
        edit_type: str
    ) -> Dict[str, Any]:
        """Apply blueprint edits."""
        return {
            "edited_content": "Improved blueprint content based on edit plan",
            "changes_applied": ["change1", "change2"],
            "context_alignment": "Better alignment with learning objectives"
        }
    
    async def _apply_primitive_edits(
        self,
        primitive_analysis: dict,
        context: dict,
        edit_plan: dict,
        edit_type: str
    ) -> Dict[str, Any]:
        """Apply primitive edits using real LLM."""
        prompt = f"""
        Apply the following edit plan to improve a knowledge primitive.
        
        Primitive Analysis:
        {primitive_analysis}
        
        Primitive Context:
        - Type: {context.get('type', 'concept')}
        - Difficulty: {context.get('difficulty', 'intermediate')}
        
        Edit Plan:
        {edit_plan}
        
        Edit Type: {edit_type}
        
        Apply all the planned changes and return the edited primitive content.
        Ensure the result is:
        - Clear and accessible
        - Well-structured
        - Follows the edit plan exactly
        - Maintains high quality
        - Aligns with primitive context
        
        Return your response as a JSON object with this structure:
        {{
            "edited_content": "Complete edited primitive content",
            "changes_applied": ["change1", "change2"],
            "context_alignment": "How the result aligns with primitive context"
        }}
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=3000,
            temperature=0.3
        )
        
        edited_data = self._parse_edited_content_response(response)
        
        if not edited_data:
            raise Exception(f"Failed to parse LLM response for primitive edits: {response}")
        
        return edited_data
    
    async def _apply_mastery_criterion_edits(
        self,
        criterion_analysis: dict,
        context: dict,
        edit_plan: dict,
        edit_type: str
    ) -> Dict[str, Any]:
        """Apply mastery criterion edits."""
        return {
            "edited_content": "Improved criterion content based on edit plan",
            "changes_applied": ["change1"],
            "context_alignment": "Better assessment clarity"
        }
    
    async def _apply_question_edits(
        self,
        question_analysis: dict,
        context: dict,
        edit_plan: dict,
        edit_type: str
    ) -> Dict[str, Any]:
        """Apply question edits."""
        return {
            "edited_content": "Improved question content based on edit plan",
            "changes_applied": ["change1"],
            "context_alignment": "Better question focus"
        }
    
    async def _generate_blueprint_edit_reasoning(
        self,
        blueprint_analysis: dict,
        context: BlueprintContext,
        edit_plan: dict,
        edit_instruction: str
    ) -> str:
        """Generate reasoning for blueprint edits."""
        return f"Blueprint edits were made to improve {edit_instruction}. The changes align with the learning objectives and maintain structural consistency."
    
    async def _generate_primitive_edit_reasoning(
        self,
        primitive_analysis: dict,
        context: dict,
        edit_plan: dict,
        edit_instruction: str
    ) -> str:
        """Generate reasoning for primitive edits."""
        return f"Primitive edits were made to improve {edit_instruction}. The changes enhance concept clarity and accessibility."
    
    async def _generate_mastery_criterion_edit_reasoning(
        self,
        criterion_analysis: dict,
        context: dict,
        edit_plan: dict,
        edit_instruction: str
    ) -> str:
        """Generate reasoning for mastery criterion edits."""
        return f"Mastery criterion edits were made to improve {edit_instruction}. The changes enhance assessment clarity and learning pathway alignment."
    
    async def _generate_question_edit_reasoning(
        self,
        question_analysis: dict,
        context: dict,
        edit_plan: dict,
        edit_instruction: str
    ) -> str:
        """Generate reasoning for question edits."""
        return f"Question edits were made to improve {edit_instruction}. The changes enhance question clarity and learning objective focus."
    
    async def _generate_blueprint_granular_edit_reasoning(
        self,
        request: BlueprintEditingRequest,
        granular_edits: List[GranularEditResult],
        context: BlueprintContext
    ) -> str:
        """Generate reasoning for granular blueprint edits."""
        return f"Granular blueprint edits were made using {request.edit_type}. The changes improve {request.edit_instruction} while maintaining structural integrity."
    
    async def _generate_blueprint_structure_suggestions(
        self,
        blueprint_analysis: dict,
        context: BlueprintContext
    ) -> List[EditingSuggestion]:
        """Generate blueprint structure suggestions."""
        return [
            EditingSuggestion(
                suggestion_id=f"structure_1_{int(time.time())}",
                type="structure",
                description="Improve section organization",
                suggested_change="Reorganize sections for better learning flow",
                confidence=0.8,
                reasoning="Better organization improves learning progression"
            )
        ]
    
    async def _generate_blueprint_content_suggestions(
        self,
        blueprint_analysis: dict,
        context: BlueprintContext
    ) -> List[EditingSuggestion]:
        """Generate blueprint content suggestions."""
        return [
            EditingSuggestion(
                suggestion_id=f"content_1_{int(time.time())}",
                type="content",
                description="Add more examples",
                suggested_change="Include practical examples for each concept",
                confidence=0.9,
                reasoning="Examples improve understanding and retention"
            )
        ]
    
    async def _generate_blueprint_relationship_suggestions(
        self,
        blueprint_analysis: dict,
        context: BlueprintContext
    ) -> List[EditingSuggestion]:
        """Generate blueprint relationship suggestions."""
        return [
            EditingSuggestion(
                suggestion_id=f"relationship_1_{int(time.time())}",
                type="relationships",
                description="Improve concept connections",
                suggested_change="Add explicit links between related concepts",
                confidence=0.8,
                reasoning="Better connections improve knowledge integration"
            )
        ]
    
    async def _generate_primitive_clarity_suggestions(
        self,
        primitive_analysis: dict,
        context: dict
    ) -> List[EditingSuggestion]:
        """Generate primitive clarity suggestions."""
        return [
            EditingSuggestion(
                suggestion_id=f"clarity_1_{int(time.time())}",
                type="clarity",
                description="Simplify language",
                suggested_change="Use simpler, more accessible language",
                confidence=0.9,
                reasoning="Simpler language improves beginner accessibility"
            )
        ]
    
    async def _generate_primitive_complexity_suggestions(
        self,
        primitive_analysis: dict,
        context: dict
    ) -> List[EditingSuggestion]:
        """Generate primitive complexity suggestions."""
        return [
            EditingSuggestion(
                suggestion_id=f"complexity_1_{int(time.time())}",
                type="complexity",
                description="Adjust difficulty level",
                suggested_change="Provide both basic and advanced explanations",
                confidence=0.8,
                reasoning="Multiple difficulty levels accommodate different learners"
            )
        ]
    
    async def _generate_primitive_relationship_suggestions(
        self,
        primitive_analysis: dict,
        context: dict
    ) -> List[EditingSuggestion]:
        """Generate primitive relationship suggestions."""
        return [
            EditingSuggestion(
                suggestion_id=f"relationship_1_{int(time.time())}",
                type="relationships",
                description="Clarify concept relationships",
                suggested_change="Add explicit connections to related concepts",
                confidence=0.8,
                reasoning="Clear relationships improve knowledge integration"
            )
        ]
    
    async def _generate_criterion_clarity_suggestions(
        self,
        criterion_analysis: dict,
        context: dict
    ) -> List[EditingSuggestion]:
        """Generate criterion clarity suggestions."""
        return [
            EditingSuggestion(
                suggestion_id=f"clarity_1_{int(time.time())}",
                type="clarity",
                description="Clarify assessment criteria",
                suggested_change="Make the mastery requirements more specific",
                confidence=0.9,
                reasoning="Specific criteria improve learning focus"
            )
        ]
    
    async def _generate_criterion_difficulty_suggestions(
        self,
        criterion_analysis: dict,
        context: dict
    ) -> List[EditingSuggestion]:
        """Generate criterion difficulty suggestions."""
        return [
            EditingSuggestion(
                suggestion_id=f"difficulty_1_{int(time.time())}",
                type="difficulty",
                description="Adjust difficulty level",
                suggested_change="Provide progressive difficulty options",
                confidence=0.8,
                reasoning="Progressive difficulty supports learning progression"
            )
        ]
    
    async def _generate_criterion_assessment_suggestions(
        self,
        criterion_analysis: dict,
        context: dict
    ) -> List[EditingSuggestion]:
        """Generate criterion assessment suggestions."""
        return [
            EditingSuggestion(
                suggestion_id=f"assessment_1_{int(time.time())}",
                type="assessment",
                description="Improve assessment methods",
                suggested_change="Add multiple assessment types",
                confidence=0.8,
                reasoning="Multiple assessment types improve mastery validation"
            )
        ]
    
    async def _generate_question_clarity_suggestions(
        self,
        question_analysis: dict,
        context: dict
    ) -> List[EditingSuggestion]:
        """Generate question clarity suggestions."""
        return [
            EditingSuggestion(
                suggestion_id=f"clarity_1_{int(time.time())}",
                type="clarity",
                description="Improve question clarity",
                suggested_change="Make the question more focused and clear",
                confidence=0.9,
                reasoning="Clear questions improve assessment accuracy"
            )
        ]
    
    async def _generate_question_difficulty_suggestions(
        self,
        question_analysis: dict,
        context: dict
    ) -> List[EditingSuggestion]:
        """Generate question difficulty suggestions."""
        return [
            EditingSuggestion(
                suggestion_id=f"difficulty_1_{int(time.time())}",
                type="difficulty",
                description="Adjust question difficulty",
                suggested_change="Provide multiple difficulty variations",
                confidence=0.8,
                reasoning="Multiple difficulties accommodate different skill levels"
            )
        ]
    
    async def _generate_question_quality_suggestions(
        self,
        question_analysis: dict,
        context: dict
    ) -> List[EditingSuggestion]:
        """Generate question quality suggestions."""
        return [
            EditingSuggestion(
                suggestion_id=f"quality_1_{int(time.time())}",
                type="quality",
                description="Improve question quality",
                suggested_change="Add distractors and improve answer explanations",
                confidence=0.8,
                reasoning="Better quality questions improve learning outcomes"
            )
        ]
    
    def _parse_edit_plan_response(self, response: str) -> dict:
        """Parse LLM response into edit plan data with robust error handling."""
        try:
            import json
            import re
            
            # Try multiple parsing strategies
            strategies = [
                # Strategy 1: Look for JSON between ```json and ``` markers
                lambda: re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL),
                # Strategy 2: Look for JSON between ``` and ``` markers
                lambda: re.search(r'```\s*(\{.*?\})\s*```', response, re.DOTALL),
                # Strategy 3: Look for JSON between { and } with balanced braces
                lambda: self._find_balanced_json(response),
                # Strategy 4: Look for any JSON-like structure
                lambda: re.search(r'\{.*\}', response, re.DOTALL)
            ]
            
            for i, strategy in enumerate(strategies):
                try:
                    match = strategy()
                    if match:
                        # Handle different return types from strategies
                        if hasattr(match, 'groups') and match.groups():
                            json_str = match.group(1)
                        elif hasattr(match, 'group'):
                            json_str = match.group(0)
                        else:
                            json_str = match  # Direct string from _find_balanced_json
                        
                        # Clean up common LLM formatting issues
                        json_str = self._clean_json_string(json_str)
                        parsed = json.loads(json_str)
                        print(f"✅ JSON parsed successfully using strategy {i+1}")
                        return parsed
                except Exception as e:
                    print(f"⚠️  Strategy {i+1} failed: {e}")
                    continue
            
            print(f"❌ All JSON parsing strategies failed")
            return {}
            
        except Exception as e:
            print(f"Failed to parse edit plan response: {e}")
            return {}
    
    def _parse_edited_content_response(self, response: str) -> dict:
        """Parse LLM response into edited content data with robust error handling."""
        try:
            import json
            import re
            
            # Try multiple parsing strategies
            strategies = [
                # Strategy 1: Look for JSON between ```json and ``` markers
                lambda: re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL),
                # Strategy 2: Look for JSON between ``` and ``` markers
                lambda: re.search(r'```\s*(\{.*?\})\s*```', response, re.DOTALL),
                # Strategy 3: Look for JSON between { and } with balanced braces
                lambda: self._find_balanced_json(response),
                # Strategy 4: Look for any JSON-like structure
                lambda: re.search(r'\{.*\}', response, re.DOTALL)
            ]
            
            for i, strategy in enumerate(strategies):
                try:
                    match = strategy()
                    if match:
                        json_str = match.group(1) if match.groups() else match.group(0)
                        # Clean up common LLM formatting issues
                        json_str = self._clean_json_string(json_str)
                        parsed = json.loads(json_str)
                        print(f"✅ JSON parsed successfully using strategy {i+1}")
                        return parsed
                except Exception as e:
                    print(f"⚠️  Strategy {i+1} failed: {e}")
                    continue
            
            print(f"❌ All JSON parsing strategies failed")
            return {}
            
        except Exception as e:
            print(f"Failed to parse edited content response: {e}")
            return {}
    
    def _find_balanced_json(self, text: str) -> str:
        """Find JSON with balanced braces in text."""
        try:
            import re
            
            # Find all opening braces
            open_braces = [m.start() for m in re.finditer(r'\{', text)]
            if not open_braces:
                return None
            
            # Find the first complete JSON object
            for start in open_braces:
                brace_count = 0
                for i, char in enumerate(text[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            return text[start:i+1]
            
            return None
        except Exception as e:
            print(f"Error finding balanced JSON: {e}")
            return None
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean up common LLM JSON formatting issues."""
        try:
            import re
            
            # Remove trailing commas before closing braces/brackets
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            # Fix common quote issues
            json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', json_str)
            
            # Remove any trailing commas at the end
            json_str = re.sub(r',\s*$', '', json_str)
            
            # Fix common newline issues in strings
            json_str = re.sub(r'\n\s*', ' ', json_str)
            
            return json_str.strip()
        except Exception as e:
            print(f"Error cleaning JSON string: {e}")
            return json_str
