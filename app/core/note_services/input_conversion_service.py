"""
Input Conversion Service for converting user input to BlockNote format.
Creates blueprints from user content for structured note generation.
"""

import time
from typing import Optional
from app.models.note_creation_models import (
    ContentToNoteRequest, InputConversionRequest, ContentConversionResponse,
    ContentFormat, BlueprintCreationResult
)
from app.services.llm_service import LLMService


class InputConversionService:
    """Service for converting user input to structured notes."""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def convert_content_to_notes(
        self, 
        request: ContentToNoteRequest
    ) -> ContentConversionResponse:
        """
        Convert user content to structured notes via blueprint creation.
        
        Args:
            request: Content conversion request with user input
            
        Returns:
            ContentConversionResponse with converted notes and blueprint
        """
        start_time = time.time()
        
        try:
            # Step 1: Create blueprint from user content
            blueprint_result = await self._create_blueprint_from_content(
                request.user_content,
                request.content_format,
                request.user_preferences
            )
            
            if not blueprint_result.success:
                return ContentConversionResponse(
                    success=False,
                    message=f"Failed to create blueprint: {blueprint_result.message}"
                )
            
            # Step 2: Generate structured notes from blueprint
            note_content, plain_text = await self._generate_structured_notes(
                blueprint_result,
                request.note_style,
                request.user_preferences
            )
            
            processing_time = time.time() - start_time
            
            return ContentConversionResponse(
                success=True,
                converted_content=note_content,
                plain_text=plain_text,
                blueprint_id=blueprint_result.blueprint_id,
                conversion_notes=f"Converted from {request.content_format} format",
                message=f"Successfully converted {request.content_format} content to structured notes"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ContentConversionResponse(
                success=False,
                processing_time=processing_time,
                message=f"Error converting content: {str(e)}"
            )
    
    async def convert_input_to_blocknote(
        self, 
        request: InputConversionRequest
    ) -> ContentConversionResponse:
        """
        Convert input content directly to BlockNote format.
        
        Args:
            request: Input conversion request
            
        Returns:
            ContentConversionResponse with BlockNote content
        """
        start_time = time.time()
        
        try:
            # Create conversion prompt based on input format
            prompt = self._create_conversion_prompt(
                request.input_content,
                request.input_format,
                request.preserve_structure,
                request.include_metadata
            )
            
            # Call LLM for conversion
            response = await self.llm_service.call_llm(
                prompt=prompt,
                max_tokens=3000,
                temperature=0.1
            )
            
            # Parse conversion response
            converted_data = self._parse_conversion_response(response)
            
            note_content = converted_data.get('blocknote_content', '')
            plain_text = converted_data.get('plain_text', '')
            
            processing_time = time.time() - start_time
            
            return ContentConversionResponse(
                success=True,
                converted_content=note_content,
                plain_text=plain_text,
                blueprint_id=None,  # No blueprint for direct conversion
                conversion_notes=f"Direct conversion from {request.input_format}",
                message=f"Successfully converted {request.input_format} to BlockNote format"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ContentConversionResponse(
                success=False,
                processing_time=processing_time,
                message=f"Error converting input: {str(e)}"
            )
    
    async def _create_blueprint_from_content(
        self,
        user_content: str,
        content_format: ContentFormat,
        user_preferences
    ) -> BlueprintCreationResult:
        """Create a learning blueprint from user content."""
        start_time = time.time()
        
        try:
            # Create blueprint creation prompt
            prompt = self._create_content_blueprint_prompt(user_content, content_format, user_preferences)
            
            # Call LLM to create blueprint
            response = await self.llm_service.call_llm(
                prompt=prompt,
                max_tokens=2500,
                temperature=0.1
            )
            
            # Parse blueprint response
            blueprint_data = self._parse_blueprint_response(response)
            
            processing_time = time.time() - start_time
            
            return BlueprintCreationResult(
                success=True,
                blueprint_id=blueprint_data.get('blueprint_id', 'bp_' + str(int(time.time()))),
                blueprint_summary=blueprint_data.get('summary', 'Generated from user content'),
                knowledge_primitives=blueprint_data.get('knowledge_primitives', []),
                cross_references=blueprint_data.get('cross_references', []),
                processing_time=processing_time,
                message="Blueprint created successfully from user content"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return BlueprintCreationResult(
                success=False,
                blueprint_id="",
                blueprint_summary="",
                knowledge_primitives=[],
                cross_references=[],
                processing_time=processing_time,
                message=f"Failed to create blueprint: {str(e)}"
            )
    
    def _create_content_blueprint_prompt(
        self,
        user_content: str,
        content_format: ContentFormat,
        user_preferences
    ) -> str:
        """Create prompt for blueprint creation from user content."""
        
        # Truncate content if too long
        content_preview = user_content[:3000] + "..." if len(user_content) > 3000 else user_content
        
        prompt = f"""
        You are an expert at creating Learning Blueprints from user content.
        
        Create a comprehensive Learning Blueprint from this user content:
        
        Format: {content_format}
        Content:
        {content_preview}
        
        User preferences:
        - Style: {user_preferences.preferred_style}
        - Include examples: {user_preferences.include_examples}
        - Include definitions: {user_preferences.include_definitions}
        - Focus on key concepts: {user_preferences.focus_on_key_concepts}
        
        Extract the core knowledge and create a structured blueprint that can be used for:
        1. Note generation
        2. RAG operations
        3. Knowledge organization
        
        Return your response as a JSON object with this structure:
        {{
            "blueprint_id": "bp_timestamp",
            "summary": "Brief overview of the knowledge domain",
            "knowledge_primitives": [
                "key concept 1",
                "key concept 2",
                "key concept 3"
            ],
            "cross_references": [
                "related topic 1",
                "related topic 2"
            ],
            "structure": {{
                "main_topics": ["topic1", "topic2"],
                "relationships": ["topic1 -> topic2"],
                "prerequisites": ["basic knowledge needed"]
            }}
        }}
        
        Focus on understanding the user's intent and extracting meaningful knowledge structure.
        """
        return prompt
    
    def _parse_blueprint_response(self, response: str) -> dict:
        """Parse LLM response into blueprint data."""
        try:
            import json
            import re
            
            # Find JSON object in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return {}
            
            return json.loads(json_match.group())
            
        except Exception as e:
            print(f"Failed to parse blueprint response: {e}")
            return {}
    
    async def _generate_structured_notes(
        self,
        blueprint_result: BlueprintCreationResult,
        note_style,
        user_preferences
    ) -> tuple[str, str]:
        """Generate structured notes from the created blueprint."""
        
        # Create note generation prompt
        prompt = self._create_structured_notes_prompt(
            blueprint_result,
            note_style,
            user_preferences
        )
        
        # Call LLM to generate notes
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=3500,
            temperature=0.3
        )
        
        # Parse note response
        note_data = self._parse_note_response(response)
        
        note_content = note_data.get('note_content', '')
        plain_text = note_data.get('plain_text', note_content)
        
        return note_content, plain_text
    
    def _create_structured_notes_prompt(
        self,
        blueprint_result: BlueprintCreationResult,
        note_style,
        user_preferences
    ) -> str:
        """Create prompt for structured note generation."""
        
        prompt = f"""
        You are an expert at creating structured notes from Learning Blueprints.
        
        Create well-structured notes based on this Learning Blueprint:
        
        Summary: {blueprint_result.blueprint_summary}
        
        Knowledge Primitives:
        {chr(10).join(f"- {primitive}" for primitive in blueprint_result.knowledge_primitives)}
        
        Cross References:
        {chr(10).join(f"- {ref}" for ref in blueprint_result.cross_references)}
        
        Note Style: {note_style}
        
        User Preferences:
        - Include examples: {user_preferences.include_examples}
        - Include definitions: {user_preferences.include_definitions}
        - Focus on key concepts: {user_preferences.focus_on_key_concepts}
        - Max length: {user_preferences.max_note_length or 'No limit'}
        
        Return your response as a JSON object with this structure:
        {{
            "note_content": "BlockNote JSON format content with proper structure",
            "plain_text": "Plain text version of the structured note",
            "structure": {{
                "main_sections": ["section1", "section2"],
                "key_points": ["point1", "point2"],
                "examples": ["example1", "example2"],
                "definitions": ["def1", "def2"]
            }}
        }}
        
        The note should be:
        - Well-organized with clear sections
        - Follow the specified note style
        - Include relevant examples and definitions based on preferences
        - Use proper BlockNote format for rich content structure
        - Maintain logical flow and readability
        """
        return prompt
    
    def _create_conversion_prompt(
        self,
        input_content: str,
        input_format: ContentFormat,
        preserve_structure: bool,
        include_metadata: bool
    ) -> str:
        """Create prompt for direct format conversion."""
        
        structure_instruction = "Preserve the original document structure and organization." if preserve_structure else "Reorganize content for better flow and readability."
        metadata_instruction = "Include content metadata like headings, lists, and formatting." if include_metadata else "Focus on content conversion without extra metadata."
        
        prompt = f"""
        You are an expert at converting content between different formats.
        
        Convert the following {input_format} content to BlockNote format:
        
        {input_content[:2500]}...
        
        Requirements:
        - {structure_instruction}
        - {metadata_instruction}
        - Maintain semantic meaning and readability
        - Use proper BlockNote JSON structure
        
        Return your response as a JSON object with this structure:
        {{
            "blocknote_content": "Complete BlockNote JSON format content",
            "plain_text": "Plain text version of the converted content",
            "conversion_notes": "Brief notes about the conversion process"
        }}
        
        Ensure the BlockNote format is valid and follows the proper structure for rich text content.
        """
        return prompt
    
    def _parse_conversion_response(self, response: str) -> dict:
        """Parse LLM response into conversion data."""
        try:
            import json
            import re
            
            # Find JSON object in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return {}
            
            return json.loads(json_match.group())
            
        except Exception as e:
            print(f"Failed to parse conversion response: {e}")
            return {}
    
    def _parse_note_response(self, response: str) -> dict:
        """Parse LLM response into note data."""
        try:
            import json
            import re
            
            # Find JSON object in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return {}
            
            return json.loads(json_match.group())
            
        except Exception as e:
            print(f"Failed to parse note response: {e}")
            return {}
