"""
Note Generation Service for creating notes from source content.
Creates notes via blueprint creation for RAG context.
"""

import time
from typing import List, Optional
from app.models.note_creation_models import (
    NoteGenerationRequest, NoteGenerationResponse, SourceChunk,
    ChunkingStrategy, BlueprintCreationResult
)
from app.core.note_services.source_chunking_service import SourceChunkingService
from app.services.llm_service import LLMService


class NoteGenerationService:
    """Service for generating notes from source content."""
    
    def __init__(self, llm_service: LLMService, chunking_service: SourceChunkingService):
        self.llm_service = llm_service
        self.chunking_service = chunking_service
    
    async def generate_notes_from_source(
        self, 
        request: NoteGenerationRequest
    ) -> NoteGenerationResponse:
        """
        Generate notes from source content via blueprint creation.
        
        Args:
            request: Note generation request with source and preferences
            
        Returns:
            NoteGenerationResponse with generated notes and blueprint info
        """
        start_time = time.time()
        
        try:
            # Step 1: Chunk source content if needed
            chunking_strategy = request.chunking_strategy or ChunkingStrategy()
            chunking_result = await self.chunking_service.chunk_source_content(
                request.source_content, 
                chunking_strategy
            )
            
            if not chunking_result.success:
                return NoteGenerationResponse(
                    success=False,
                    message=f"Failed to chunk source content: {chunking_result.message}"
                )
            
            # Step 2: Create blueprint from chunks
            blueprint_result = await self._create_blueprint_from_chunks(
                chunking_result.chunks,
                request.user_preferences
            )
            
            if not blueprint_result.success:
                return NoteGenerationResponse(
                    success=False,
                    message=f"Failed to create blueprint: {blueprint_result.message}"
                )
            
            # Step 3: Generate notes from blueprint
            note_content, plain_text = await self._generate_notes_from_blueprint(
                blueprint_result,
                request.note_style,
                request.user_preferences,
                request.target_length,
                request.focus_areas
            )
            
            processing_time = time.time() - start_time
            
            return NoteGenerationResponse(
                success=True,
                note_content=note_content,
                plain_text=plain_text,
                blueprint_id=blueprint_result.blueprint_id,
                chunks_processed=chunking_result.chunks,
                processing_time=processing_time,
                message=f"Successfully generated notes from {len(chunking_result.chunks)} source chunks"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return NoteGenerationResponse(
                success=False,
                processing_time=processing_time,
                message=f"Error generating notes: {str(e)}"
            )
    
    async def _create_blueprint_from_chunks(
        self, 
        chunks: List[SourceChunk], 
        user_preferences
    ) -> BlueprintCreationResult:
        """Create a learning blueprint from source chunks."""
        start_time = time.time()
        
        try:
            # Combine chunk content for blueprint creation
            combined_content = self._combine_chunks_for_blueprint(chunks)
            
            # Create blueprint creation prompt
            prompt = self._create_blueprint_prompt(combined_content, chunks, user_preferences)
            
            # Call LLM to create blueprint
            response = await self.llm_service.call_llm(
                prompt=prompt,
                max_tokens=3000,
                temperature=0.1
            )
            
            # Parse blueprint response
            blueprint_data = self._parse_blueprint_response(response)
            
            processing_time = time.time() - start_time
            
            return BlueprintCreationResult(
                success=True,
                blueprint_id=blueprint_data.get('blueprint_id', 'bp_' + str(int(time.time()))),
                blueprint_summary=blueprint_data.get('summary', 'Generated blueprint'),
                knowledge_primitives=blueprint_data.get('knowledge_primitives', []),
                cross_references=blueprint_data.get('cross_references', []),
                processing_time=processing_time,
                message="Blueprint created successfully from source chunks"
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
    
    def _combine_chunks_for_blueprint(self, chunks: List[SourceChunk]) -> str:
        """Combine chunk content for blueprint creation."""
        combined = []
        
        for chunk in chunks:
            combined.append(f"=== {chunk.topic} ===\n{chunk.content}\n")
        
        return '\n'.join(combined)
    
    def _create_blueprint_prompt(
        self, 
        combined_content: str, 
        chunks: List[SourceChunk],
        user_preferences
    ) -> str:
        """Create prompt for blueprint creation."""
        prompt = f"""
        You are an expert at creating Learning Blueprints from source content.
        
        Create a comprehensive Learning Blueprint from the following source content:
        
        {combined_content[:5000]}...
        
        The content has been chunked into {len(chunks)} sections:
        {[chunk.topic for chunk in chunks]}
        
        User preferences:
        - Style: {user_preferences.preferred_style}
        - Include examples: {user_preferences.include_examples}
        - Include definitions: {user_preferences.include_definitions}
        - Focus on key concepts: {user_preferences.focus_on_key_concepts}
        
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
        
        Focus on extracting the core knowledge primitives and creating a clear structure
        that can be used for note generation and RAG operations.
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
    
    async def _generate_notes_from_blueprint(
        self,
        blueprint_result: BlueprintCreationResult,
        note_style,
        user_preferences,
        target_length: Optional[int],
        focus_areas: List[str]
    ) -> tuple[str, str]:
        """Generate notes from the created blueprint."""
        
        # Create note generation prompt
        prompt = self._create_note_generation_prompt(
            blueprint_result,
            note_style,
            user_preferences,
            target_length,
            focus_areas
        )
        
        # Call LLM to generate notes
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=4000,
            temperature=0.3
        )
        
        # Parse note response
        note_data = self._parse_note_response(response)
        
        note_content = note_data.get('note_content', '')
        plain_text = note_data.get('plain_text', note_content)
        
        return note_content, plain_text
    
    def _create_note_generation_prompt(
        self,
        blueprint_result: BlueprintCreationResult,
        note_style,
        user_preferences,
        target_length: Optional[int],
        focus_areas: List[str]
    ) -> str:
        """Create prompt for note generation from blueprint."""
        
        length_instruction = ""
        if target_length:
            length_instruction = f"Target length: approximately {target_length} words."
        
        focus_instruction = ""
        if focus_areas:
            focus_instruction = f"Focus areas: {', '.join(focus_areas)}"
        
        prompt = f"""
        You are an expert at creating structured notes from Learning Blueprints.
        
        Create notes based on this Learning Blueprint:
        
        Summary: {blueprint_result.blueprint_summary}
        
        Knowledge Primitives:
        {chr(10).join(f"- {primitive}" for primitive in blueprint_result.knowledge_primitives)}
        
        Cross References:
        {chr(10).join(f"- {ref}" for ref in blueprint_result.cross_references)}
        
        Note Style: {note_style}
        {length_instruction}
        {focus_instruction}
        
        User Preferences:
        - Include examples: {user_preferences.include_examples}
        - Include definitions: {user_preferences.include_definitions}
        - Focus on key concepts: {user_preferences.focus_on_key_concepts}
        - Max length: {user_preferences.max_note_length or 'No limit'}
        
        Return your response as a JSON object with this structure:
        {{
            "note_content": "BlockNote JSON format content",
            "plain_text": "Plain text version of the note",
            "structure": {{
                "main_sections": ["section1", "section2"],
                "key_points": ["point1", "point2"],
                "examples": ["example1", "example2"]
            }}
        }}
        
        The note should be well-structured, engaging, and follow the specified style.
        Use the BlockNote format for rich content structure.
        """
        return prompt
    
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
