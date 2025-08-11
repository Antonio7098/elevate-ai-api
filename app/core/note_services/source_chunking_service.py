"""
Source Chunking Service for intelligent content segmentation.
Implements hybrid approach: algorithmic detection + LLM validation.
"""

import re
import time
from typing import List, Dict, Any, Optional
from app.models.note_creation_models import (
    SourceChunk, ChunkingStrategy, AlgorithmicDetectionResult, ChunkingResult
)
from app.services.llm_service import LLMService


class SourceChunkingService:
    """Service for intelligent source content chunking."""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        
        # Common section patterns
        self.section_patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^<h[1-6][^>]*>(.+?)</h[1-6]>',  # HTML headers
            r'^[A-Z][A-Z\s]{3,}$',  # ALL CAPS section titles
            r'^\d+\.\s+(.+)$',  # Numbered sections
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Title case phrases
        ]
        
        # Font size patterns (common in PDFs and formatted docs)
        self.font_patterns = [
            r'font-size:\s*(\d+)',  # CSS font-size
            r'size="(\d+)"',  # HTML size attribute
        ]
    
    async def chunk_source_content(
        self, 
        source_content: str, 
        strategy: ChunkingStrategy
    ) -> ChunkingResult:
        """
        Main method to chunk source content using hybrid approach.
        
        Args:
            source_content: The source text to chunk
            strategy: Chunking configuration
            
        Returns:
            ChunkingResult with chunks and metadata
        """
        start_time = time.time()
        
        if len(source_content) < (strategy.max_chunk_size or 8000):
            # Content is small enough, no chunking needed
            chunk = SourceChunk(
                chunk_id="chunk_001",
                content=source_content,
                start_position=0,
                end_position=len(source_content),
                topic="main_content"
            )
            
            return ChunkingResult(
                success=True,
                chunks=[chunk],
                total_chunks=1,
                processing_strategy="single_chunk",
                llm_validation_used=False,
                processing_time=time.time() - start_time,
                message="Content was small enough to process as single chunk"
            )
        
        # Step 1: Algorithmic detection
        detection_result = self._detect_sections_algorithmically(source_content, strategy)
        
        # Step 2: LLM validation if needed
        llm_validation_used = False
        if detection_result.needs_llm_validation:
            detection_result = await self._validate_with_llm(
                source_content, detection_result, strategy
            )
            llm_validation_used = True
        
        # Step 3: Create final chunks
        chunks = self._create_chunks_from_detection(source_content, detection_result, strategy)
        
        # Step 4: Apply size constraints and overlap
        chunks = self._apply_size_constraints(chunks, strategy)
        
        processing_time = time.time() - start_time
        
        return ChunkingResult(
            success=True,
            chunks=chunks,
            total_chunks=len(chunks),
            processing_strategy="hybrid_chunking",
            llm_validation_used=llm_validation_used,
            processing_time=processing_time,
            message=f"Successfully chunked content into {len(chunks)} chunks"
        )
    
    def _detect_sections_algorithmically(
        self, 
        content: str, 
        strategy: ChunkingStrategy
    ) -> AlgorithmicDetectionResult:
        """Fast algorithmic detection of content sections."""
        sections = []
        confidence_scores = []
        suggested_chunks = []
        
        lines = content.split('\n')
        current_section_start = 0
        current_section_content = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if line matches section patterns
            section_score = self._calculate_section_score(line)
            
            if section_score > 0.7:  # High confidence section break
                if current_section_content:
                    # End current section
                    section_content = '\n'.join(current_section_content)
                    sections.append({
                        'start_line': current_section_start,
                        'end_line': i - 1,
                        'content': section_content,
                        'title': line,
                        'pattern_matched': self._identify_pattern(line)
                    })
                    confidence_scores.append(section_score)
                    
                    # Create chunk
                    chunk = SourceChunk(
                        chunk_id=f"chunk_{len(sections):03d}",
                        content=section_content,
                        start_position=current_section_start,
                        end_position=i - 1,
                        topic=line[:100]  # Use first 100 chars as topic
                    )
                    suggested_chunks.append(chunk)
                    
                    # Start new section
                    current_section_start = i
                    current_section_content = [line]
                else:
                    current_section_content.append(line)
            else:
                current_section_content.append(line)
        
        # Add final section
        if current_section_content:
            section_content = '\n'.join(current_section_content)
            sections.append({
                'start_line': current_section_start,
                'end_line': len(lines) - 1,
                'content': section_content,
                'title': current_section_content[0][:100],
                'pattern_matched': 'content'
            })
            confidence_scores.append(0.9)  # High confidence for content sections
            
            chunk = SourceChunk(
                chunk_id=f"chunk_{len(sections):03d}",
                content=section_content,
                start_position=current_section_start,
                end_position=len(content),
                topic=current_section_content[0][:100]
            )
            suggested_chunks.append(chunk)
        
        # Determine if LLM validation is needed
        needs_llm_validation = (
            len(sections) > 10 or  # Many sections
            any(score < 0.6 for score in confidence_scores) or  # Low confidence
            not strategy.use_algorithmic_detection  # Force LLM validation
        )
        
        return AlgorithmicDetectionResult(
            detected_sections=sections,
            confidence_scores=confidence_scores,
            suggested_chunks=suggested_chunks,
            needs_llm_validation=needs_llm_validation
        )
    
    def _calculate_section_score(self, line: str) -> float:
        """Calculate confidence score for a line being a section break."""
        score = 0.0
        
        # Check markdown headers
        if re.match(r'^#{1,6}\s+', line):
            score += 0.9
        
        # Check HTML headers
        if re.match(r'^<h[1-6][^>]*>', line):
            score += 0.8
        
        # Check ALL CAPS titles
        if re.match(r'^[A-Z][A-Z\s]{3,}$', line):
            score += 0.7
        
        # Check numbered sections
        if re.match(r'^\d+\.\s+', line):
            score += 0.6
        
        # Check title case phrases
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', line):
            score += 0.5
        
        # Bonus for short lines (likely titles)
        if len(line) < 100:
            score += 0.1
        
        return min(score, 1.0)
    
    def _identify_pattern(self, line: str) -> str:
        """Identify which pattern matched for a section line."""
        if re.match(r'^#{1,6}\s+', line):
            return 'markdown_header'
        elif re.match(r'^<h[1-6][^>]*>', line):
            return 'html_header'
        elif re.match(r'^[A-Z][A-Z\s]{3,}$', line):
            return 'all_caps_title'
        elif re.match(r'^\d+\.\s+', line):
            return 'numbered_section'
        elif re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', line):
            return 'title_case'
        else:
            return 'content'
    
    async def _validate_with_llm(
        self, 
        content: str, 
        detection_result: AlgorithmicDetectionResult,
        strategy: ChunkingStrategy
    ) -> AlgorithmicDetectionResult:
        """Use LLM to validate and refine algorithmic detection."""
        # Create prompt for LLM validation
        prompt = self._create_validation_prompt(content, detection_result)
        
        try:
            response = await self.llm_service.call_llm(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.1
            )
            
            # Parse LLM response and update detection result
            validated_chunks = self._parse_llm_validation_response(response, content)
            if validated_chunks:
                detection_result.suggested_chunks = validated_chunks
            detection_result.needs_llm_validation = False
            
        except Exception as e:
            # If LLM validation fails, keep algorithmic result
            print(f"LLM validation failed: {e}. Using algorithmic detection.")
        
        return detection_result
    
    def _create_validation_prompt(
        self, 
        content: str, 
        detection_result: AlgorithmicDetectionResult
    ) -> str:
        """Create prompt for LLM validation of chunking."""
        prompt = f"""
        You are an expert at analyzing document structure and identifying natural content boundaries.
        
        I have algorithmically detected the following sections in a document:
        
        {self._format_detection_for_prompt(detection_result)}
        
        Please validate and refine these section breaks. Consider:
        1. Are the boundaries semantically meaningful?
        2. Are there missing section breaks?
        3. Are any breaks unnecessary?
        4. What would be the best topic title for each section?
        
        Return your response as a JSON array of chunks with this structure:
        {{
            "chunk_id": "chunk_001",
            "start_position": 0,
            "end_position": 1500,
            "topic": "Introduction and Overview",
            "reasoning": "Natural break after introduction section"
        }}
        
        Focus on creating coherent, topic-based chunks that maintain the document's logical flow.
        """
        return prompt
    
    def _format_detection_for_prompt(self, detection_result: AlgorithmicDetectionResult) -> str:
        """Format detection result for LLM prompt."""
        formatted = []
        for i, section in enumerate(detection_result.detected_sections):
            formatted.append(f"""
            Section {i+1}:
            - Title: {section['title']}
            - Pattern: {section['pattern_matched']}
            - Lines: {section['start_line']}-{section['end_line']}
            - Content preview: {section['content'][:200]}...
            """)
        return '\n'.join(formatted)
    
    def _parse_llm_validation_response(self, response: str, content: str) -> List[SourceChunk]:
        """Parse LLM validation response into SourceChunk objects."""
        try:
            # Extract JSON from response
            import json
            
            # Find JSON array in response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return []
            
            chunks_data = json.loads(json_match.group())
            chunks = []
            
            for chunk_data in chunks_data:
                chunk = SourceChunk(
                    chunk_id=chunk_data.get('chunk_id', f"chunk_{len(chunks):03d}"),
                    content=content[chunk_data['start_position']:chunk_data['end_position']],
                    start_position=chunk_data['start_position'],
                    end_position=chunk_data['end_position'],
                    topic=chunk_data.get('topic', 'Untitled Section')
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            print(f"Failed to parse LLM validation response: {e}")
            return []
    
    def _create_chunks_from_detection(
        self, 
        content: str, 
        detection_result: AlgorithmicDetectionResult,
        strategy: ChunkingStrategy
    ) -> List[SourceChunk]:
        """Create final chunks from detection results."""
        chunks = []
        
        for chunk in detection_result.suggested_chunks:
            # Ensure chunk doesn't exceed max size
            if strategy.max_chunk_size and len(chunk.content) > strategy.max_chunk_size:
                # Split large chunk
                sub_chunks = self._split_large_chunk(chunk, strategy)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk)
        
        return chunks
    
    def _split_large_chunk(self, chunk: SourceChunk, strategy: ChunkingStrategy) -> List[SourceChunk]:
        """Split a chunk that exceeds maximum size."""
        if not strategy.max_chunk_size:
            return [chunk]
        
        sub_chunks = []
        content = chunk.content
        start_pos = chunk.start_position
        chunk_num = 1
        
        while len(content) > strategy.max_chunk_size:
            # Find good break point
            break_point = self._find_break_point(content, strategy.max_chunk_size)
            
            # Create sub-chunk
            sub_content = content[:break_point]
            sub_chunk = SourceChunk(
                chunk_id=f"{chunk.chunk_id}_sub_{chunk_num:02d}",
                content=sub_content,
                start_position=start_pos,
                end_position=start_pos + break_point,
                topic=f"{chunk.topic} (Part {chunk_num})",
                parent_chunk_id=chunk.chunk_id
            )
            sub_chunks.append(sub_chunk)
            
            # Update for next iteration
            content = content[break_point - strategy.chunk_overlap:]
            start_pos += break_point - strategy.chunk_overlap
            chunk_num += 1
        
        # Add remaining content as final sub-chunk
        if content:
            sub_chunk = SourceChunk(
                chunk_id=f"{chunk.chunk_id}_sub_{chunk_num:02d}",
                content=content,
                start_position=start_pos,
                end_position=chunk.end_position,
                topic=f"{chunk.topic} (Part {chunk_num})",
                parent_chunk_id=chunk.chunk_id
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def _find_break_point(self, content: str, max_size: int) -> int:
        """Find a good break point in content."""
        # Look for paragraph breaks near the target size
        target = max_size
        
        # Look for paragraph breaks
        for i in range(target, max(0, target - 500), -1):
            if i < len(content) and content[i:i+2] == '\n\n':
                return i + 2
        
        # Look for sentence breaks
        for i in range(target, max(0, target - 300), -1):
            if i < len(content) and content[i] in '.!?':
                return i + 1
        
        # Look for word breaks
        for i in range(target, max(0, target - 100), -1):
            if i < len(content) and content[i] == ' ':
                return i + 1
        
        # Fallback to target size
        return min(target, len(content))
    
    def _apply_size_constraints(
        self, 
        chunks: List[SourceChunk], 
        strategy: ChunkingStrategy
    ) -> List[SourceChunk]:
        """Apply final size constraints and overlap."""
        if not strategy.max_chunk_size:
            return chunks
        
        final_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Add overlap with previous chunk if specified
            if strategy.chunk_overlap > 0 and i > 0:
                prev_chunk = chunks[i-1]
                overlap_start = max(0, chunk.start_position - strategy.chunk_overlap)
                overlap_content = prev_chunk.content[-strategy.chunk_overlap:]
                
                # Update chunk with overlap
                chunk.content = overlap_content + chunk.content
                chunk.start_position = overlap_start
            
            final_chunks.append(chunk)
        
        return final_chunks
