"""
Granular Editing Service for precise content editing.
Handles line-level, section-level, and block-level edits with context preservation.
"""

import re
from typing import List, Optional, Dict, Any, Tuple
from app.models.note_creation_models import (
    NoteEditingRequest, GranularEditResult, NoteSectionContext
)
from app.services.llm_service import LLMService


class GranularEditingService:
    """Service for granular content editing with context preservation."""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def execute_granular_edit(
        self, 
        request: NoteEditingRequest, 
        current_content: str,
        context: NoteSectionContext
    ) -> Tuple[str, List[GranularEditResult]]:
        """
        Execute a granular edit operation.
        
        Args:
            request: Granular editing request
            current_content: Current note content
            context: Note section context
            
        Returns:
            Tuple of (edited_content, granular_edit_results)
        """
        try:
            if request.edit_type in ["edit_line", "add_line", "remove_line", "replace_line"]:
                return await self._execute_line_level_edit(request, current_content, context)
            
            elif request.edit_type in ["edit_section", "add_section", "remove_section", "reorder_sections"]:
                return await self._execute_section_level_edit(request, current_content, context)
            
            elif request.edit_type in ["edit_block", "add_block", "remove_block", "move_block"]:
                return await self._execute_block_level_edit(request, current_content, context)
            
            else:
                # Fall back to note-level editing
                return await self._execute_note_level_edit(request, current_content, context)
                
        except Exception as e:
            raise Exception(f"Granular edit failed: {str(e)}")
    
    async def _execute_line_level_edit(
        self, 
        request: NoteEditingRequest, 
        content: str, 
        context: NoteSectionContext
    ) -> Tuple[str, List[GranularEditResult]]:
        """Execute line-level editing operations."""
        
        lines = content.split('\n')
        edit_results = []
        
        if request.edit_type == "edit_line":
            if not request.target_line_number or request.target_line_number > len(lines):
                raise ValueError(f"Invalid line number: {request.target_line_number}")
            
            line_idx = request.target_line_number - 1
            original_line = lines[line_idx]
            
            # Use AI to edit the specific line
            edited_line = await self._ai_edit_line(
                original_line, 
                request.edit_instruction, 
                context
            )
            
            lines[line_idx] = edited_line
            
            edit_results.append(GranularEditResult(
                edit_type="edit_line",
                target_position=request.target_line_number,
                target_identifier=str(request.target_line_number),
                original_content=original_line,
                new_content=edited_line,
                context_preserved=True,
                surrounding_context=self._get_surrounding_context(lines, line_idx)
            ))
        
        elif request.edit_type == "add_line":
            if not request.insertion_position:
                raise ValueError("Insertion position required for add_line")
            
            insertion_idx = request.insertion_position - 1
            new_line = request.new_content or await self._ai_generate_line(
                request.edit_instruction, 
                context,
                self._get_surrounding_context(lines, insertion_idx)
            )
            
            lines.insert(insertion_idx, new_line)
            
            edit_results.append(GranularEditResult(
                edit_type="add_line",
                target_position=request.insertion_position,
                target_identifier=str(request.insertion_position),
                original_content="",
                new_content=new_line,
                context_preserved=True,
                surrounding_context=self._get_surrounding_context(lines, insertion_idx)
            ))
        
        elif request.edit_type == "remove_line":
            if not request.target_line_number or request.target_line_number > len(lines):
                raise ValueError(f"Invalid line number: {request.target_line_number}")
            
            line_idx = request.target_line_number - 1
            removed_line = lines.pop(line_idx)
            
            edit_results.append(GranularEditResult(
                edit_type="remove_line",
                target_position=request.target_line_number,
                target_identifier=str(request.target_line_number),
                original_content=removed_line,
                new_content="",
                context_preserved=True,
                surrounding_context=self._get_surrounding_context(lines, line_idx)
            ))
        
        elif request.edit_type == "replace_line":
            if not request.target_line_number or request.target_line_number > len(lines):
                raise ValueError(f"Invalid line number: {request.target_line_number}")
            
            line_idx = request.target_line_number - 1
            original_line = lines[line_idx]
            new_line = request.new_content or await self._ai_generate_line(
                request.edit_instruction, 
                context,
                self._get_surrounding_context(lines, line_idx)
            )
            
            lines[line_idx] = new_line
            
            edit_results.append(GranularEditResult(
                edit_type="replace_line",
                target_position=request.target_line_number,
                target_identifier=str(request.target_line_number),
                original_content=original_line,
                new_content=new_line,
                context_preserved=True,
                surrounding_context=self._get_surrounding_context(lines, line_idx)
            ))
        
        return '\n'.join(lines), edit_results
    
    async def _execute_section_level_edit(
        self, 
        request: NoteEditingRequest, 
        content: str, 
        context: NoteSectionContext
    ) -> Tuple[str, List[GranularEditResult]]:
        """Execute section-level editing operations."""
        
        sections = self._parse_sections(content)
        edit_results = []
        
        if request.edit_type == "edit_section":
            if not request.target_section_title:
                raise ValueError("Target section title required for edit_section")
            
            section_idx = self._find_section_index(sections, request.target_section_title)
            if section_idx == -1:
                raise ValueError(f"Section '{request.target_section_title}' not found")
            
            original_section = sections[section_idx]
            edited_section = await self._ai_edit_section(
                original_section, 
                request.edit_instruction, 
                context
            )
            
            sections[section_idx] = edited_section
            
            edit_results.append(GranularEditResult(
                edit_type="edit_section",
                target_position=section_idx + 1,
                target_identifier=request.target_section_title,
                original_content=original_section,
                new_content=edited_section,
                context_preserved=True,
                surrounding_context=self._get_section_context(sections, section_idx)
            ))
        
        elif request.edit_type == "add_section":
            if not request.insertion_position or not request.target_section_title:
                raise ValueError("Insertion position and section title required for add_section")
            
            insertion_idx = request.insertion_position - 1
            new_section = await self._ai_generate_section(
                request.target_section_title,
                request.new_content or request.edit_instruction,
                context,
                self._get_section_context(sections, insertion_idx)
            )
            
            sections.insert(insertion_idx, new_section)
            
            edit_results.append(GranularEditResult(
                edit_type="add_section",
                target_position=request.insertion_position,
                target_identifier=request.target_section_title,
                original_content="",
                new_content=new_section,
                context_preserved=True,
                surrounding_context=self._get_section_context(sections, insertion_idx)
            ))
        
        elif request.edit_type == "remove_section":
            if not request.target_section_title:
                raise ValueError("Target section title required for remove_section")
            
            section_idx = self._find_section_index(sections, request.target_section_title)
            if section_idx == -1:
                raise ValueError(f"Section '{request.target_section_title}' not found")
            
            removed_section = sections.pop(section_idx)
            
            edit_results.append(GranularEditResult(
                edit_type="remove_section",
                target_position=section_idx + 1,
                target_identifier=request.target_section_title,
                original_content=removed_section,
                new_content="",
                context_preserved=True,
                surrounding_context=self._get_section_context(sections, section_idx)
            ))
        
        return '\n\n'.join(sections), edit_results
    
    async def _execute_block_level_edit(
        self, 
        request: NoteEditingRequest, 
        content: str, 
        context: NoteSectionContext
    ) -> Tuple[str, List[GranularEditResult]]:
        """Execute block-level editing operations for BlockNote format."""
        
        try:
            import json
            blocks = json.loads(content)
        except json.JSONDecodeError:
            # If not valid JSON, treat as plain text
            return await self._execute_note_level_edit(request, content, context)
        
        edit_results = []
        
        if request.edit_type == "edit_block":
            if not request.target_block_id:
                raise ValueError("Target block ID required for edit_block")
            
            block_idx = self._find_block_index(blocks, request.target_block_id)
            if block_idx == -1:
                raise ValueError(f"Block '{request.target_block_id}' not found")
            
            original_block = blocks["content"][block_idx]
            edited_block = await self._ai_edit_block(
                original_block, 
                request.edit_instruction, 
                context
            )
            
            blocks["content"][block_idx] = edited_block
            
            edit_results.append(GranularEditResult(
                edit_type="edit_block",
                target_position=block_idx + 1,
                target_identifier=request.target_block_id,
                original_content=str(original_block),
                new_content=str(edited_block),
                context_preserved=True,
                surrounding_context=self._get_block_context(blocks, block_idx)
            ))
        
        return json.dumps(blocks), edit_results
    
    async def _execute_note_level_edit(
        self, 
        request: NoteEditingRequest, 
        content: str, 
        context: NoteSectionContext
    ) -> Tuple[str, List[GranularEditResult]]:
        """Execute note-level editing operations (fallback)."""
        
        # Use existing note-level editing logic
        prompt = f"""
        Edit the following note content according to the instruction.
        
        Content: {content}
        Instruction: {request.edit_instruction}
        Edit Type: {request.edit_type}
        
        Return the edited content maintaining the same structure and format.
        """
        
        edited_content = await self.llm_service.call_llm(prompt, max_tokens=4000)
        
        edit_results = [GranularEditResult(
            edit_type=request.edit_type,
            target_position=1,
            target_identifier="entire_note",
            original_content=content,
            new_content=edited_content,
            context_preserved=True,
            surrounding_context=""
        )]
        
        return edited_content, edit_results
    
    # Helper methods for AI-powered editing
    
    async def _ai_edit_line(
        self, 
        line: str, 
        instruction: str, 
        context: NoteSectionContext
    ) -> str:
        """Use AI to edit a specific line."""
        
        prompt = f"""
        Edit ONLY this specific line according to the instruction.
        
        Line: "{line}"
        Instruction: {instruction}
        
        Blueprint Context:
        - Section: {context.section_hierarchy[-1]['title'] if context.section_hierarchy else 'Unknown'}
        - Knowledge Primitives: {', '.join(context.knowledge_primitives)}
        
        Return ONLY the edited line. Do not add any other content or explanations.
        """
        
        return await self.llm_service.call_llm(prompt, max_tokens=200)
    
    async def _ai_generate_line(
        self, 
        instruction: str, 
        context: NoteSectionContext,
        surrounding_context: str
    ) -> str:
        """Use AI to generate a new line."""
        
        prompt = f"""
        Generate a new line of content based on the instruction and context.
        
        Instruction: {instruction}
        Surrounding Context: {surrounding_context}
        
        Blueprint Context:
        - Section: {context.section_hierarchy[-1]['title'] if context.section_hierarchy else 'Unknown'}
        - Knowledge Primitives: {', '.join(context.knowledge_primitives)}
        
        Return ONLY the new line. Do not add any other content or explanations.
        """
        
        return await self.llm_service.call_llm(prompt, max_tokens=200)
    
    async def _ai_edit_section(
        self, 
        section: str, 
        instruction: str, 
        context: NoteSectionContext
    ) -> str:
        """Use AI to edit a specific section."""
        
        prompt = f"""
        Edit this specific section according to the instruction.
        
        Section: {section}
        Instruction: {instruction}
        
        Blueprint Context:
        - Section: {context.section_hierarchy[-1]['title'] if context.section_hierarchy else 'Unknown'}
        - Knowledge Primitives: {', '.join(context.knowledge_primitives)}
        
        Return the edited section. Maintain the same structure and format.
        """
        
        return await self.llm_service.call_llm(prompt, max_tokens=1000)
    
    async def _ai_generate_section(
        self, 
        title: str, 
        content: str, 
        context: NoteSectionContext,
        surrounding_context: str
    ) -> str:
        """Use AI to generate a new section."""
        
        prompt = f"""
        Generate a new section with the given title and content.
        
        Title: {title}
        Content: {content}
        Surrounding Context: {surrounding_context}
        
        Blueprint Context:
        - Section: {context.section_hierarchy[-1]['title'] if context.section_hierarchy else 'Unknown'}
        - Knowledge Primitives: {', '.join(context.knowledge_primitives)}
        
        Return the complete section with proper formatting.
        """
        
        return await self.llm_service.call_llm(prompt, max_tokens=1000)
    
    async def _ai_edit_block(
        self, 
        block: Dict[str, Any], 
        instruction: str, 
        context: NoteSectionContext
    ) -> Dict[str, Any]:
        """Use AI to edit a specific BlockNote block."""
        
        prompt = f"""
        Edit this BlockNote block according to the instruction.
        
        Block: {block}
        Instruction: {instruction}
        
        Blueprint Context:
        - Section: {context.section_hierarchy[-1]['title'] if context.section_hierarchy else 'Unknown'}
        - Knowledge Primitives: {', '.join(context.knowledge_primitives)}
        
        Return the edited block in valid BlockNote format.
        """
        
        edited_block_str = await self.llm_service.call_llm(prompt, max_tokens=500)
        
        try:
            import json
            return json.loads(edited_block_str)
        except json.JSONDecodeError:
            # If AI didn't return valid JSON, return original block
            return block
    
    # Utility methods
    
    def _parse_sections(self, content: str) -> List[str]:
        """Parse content into sections based on headers."""
        # Enhanced section parsing for markdown
        lines = content.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            if line.strip().startswith('#'):
                # If we have content in current section, save it
                if current_section:
                    sections.append('\n'.join(current_section).strip())
                # Start new section
                current_section = [line]
            else:
                current_section.append(line)
        
        # Don't forget the last section
        if current_section:
            sections.append('\n'.join(current_section).strip())
        
        return [s for s in sections if s.strip()]
    
    def _find_section_index(self, sections: List[str], title: str) -> int:
        """Find the index of a section by title."""
        title_lower = title.lower().strip()
        for i, section in enumerate(sections):
            # Look for the title in the first line (header)
            first_line = section.split('\n')[0].strip()
            if title_lower in first_line.lower():
                return i
        return -1
    
    def _find_block_index(self, blocks: Dict[str, Any], block_id: str) -> int:
        """Find the index of a block by ID."""
        if "content" not in blocks:
            return -1
        
        for i, block in enumerate(blocks["content"]):
            if block.get("id") == block_id:
                return i
        return -1
    
    def _get_surrounding_context(self, lines: List[str], line_idx: int, context_lines: int = 2) -> str:
        """Get surrounding context for a line."""
        start_idx = max(0, line_idx - context_lines)
        end_idx = min(len(lines), line_idx + context_lines + 1)
        return '\n'.join(lines[start_idx:end_idx])
    
    def _get_section_context(self, sections: List[str], section_idx: int, context_sections: int = 1) -> str:
        """Get surrounding context for a section."""
        start_idx = max(0, section_idx - context_sections)
        end_idx = min(len(sections), section_idx + context_sections + 1)
        return '\n\n'.join(sections[start_idx:end_idx])
    
    def _get_block_context(self, blocks: Dict[str, Any], block_idx: int, context_blocks: int = 1) -> str:
        """Get surrounding context for a block."""
        if "content" not in blocks:
            return ""
        
        start_idx = max(0, block_idx - context_blocks)
        end_idx = min(len(blocks["content"]), block_idx + context_blocks + 1)
        context_blocks_list = blocks["content"][start_idx:end_idx]
        return str(context_blocks_list)
