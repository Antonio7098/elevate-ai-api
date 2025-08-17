"""
Granular Blueprint Editing Service

Provides precise, targeted editing capabilities for blueprint components
including sections, primitives, mastery criteria, and questions.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from app.services.llm_service import LLMService
from app.models.blueprint_editing_models import (
    BlueprintEditingRequest, GranularEditResult, BlueprintContext
)


class GranularBlueprintEditingService:
    """Service for granular editing of blueprint components."""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def execute_blueprint_granular_edit(
        self,
        request: BlueprintEditingRequest,
        context: BlueprintContext
    ) -> Tuple[Dict[str, Any], List[GranularEditResult]]:
        """
        Execute a granular edit operation on a blueprint.
        
        Args:
            request: The editing request
            context: Blueprint context information
            
        Returns:
            Tuple of (edited_content, granular_edits)
        """
        try:
            # Determine the type of granular edit
            if request.edit_type.startswith("edit_section"):
                return await self._execute_section_edit(request, context)
            elif request.edit_type.startswith("edit_primitive"):
                return await self._execute_primitive_edit(request, context)
            elif request.edit_type.startswith("edit_criterion"):
                return await self._execute_criterion_edit(request, context)
            elif request.edit_type.startswith("edit_question"):
                return await self._execute_question_edit(request, context)
            elif request.edit_type.startswith("add_"):
                return await self._execute_add_operation(request, context)
            elif request.edit_type.startswith("remove_"):
                return await self._execute_remove_operation(request, context)
            elif request.edit_type.startswith("reorder_"):
                return await self._execute_reorder_operation(request, context)
            else:
                # Fallback to general content edit
                return await self._execute_general_content_edit(request, context)
                
        except Exception as e:
            print(f"Error in granular edit: {e}")
            # Return fallback content
            return self._get_fallback_content(context), []
    
    async def _execute_section_edit(
        self, 
        request: BlueprintEditingRequest, 
        context: BlueprintContext
    ) -> Tuple[Dict[str, Any], List[GranularEditResult]]:
        """Execute a section-level edit operation."""
        
        prompt = f"""
        Execute a granular edit on a blueprint section.
        
        Blueprint Context:
        - ID: {context.blueprint_id}
        - Title: {context.title}
        - Sections: {context.sections_count}
        
        Edit Request:
        - Type: {request.edit_type}
        - Instruction: {request.edit_instruction}
        
        Please perform the requested edit and return:
        1. The edited section content
        2. A list of specific changes made
        
        Return as JSON with this structure:
        {{
            "edited_content": "The edited section content",
            "changes_made": [
                {{
                    "type": "edit_type",
                    "description": "What was changed",
                    "position": "Where the change was made",
                    "original": "Original content",
                    "new": "New content"
                }}
            ]
        }}
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.2
        )
        
        edit_data = self._parse_edit_response(response)
        edited_content = edit_data.get('edited_content', '')
        
        # Create granular edit results
        granular_edits = []
        for change in edit_data.get('changes_made', []):
            granular_edits.append(GranularEditResult(
                edit_id=f"section_edit_{int(time.time())}_{len(granular_edits)}",
                edit_type=change.get('type', request.edit_type),
                target_position=change.get('position', ''),
                original_content=change.get('original', ''),
                new_content=change.get('new', ''),
                confidence=0.9,
                reasoning=f"Applied {request.edit_type} based on user instruction: {request.edit_instruction}",
                metadata={'section_id': 'target_section', 'change_type': 'section_edit'}
            ))
        
        return {'sections': [edited_content]}, granular_edits
    
    async def _execute_primitive_edit(
        self, 
        request: BlueprintEditingRequest, 
        context: BlueprintContext
    ) -> Tuple[Dict[str, Any], List[GranularEditResult]]:
        """Execute a primitive-level edit operation."""
        
        prompt = f"""
        Execute a granular edit on a knowledge primitive within a blueprint.
        
        Blueprint Context:
        - ID: {context.blueprint_id}
        - Title: {context.title}
        - Primitives: {context.primitives_count}
        
        Edit Request:
        - Type: {request.edit_type}
        - Instruction: {request.edit_instruction}
        
        Please perform the requested edit and return:
        1. The edited primitive content
        2. A list of specific changes made
        
        Return as JSON with this structure:
        {{
            "edited_content": "The edited primitive content",
            "changes_made": [
                {{
                    "type": "edit_type",
                    "description": "What was changed",
                    "position": "Where the change was made",
                    "original": "Original content",
                    "new": "New content"
                }}
            ]
        }}
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.2
        )
        
        edit_data = self._parse_edit_response(response)
        edited_content = edit_data.get('edited_content', '')
        
        # Create granular edit results
        granular_edits = []
        for change in edit_data.get('changes_made', []):
            granular_edits.append(GranularEditResult(
                edit_id=f"primitive_edit_{int(time.time())}_{len(granular_edits)}",
                edit_type=change.get('type', request.edit_type),
                target_position=change.get('position', ''),
                original_content=change.get('original', ''),
                new_content=change.get('new', ''),
                confidence=0.9,
                reasoning=f"Applied {request.edit_type} based on user instruction: {request.edit_instruction}",
                metadata={'primitive_id': 'target_primitive', 'change_type': 'primitive_edit'}
            ))
        
        return {'primitives': [edited_content]}, granular_edits
    
    async def _execute_criterion_edit(
        self, 
        request: BlueprintEditingRequest, 
        context: BlueprintContext
    ) -> Tuple[Dict[str, Any], List[GranularEditResult]]:
        """Execute a mastery criterion-level edit operation."""
        
        prompt = f"""
        Execute a granular edit on a mastery criterion within a blueprint.
        
        Blueprint Context:
        - ID: {context.blueprint_id}
        - Title: {context.title}
        - Criteria: {context.criteria_count}
        
        Edit Request:
        - Type: {request.edit_type}
        - Instruction: {request.edit_instruction}
        
        Please perform the requested edit and return:
        1. The edited criterion content
        2. A list of specific changes made
        
        Return as JSON with this structure:
        {{
            "edited_content": "The edited criterion content",
            "changes_made": [
                {{
                    "type": "edit_type",
                    "description": "What was changed",
                    "position": "Where the change was made",
                    "original": "Original content",
                    "new": "New content"
                }}
            ]
        }}
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.2
        )
        
        edit_data = self._parse_edit_response(response)
        edited_content = edit_data.get('edited_content', '')
        
        # Create granular edit results
        granular_edits = []
        for change in edit_data.get('changes_made', []):
            granular_edits.append(GranularEditResult(
                edit_id=f"criterion_edit_{int(time.time())}_{len(granular_edits)}",
                edit_type=change.get('type', request.edit_type),
                target_position=change.get('position', ''),
                original_content=change.get('original', ''),
                new_content=change.get('new', ''),
                confidence=0.9,
                reasoning=f"Applied {request.edit_type} based on user instruction: {request.edit_instruction}",
                metadata={'criterion_id': 'target_criterion', 'change_type': 'criterion_edit'}
            ))
        
        return {'criteria': [edited_content]}, granular_edits
    
    async def _execute_question_edit(
        self, 
        request: BlueprintEditingRequest, 
        context: BlueprintContext
    ) -> Tuple[Dict[str, Any], List[GranularEditResult]]:
        """Execute a question-level edit operation."""
        
        prompt = f"""
        Execute a granular edit on a question within a blueprint.
        
        Blueprint Context:
        - ID: {context.blueprint_id}
        - Title: {context.title}
        - Questions: {context.questions_count}
        
        Edit Request:
        - Type: {request.edit_type}
        - Instruction: {request.edit_instruction}
        
        Please perform the requested edit and return:
        1. The edited question content
        2. A list of specific changes made
        
        Return as JSON with this structure:
        {{
            "edited_content": "The edited question content",
            "changes_made": [
                {{
                    "type": "edit_type",
                    "description": "What was changed",
                    "position": "Where the change was made",
                    "original": "Original content",
                    "new": "New content"
                }}
            ]
        }}
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.2
        )
        
        edit_data = self._parse_edit_response(response)
        edited_content = edit_data.get('edited_content', '')
        
        # Create granular edit results
        granular_edits = []
        for change in edit_data.get('changes_made', []):
            granular_edits.append(GranularEditResult(
                edit_id=f"question_edit_{int(time.time())}_{len(granular_edits)}",
                edit_type=change.get('type', request.edit_type),
                target_position=change.get('position', ''),
                original_content=change.get('original', ''),
                new_content=change.get('new', ''),
                confidence=0.9,
                reasoning=f"Applied {request.edit_type} based on user instruction: {request.edit_instruction}",
                metadata={'question_id': 'target_question', 'change_type': 'question_edit'}
            ))
        
        return {'questions': [edited_content]}, granular_edits
    
    async def _execute_add_operation(
        self, 
        request: BlueprintEditingRequest, 
        context: BlueprintContext
    ) -> Tuple[Dict[str, Any], List[GranularEditResult]]:
        """Execute an add operation."""
        
        prompt = f"""
        Add new content to a blueprint based on the user's request.
        
        Blueprint Context:
        - ID: {context.blueprint_id}
        - Title: {context.title}
        
        Add Request:
        - Type: {request.edit_type}
        - Instruction: {request.edit_instruction}
        
        Please create the new content and return:
        1. The new content to add
        2. Where it should be placed
        3. How it integrates with existing content
        
        Return as JSON with this structure:
        {{
            "new_content": "The new content to add",
            "placement": "Where to place the new content",
            "integration_notes": "How it integrates with existing content",
            "changes_made": [
                {{
                    "type": "add",
                    "description": "What was added",
                    "position": "Where it was placed",
                    "content": "The new content"
                }}
            ]
        }}
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.3
        )
        
        add_data = self._parse_edit_response(response)
        new_content = add_data.get('new_content', '')
        
        # Create granular edit results
        granular_edits = []
        for change in add_data.get('changes_made', []):
            granular_edits.append(GranularEditResult(
                edit_id=f"add_{int(time.time())}_{len(granular_edits)}",
                edit_type='add',
                target_position=change.get('position', ''),
                original_content='',
                new_content=change.get('content', ''),
                confidence=0.9,
                reasoning=f"Added new content based on user instruction: {request.edit_instruction}",
                metadata={'operation': 'add', 'placement': change.get('position', '')}
            ))
        
        return {'added_content': new_content}, granular_edits
    
    async def _execute_remove_operation(
        self, 
        request: BlueprintEditingRequest, 
        context: BlueprintContext
    ) -> Tuple[Dict[str, Any], List[GranularEditResult]]:
        """Execute a remove operation."""
        
        prompt = f"""
        Remove content from a blueprint based on the user's request.
        
        Blueprint Context:
        - ID: {context.blueprint_id}
        - Title: {context.title}
        
        Remove Request:
        - Type: {request.edit_type}
        - Instruction: {request.edit_instruction}
        
        Please identify what to remove and return:
        1. What content should be removed
        2. The impact of removal
        3. How to handle dependencies
        
        Return as JSON with this structure:
        {{
            "removed_content": "What was removed",
            "removal_impact": "Impact of the removal",
            "dependencies": "How dependencies were handled",
            "changes_made": [
                {{
                    "type": "remove",
                    "description": "What was removed",
                    "position": "Where it was removed from",
                    "content": "The removed content"
                }}
            ]
        }}
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.2
        )
        
        remove_data = self._parse_edit_response(response)
        removed_content = remove_data.get('removed_content', '')
        
        # Create granular edit results
        granular_edits = []
        for change in remove_data.get('changes_made', []):
            granular_edits.append(GranularEditResult(
                edit_id=f"remove_{int(time.time())}_{len(granular_edits)}",
                edit_type='remove',
                target_position=change.get('position', ''),
                original_content=change.get('content', ''),
                new_content='',
                confidence=0.9,
                reasoning=f"Removed content based on user instruction: {request.edit_instruction}",
                metadata={'operation': 'remove', 'impact': remove_data.get('removal_impact', '')}
            ))
        
        return {'removed_content': removed_content}, granular_edits
    
    async def _execute_reorder_operation(
        self, 
        request: BlueprintEditingRequest, 
        context: BlueprintContext
    ) -> Tuple[Dict[str, Any], List[GranularEditResult]]:
        """Execute a reorder operation."""
        
        prompt = f"""
        Reorder content within a blueprint based on the user's request.
        
        Blueprint Context:
        - ID: {context.blueprint_id}
        - Title: {context.title}
        
        Reorder Request:
        - Type: {request.edit_type}
        - Instruction: {request.edit_instruction}
        
        Please determine the new order and return:
        1. The new ordering of content
        2. What was moved where
        3. The reasoning for the new order
        
        Return as JSON with this structure:
        {{
            "new_order": "The new ordering of content",
            "reordering_reason": "Why this order makes sense",
            "changes_made": [
                {{
                    "type": "reorder",
                    "description": "What was reordered",
                    "from_position": "Original position",
                    "to_position": "New position",
                    "content": "The content that was moved"
                }}
            ]
        }}
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.2
        )
        
        reorder_data = self._parse_edit_response(response)
        new_order = reorder_data.get('new_order', '')
        
        # Create granular edit results
        granular_edits = []
        for change in reorder_data.get('changes_made', []):
            granular_edits.append(GranularEditResult(
                edit_id=f"reorder_{int(time.time())}_{len(granular_edits)}",
                edit_type='reorder',
                target_position=change.get('to_position', ''),
                original_content=f"From: {change.get('from_position', '')}",
                new_content=f"To: {change.get('to_position', '')}",
                confidence=0.9,
                reasoning=f"Reordered content based on user instruction: {request.edit_instruction}",
                metadata={
                    'operation': 'reorder', 
                    'from_position': change.get('from_position', ''),
                    'to_position': change.get('to_position', '')
                }
            ))
        
        return {'new_order': new_order}, granular_edits
    
    async def _execute_general_content_edit(
        self, 
        request: BlueprintEditingRequest, 
        context: BlueprintContext
    ) -> Tuple[Dict[str, Any], List[GranularEditResult]]:
        """Execute a general content edit operation."""
        
        prompt = f"""
        Edit blueprint content based on the user's request.
        
        Blueprint Context:
        - ID: {context.blueprint_id}
        - Title: {context.title}
        
        Edit Request:
        - Type: {request.edit_type}
        - Instruction: {request.edit_instruction}
        
        Please perform the requested edit and return:
        1. The edited content
        2. A list of specific changes made
        
        Return as JSON with this structure:
        {{
            "edited_content": "The edited content",
            "changes_made": [
                {{
                    "type": "edit_type",
                    "description": "What was changed",
                    "position": "Where the change was made",
                    "original": "Original content",
                    "new": "New content"
                }}
            ]
        }}
        """
        
        response = await self.llm_service.call_llm(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.3
        )
        
        edit_data = self._parse_edit_response(response)
        edited_content = edit_data.get('edited_content', '')
        
        # Create granular edit results
        granular_edits = []
        for change in edit_data.get('changes_made', []):
            granular_edits.append(GranularEditResult(
                edit_id=f"general_edit_{int(time.time())}_{len(granular_edits)}",
                edit_type=change.get('type', request.edit_type),
                target_position=change.get('position', ''),
                original_content=change.get('original', ''),
                new_content=change.get('new', ''),
                confidence=0.8,
                reasoning=f"Applied general edit based on user instruction: {request.edit_instruction}",
                metadata={'operation': 'general_edit', 'change_type': change.get('type', 'unknown')}
            ))
        
        return {'edited_content': edited_content}, granular_edits
    
    def _get_fallback_content(self, context: BlueprintContext) -> Dict[str, Any]:
        """Get fallback content when editing fails."""
        return {
            'blueprint_id': context.blueprint_id,
            'title': context.title,
            'message': 'Edit operation failed, using fallback content',
            'sections': [],
            'primitives': [],
            'criteria': [],
            'questions': []
        }
    
    def _parse_edit_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into edit data."""
        try:
            import json
            import re
            
            # Find JSON object in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return {}
            
            return json.loads(json_match.group())
            
        except Exception as e:
            print(f"Failed to parse edit response: {e}")
            return {}
