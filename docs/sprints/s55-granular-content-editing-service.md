# Sprint 55: Granular Content Editing Service

**Signed off** DO NOT PROCEED UNLESS SIGNED OFF BY ANTONIO
**Date Range:** [To be determined]
**Primary Focus:** Python AI - Granular Content Editing Service for Blueprints, Primitives, Mastery Criteria, and Questions
**Overview:** Create a focused, direct content editing service that allows users to add, edit, and modify specific content elements without complex AI analysis workflows. Focus on granular edits first, with complete rewrite capabilities as secondary features.

---

## I. Sprint Goals & Objectives

### Primary Goals:
1. **Implement granular content editing service** for direct manipulation of blueprint components
2. **Create focused editing tools** for sections, primitives, mastery criteria, and questions
3. **Establish proper content hierarchy management** following the Prisma schema relationships
4. **Build comprehensive testing suite** including performance and optimization tests
5. **Ensure content integrity** through proper validation and relationship maintenance
6. **Maintain separation** between granular editing and note creation systems
7. **Respect existing architecture** while adding new granular capabilities
8. **Implement granular note editing** for BlockNote JSON objects with proper schema compliance

### Success Criteria:
- Users can add/edit/delete blueprint sections with proper hierarchy management
- Users can create/modify knowledge primitives linked to specific sections
- Users can add/edit mastery criteria linked to primitives
- Users can create/modify questions linked to mastery criteria
- All operations maintain referential integrity following Prisma schema
- Service responds within 200ms for simple operations
- Comprehensive test coverage (>90%) with performance benchmarks
- No complex AI analysis workflows - direct content manipulation only
- **REAL LLM TESTING SUCCESS:**
    - All tests use actual Gemini 2.5 Flash calls (no mocking)
    - Tests only pass when edits are genuinely successful
    - Clear display of OLD vs NEW content for verification
    - Service tracking shows Gemini vs OpenRouter usage and fallback reasons
    - No false positives - failed edits must result in test failures
- **ARCHITECTURE SUCCESS:**
    - Granular editing service operates independently from note creation system
    - Both systems respect Prisma schema relationships and constraints
    - Clear separation of concerns between content editing and note generation
    - Existing note creation workflows remain unaffected
- **NOTE EDITING SUCCESS:**
    - Users can edit BlockNote JSON objects at block, inline content, and style levels
    - BlockNote structure integrity is maintained during all operations
    - Note content can be converted between BlockNote, Markdown, and HTML formats
    - Real-time validation of BlockNote JSON structure and content

---

## II. Planned Tasks & To-Do List

### Phase 1: Core Service Architecture
- [ ] **Task 1:** Design and implement the GranularContentEditingService class
    - *Sub-task 1.1:* Create service interface with proper dependency injection
    - *Sub-task 1.2:* Implement content validation and relationship checking following Prisma schema
    - *Sub-task 1.3:* Add error handling and rollback mechanisms
    - *Sub-task 1.4:* Integrate LangGraph for workflow orchestration
- [ ] **Task 2:** Implement section management operations
    - *Sub-task 2.1:* `add_section(blueprint_id, section_data, parent_section_id=None)` with proper hierarchy
    - *Sub-task 2.2:* `edit_section(section_id, new_data)` maintaining relationships
    - *Sub-task 2.3:* `delete_section(section_id)` with cascade handling for child sections
    - *Sub-task 2.4:* `reorder_sections(blueprint_id, section_order)` updating orderIndex
- [ ] **Task 3:** Implement primitive management operations
    - *Sub-task 3.1:* `add_primitive(blueprint_id, section_id, primitive_data)` with proper linking
    - *Sub-task 3.2:* `edit_primitive(primitive_id, new_data)` maintaining relationships
    - *Sub-task 3.3:* `move_primitive(primitive_id, new_section_id)` updating blueprintSectionId
    - *Sub-task 3.4:* `delete_primitive(primitive_id)` with relationship cleanup

### Phase 2: Mastery Criteria, Questions, and Note Editing
- [ ] **Task 4:** Implement enhanced mastery criterion operations (Multi-Primitive Support)
    - *Sub-task 4.1:* `add_mastery_criterion(primitive_ids: List[str], section_id, criterion_data, uue_stage)` with multi-primitive linking
    - *Sub-task 4.2:* `edit_mastery_criterion(criterion_id, new_data, updated_primitive_ids)` maintaining multi-primitive relationships
    - *Sub-task 4.3:* `reorder_criteria(section_id, criterion_order)` updating weights and complexity scores
    - *Sub-task 4.4:* `delete_mastery_criterion(criterion_id)` with question cleanup and relationship cleanup
    - *Sub-task 4.5:* `validate_criterion_complexity(criterion_id)` ensuring UUE stage matches primitive count
- [ ] **Task 5:** Implement enhanced question management operations (Multi-Concept Integration)
    - *Sub-task 5.1:* `add_question(criterion_id, question_data)` linking to MasteryCriterion with multi-concept testing
    - *Sub-task 5.2:* `edit_question(question_id, new_data)` maintaining multi-concept relationships
    - *Sub-task 5.3:* `bulk_add_questions(criterion_id, questions_list)` for QuestionInstance creation with concept integration
    - *Sub-task 5.4:* `delete_question(question_id)` with proper cleanup
    - *Sub-task 5.5:* `generate_synthesis_questions(criterion_id)` creating questions that test multiple concept integration
- [ ] **Task 6:** Implement enhanced relationship management (Multi-Primitive Criteria)
    - *Sub-task 6.1:* Create MasteryCriterionPrimitive junction table management
    - *Sub-task 6.2:* Implement UUE stage complexity validation (UNDERSTAND: 1-2, USE: 2-4, EXPLORE: 4+)
    - *Sub-task 6.3:* Add conceptual cluster identification and management
    - *Sub-task 6.4:* Implement prerequisite chain validation for multi-primitive criteria
- [ ] **Task 7:** Implement BlockNote JSON editing capabilities
    - *Sub-task 7.1:* Create BlockNote structure validation schemas
    - *Sub-task 7.2:* Implement block-level editing operations (insert, update, delete, move)
    - *Sub-task 7.3:* Add inline content editing (styled text, links, formatting)
    - *Sub-task 7.4:* Implement block nesting and hierarchy management
- [ ] **Task 8:** Implement advanced BlockNote operations
    - *Sub-task 8.1:* Add bulk block operations (move, copy, delete)
    - *Sub-task 8.2:* Implement block type conversion (paragraph, heading, list, etc.)
    - *Sub-task 8.3:* Add style and formatting operations
    - *Sub-task 8.4:* Implement block search and filtering

### Phase 3: Content Validation and Integrity
- [ ] **Task 9:** Implement comprehensive content validation
    - *Sub-task 9.1:* Create validation schemas for each content type following Prisma models
    - *Sub-task 9.2:* Implement relationship integrity checks (foreign key constraints)
    - *Sub-task 9.3:* Add content quality validation (length, format, etc.)
    - *Sub-task 9.4:* Implement circular dependency detection for section hierarchies
    - *Sub-task 9.5:* Add BlockNote JSON structure validation
    - *Sub-task 9.6:* Implement multi-primitive criterion validation (UUE stage complexity matching)
    - *Sub-task 9.7:* Add conceptual overlap detection for criteria
    - *Sub-task 9.8:* Implement prerequisite chain validation for multi-primitive relationships
- [ ] **Task 10:** Add transaction management and rollback
    - *Sub-task 10.1:* Implement database transaction wrapping with Prisma
    - *Sub-task 10.2:* Add rollback mechanisms for failed operations
    - *Sub-task 10.3:* Implement partial rollback for complex operations
- [ ] **Task 11:** Implement LLM-powered content validation
    - *Sub-task 11.1:* Create LLM validation tasks for content quality
    - *Sub-task 11.2:* Implement content improvement suggestions
    - *Sub-task 11.3:* Add validation workflow with LangGraph
    - *Sub-task 11.4:* Add LLM-powered BlockNote editing workflows
- [ ] **Task 12:** Implement enhanced LLM validation for multi-primitive criteria
    - *Sub-task 12.1:* Create LLM tasks for conceptual cluster identification
    - *Sub-task 12.2:* Implement LLM-powered UUE stage complexity validation
    - *Sub-task 12.3:* Add LLM validation for prerequisite chain logic
    - *Sub-task 12.4:* Implement LLM-powered synthesis question generation
- [ ] **Task 13:** Implement modern agentic system resilience patterns
    - *Sub-task 13.1:* Add circuit breaker pattern for LLM service calls
    - *Sub-task 13.2:* Implement retry mechanisms with exponential backoff
    - *Sub-task 13.3:* Add graceful degradation for service failures
    - *Sub-task 13.4:* Implement health checks and service discovery


### Phase 4: Testing and Validation
- [ ] **Task 14:** Create comprehensive test suite with REAL LLM calls
    - *Sub-task 14.1:* Unit tests for each service method (with real LLM integration)
    - *Sub-task 14.2:* Integration tests for content relationships (real LLM calls)
    - *Sub-task 14.3:* Performance tests with large datasets (real LLM calls)
    - *Sub-task 14.4:* Stress tests for concurrent operations (real LLM calls)
    - *Sub-task 14.5:* **REAL LLM TESTING REQUIREMENTS:**
        - Use Gemini 2.5 Flash for all AI operations
        - Fallback to OpenRouter only if Gemini fails
        - NO mocking of LLM responses - real AI calls only
        - Mock only Core API data (blueprints, primitives, etc.)
        - Tests must show OLD vs NEW content for successful edits
        - Tests only pass if edits are actually successful
        - Display service used (Gemini/OpenRouter) and fallback reasons
    - *Sub-task 14.6:* **LangGraph Workflow Testing:**
        - Test complex workflows with state persistence
        - Verify checkpointing and workflow resumption
        - Test conditional routing and parallel execution
        - Validate state management across workflow steps
- [ ] **Task 15:** Enhanced multi-primitive criteria testing
    - *Sub-task 15.1:* Test multi-primitive criterion creation and validation
    - *Sub-task 15.2:* Test UUE stage complexity validation (UNDERSTAND: 1-2, USE: 2-4, EXPLORE: 4+)
    - *Sub-task 15.3:* Test conceptual cluster identification and management
    - *Sub-task 15.4:* Test prerequisite chain validation for multi-primitive relationships
    - *Sub-task 15.5:* Test synthesis question generation with multiple concepts
- [ ] **Task 16:** Unified content editing testing
    - *Sub-task 16.1:* Test blueprint content operations (sections, primitives, criteria, questions)
    - *Sub-task 16.2:* Test BlockNote JSON structure validation and operations
    - *Sub-task 16.3:* Test inline content editing and formatting
    - *Sub-task 16.4:* Test LLM-powered editing workflows for all content types
    - *Sub-task 16.5:* Performance testing with large content sets

### Phase 5: Performance Testing and Optimization
- [ ] **Task 17:** Implement comprehensive performance testing
    - *Sub-task 17.1:* Create performance benchmarks for all operations
    - *Sub-task 17.2:* Test with large content sets (1000+ blocks, 100+ primitives)
    - *Sub-task 17.3:* Implement caching strategies for frequently accessed content
    - *Sub-task 17.4:* Add performance monitoring and metrics collection
- [ ] **Task 18:** Optimize service performance
    - *Sub-task 18.1:* Implement database query optimization
    - *Sub-task 18.2:* Add connection pooling and resource management
    - *Sub-task 18.3:* Implement async operation batching
    - *Sub-task 18.4:* Add performance profiling and bottleneck identification

### Phase 6: API Integration and Documentation
- [ ] **Task 19:** Create API endpoints for the service
    - *Sub-task 19.1:* Design RESTful endpoint structure for unified content editing
    - *Sub-task 19.2:* Implement endpoint handlers with proper error handling
    - *Sub-task 19.3:* Add request/response validation using Pydantic models
    - *Sub-task 19.4:* Implement proper error responses and status codes
- [ ] **Task 20:** Documentation and examples
    - *Sub-task 20.1:* API documentation with examples for all content types
    - *Sub-task 20.2:* Service usage examples showing Prisma schema compliance
    - *Sub-task 20.3:* Performance guidelines and best practices
    - *Sub-task 20.4:* Troubleshooting guide for common issues
- [ ] **Task 21:** LangGraph workflow documentation
    - *Sub-task 21.1:* Workflow usage examples and patterns
    - *Sub-task 21.2:* State management and persistence guide
    - *Sub-task 21.3:* Complex workflow orchestration examples
- [ ] **Task 22:** Architecture and integration documentation
    - *Sub-task 22.1:* Document unified content editing approach
    - *Sub-task 22.2:* Prisma schema relationship documentation
    - *Sub-task 22.3:* Service integration patterns and examples
- [ ] **Task 23:** Unified content editing guide
    - *Sub-task 23.1:* Comprehensive guide for all content types (blueprint, BlockNote, etc.)
    - *Sub-task 23.2:* BlockNote JSON structure and editing guide
    - *Sub-task 23.3:* LLM-powered editing workflow examples for all content types

---

## III. Implementation Details

### Service Architecture with LangGraph Integration
```python
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Optional, List
from prisma import Prisma
from app.models.content_editing_models import *

# State management for complex editing workflows
class EditingState(TypedDict):
    blueprint_id: int
    section_id: Optional[int]
    primitive_id: Optional[str]
    criterion_id: Optional[int]
    question_id: Optional[int]
    note_id: Optional[int]  # NEW: For note editing
    original_content: dict
    edited_content: dict
    validation_errors: List[str]
    llm_service_used: str
    fallback_reason: Optional[str]

class GranularContentEditingService:
    """
    Service for direct manipulation of blueprint content components.
    Operates independently from note creation systems.
    Respects Prisma schema relationships and constraints.
    Includes BlockNote JSON editing capabilities.
    """
    def __init__(self, db_client: Prisma, validation_service, llm_service):
        self.db = db_client
        self.validator = validation_service
        self.llm = llm_service
        self.checkpointer = InMemorySaver()
        
    # Section operations - following BlueprintSection model
    async def add_section(self, blueprint_id: int, section_data: dict, parent_section_id: Optional[int] = None) -> BlueprintSection:
        """Add section with proper hierarchy management"""
        # Validate blueprint_id exists
        # Set depth based on parent_section_id
        # Set orderIndex for proper ordering
        # Create BlueprintSection with all required fields
        
    async def edit_section(self, section_id: int, new_data: dict) -> BlueprintSection:
        """Edit section maintaining relationships"""
        # Update BlueprintSection fields
        # Maintain blueprintId and userId relationships
        
    async def delete_section(self, section_id: int, cascade: bool = True) -> bool:
        """Delete section with cascade handling for child sections"""
        # Check for child sections (BlueprintSection with parentSectionId)
        # Cascade delete if requested
        # Clean up related NoteSection, KnowledgePrimitive, MasteryCriterion
        
    async def reorder_sections(self, blueprint_id: int, section_order: List[int]) -> bool:
        """Reorder sections updating orderIndex"""
        # Update orderIndex for each section
        # Maintain parent-child relationships
        
    # Primitive operations - following KnowledgePrimitive model
    async def add_primitive(self, blueprint_id: int, section_id: int, primitive_data: dict) -> KnowledgePrimitive:
        """Add primitive with proper linking to blueprint and section"""
        # Create KnowledgePrimitive with blueprintId and blueprintSectionId
        # Generate unique primitiveId
        # Set conceptTags and complexity scores
        
    async def edit_primitive(self, primitive_id: int, new_data: dict) -> KnowledgePrimitive:
        """Edit primitive maintaining relationships"""
        # Update KnowledgePrimitive fields
        # Maintain blueprintId and blueprintSectionId links
        
    async def move_primitive(self, primitive_id: int, new_section_id: int) -> bool:
        """Move primitive to different section"""
        # Update blueprintSectionId
        # Validate new section exists and belongs to same blueprint
        
    async def delete_primitive(self, primitive_id: int) -> bool:
        """Delete primitive with relationship cleanup"""
        # Clean up related MasteryCriterion
        # Remove from LearningPathStep if present
        
    # Mastery criterion operations - following MasteryCriterion model
    async def add_mastery_criterion(self, primitive_id: str, section_id: int, criterion_data: dict) -> MasteryCriterion:
        """Add mastery criterion with proper linking"""
        # Create MasteryCriterion with knowledgePrimitiveId and blueprintSectionId
        # Set UUE stage, assessment type, mastery threshold
        
    async def edit_mastery_criterion(self, criterion_id: int, new_data: dict) -> MasteryCriterion:
        """Edit criterion maintaining relationships"""
        # Update MasteryCriterion fields
        # Maintain primitive and section links
        
    async def reorder_criteria(self, section_id: int, criterion_order: List[int]) -> bool:
        """Reorder criteria updating weights"""
        # Update weight field for proper ordering
        
    async def delete_mastery_criterion(self, criterion_id: int) -> bool:
        """Delete criterion with question cleanup"""
        # Clean up related QuestionInstance
        # Remove from UserCriterionMastery
        
    # Question operations - following QuestionInstance model
    async def add_question(self, criterion_id: int, question_data: dict) -> QuestionInstance:
        """Add question linking to MasteryCriterion"""
        # Create QuestionInstance with masteryCriterionId
        # Set questionText, answer, explanation, difficulty
        
    async def edit_question(self, question_id: int, new_data: dict) -> QuestionInstance:
        """Edit question maintaining relationships"""
        # Update QuestionInstance fields
        # Maintain masteryCriterionId link
        
    async def bulk_add_questions(self, criterion_id: int, questions_list: List[dict]) -> List[QuestionInstance]:
        """Bulk add questions for QuestionInstance creation"""
        # Create multiple QuestionInstance records
        # Use transaction for atomicity
        
    async def delete_question(self, question_id: int) -> bool:
        """Delete question with proper cleanup"""
        # Clean up related UserQuestionAnswer
        # Remove from QuestionSetStudySession if present

    # NEW: BlockNote editing operations - following NoteSection model
    async def edit_note_content(self, note_id: int, new_content: dict) -> NoteSection:
        """Edit note content maintaining BlockNote structure"""
        # Update NoteSection contentBlocks (BlockNote JSON)
        # Maintain contentVersion and other fields
        # Validate BlockNote JSON structure
        
    async def add_note_block(self, note_id: int, block_data: dict, position: Optional[int] = None) -> NoteSection:
        """Add new block to note content"""
        # Insert new block into contentBlocks array
        # Handle positioning and nesting
        # Update contentVersion
        
    async def update_note_block(self, note_id: int, block_id: str, new_block_data: dict) -> NoteSection:
        """Update specific block in note content"""
        # Find and update specific block by ID
        # Maintain block structure and relationships
        # Update contentVersion
        
    async def delete_note_block(self, note_id: int, block_id: str) -> NoteSection:
        """Delete specific block from note content"""
        # Remove block and handle children
        # Update contentVersion
        
    async def move_note_block(self, note_id: int, block_id: str, new_position: int, new_parent_id: Optional[str] = None) -> NoteSection:
        """Move block to new position or parent"""
        # Update block position and parent relationships
        # Maintain BlockNote hierarchy
        # Update contentVersion

    # NEW: Modern agentic system resilience patterns
    async def execute_with_circuit_breaker(self, operation: callable, *args, **kwargs):
        """Execute operation with circuit breaker protection"""
        # Circuit breaker implementation for LLM service calls
        # Automatic fallback to OpenRouter if Gemini fails
        # Retry with exponential backoff
        
    async def execute_with_retry(self, operation: callable, max_retries: int = 3, *args, **kwargs):
        """Execute operation with retry mechanism and exponential backoff"""
        # Implement retry logic with exponential backoff
        # Graceful degradation for service failures
        # Health check integration
        
    # NEW: Enhanced multi-primitive criteria operations
    async def create_multi_primitive_criterion(self, primitive_ids: List[str], section_id: int, criterion_data: dict, uue_stage: str) -> dict:
        """Create mastery criterion linking to multiple primitives with UUE stage validation"""
        # Validate UUE stage complexity matches primitive count
        # UNDERSTAND: 1-2 primitives, USE: 2-4 primitives, EXPLORE: 4+ primitives
        # Create MasteryCriterionPrimitive junction table entries
        # Validate prerequisite chains and conceptual relationships
        
    async def validate_criterion_complexity(self, criterion_id: int) -> dict:
        """Validate that criterion complexity matches its UUE stage"""
        # Check primitive count against UUE stage requirements
        # Validate conceptual cluster coherence
        # Ensure prerequisite relationships are logical
        # Return validation results with improvement suggestions
        
    async def generate_synthesis_questions(self, criterion_id: int, question_count: int = 5) -> List[dict]:
        """Generate questions that test integration of multiple concepts"""
        # Analyze linked primitives and their relationships
        # Generate questions requiring synthesis of multiple concepts
        # Ensure questions match UUE stage complexity
        # Return questions with concept integration requirements


# LangGraph workflows for complex editing operations
@task
def validate_content_llm(content: dict, content_type: str) -> dict:
    """Use LLM to validate and improve content quality"""
    # Real LLM call to Gemini 2.5 Flash
    prompt = f"Validate and improve this {content_type} content: {json.dumps(content)}"
    response = llm_service.generate(prompt, model="gemini_2_5_flash")
    return {"validated_content": response, "llm_service": "gemini_2_5_flash"}

@task
def generate_content_llm(description: str, content_type: str) -> dict:
    """Use LLM to generate new content based on description"""
    # Real LLM call to Gemini 2.5 Flash
    prompt = f"Generate {content_type} content based on: {description}"
    response = llm_service.generate(prompt, model="gemini_2_5_flash")
    return {"generated_content": response, "llm_service": "gemini_2_5_flash"}

@task
def bulk_edit_content_llm(content_list: List[dict], edit_instruction: str) -> dict:
    """Use LLM to bulk edit multiple content items"""
    # Real LLM call to Gemini 2.5 Flash
    prompt = f"Apply this edit instruction to all content: {edit_instruction}\nContent: {json.dumps(content_list)}"
    response = llm_service.generate(prompt, model="gemini_2_5_flash")
    return {"edited_content_list": response, "llm_service": "gemini_2_5_flash"}

@task
def edit_blocknote_llm(blocknote_content: dict, edit_instruction: str) -> dict:
    """Use LLM to edit BlockNote content structure"""
    # Real LLM call to Gemini 2.5 Flash
    prompt = f"Edit this BlockNote content: {edit_instruction}\nBlockNote JSON: {json.dumps(blocknote_content)}"
    response = llm_service.generate(prompt, model="gemini_2_5_flash")
    return {"edited_blocknote": response, "llm_service": "gemini_2_5_flash"}

@task
def execute_resilient_llm_call(operation: str, content: dict, fallback_model: str = "openrouter_glm4") -> dict:
    """Execute LLM operation with circuit breaker and fallback"""
    try:
        # Primary call to Gemini 2.5 Flash
        result = await llm_service.generate(operation, content, model="gemini_2_5_flash")
        return {"result": result, "service": "gemini_2_5_flash", "fallback_used": False}
    except Exception as e:
        # Fallback to OpenRouter GLM-4
        result = await llm_service.generate(operation, content, model=fallback_model)
        return {"result": result, "service": fallback_model, "fallback_used": True, "fallback_reason": str(e)}

@task
def validate_multi_primitive_criterion_llm(criterion_data: dict, primitive_ids: List[str], uue_stage: str) -> dict:
    """Use LLM to validate multi-primitive criterion complexity and relationships"""
    # Real LLM call to Gemini 2.5 Flash
    prompt = f"""Validate this mastery criterion for multi-primitive relationships:
    
    Criterion: {json.dumps(criterion_data)}
    Linked Primitives: {primitive_ids}
    UUE Stage: {uue_stage}
    
    Validation Rules:
    - UNDERSTAND: 1-2 primitives (basic comprehension)
    - USE: 2-4 primitives (application and synthesis)  
    - EXPLORE: 4+ primitives (advanced integration and creation)
    
    Check for:
    1. UUE stage complexity matches primitive count
    2. Conceptual cluster coherence
    3. Logical prerequisite relationships
    4. No conceptual overlap or duplication"""
    
    response = llm_service.generate(prompt, model="gemini_2_5_flash")
    return {"validation_result": response, "llm_service": "gemini_2_5_flash"}

@task
def generate_synthesis_questions_llm(criterion_data: dict, primitive_ids: List[str], uue_stage: str) -> dict:
    """Use LLM to generate synthesis questions testing multiple concept integration"""
    # Real LLM call to Gemini 2.5 Flash
    prompt = f"""Generate synthesis questions for this multi-primitive criterion:
    
    Criterion: {json.dumps(criterion_data)}
    Linked Primitives: {primitive_ids}
    UUE Stage: {uue_stage}
    
    Requirements:
    - Questions must test integration of multiple concepts
    - UNDERSTAND: Basic concept connections
    - USE: Application combining 2-4 concepts
    - EXPLORE: Advanced synthesis of 4+ concepts
    - Questions should prepare students for real-world problem-solving"""
    
    response = llm_service.generate(prompt, model="gemini_2_5_flash")
    return {"synthesis_questions": response, "llm_service": "gemini_2_5_flash"}


@entrypoint(checkpointer=InMemorySaver())
def complex_content_workflow(blueprint_id: int, edit_instruction: str, content_type: str):
    """Complex workflow for multi-step content editing with LLM validation"""
    # Step 1: Get original content
    original_content = get_content_from_db(blueprint_id, content_type)
    
    # Step 2: Generate new content using LLM
    generated_result = generate_content_llm(edit_instruction, content_type)
    
    # Step 3: Validate generated content using LLM
    validated_result = validate_content_llm(generated_result["generated_content"], content_type)
    
    # Step 4: Apply changes to database
    updated_content = apply_content_changes(blueprint_id, validated_result["validated_content"])
    
    return {
        "original_content": original_content,
        "edited_content": updated_content,
        "llm_service_used": validated_result["llm_service"],
        "validation_passed": True
    }

@entrypoint(checkpointer=InMemorySaver())
def blocknote_editing_workflow(note_id: int, edit_instruction: str):
    """Complex workflow for BlockNote content editing with LLM"""
    # Step 1: Get original BlockNote content
    original_note = get_note_from_db(note_id)
    
    # Step 2: Use LLM to edit BlockNote structure
    edited_result = edit_blocknote_llm(original_note.contentBlocks, edit_instruction)
    
    # Step 3: Validate BlockNote structure
    validated_result = validate_blocknote_structure(edited_result["edited_blocknote"])
    
    # Step 4: Apply changes to database
    updated_note = apply_note_changes(note_id, validated_result["validated_content"])
    
    return {
        "original_note": original_note,
        "edited_note": updated_note,
        "llm_service_used": edited_result["llm_service"],
        "validation_passed": True
    }

@entrypoint(checkpointer=InMemorySaver())
def resilient_content_editing_workflow(blueprint_id: int, edit_instruction: str, content_type: str):
    """Resilient workflow with circuit breaker, retry, and fallback"""
    # Step 1: Get original content with health check
    original_content = await get_content_with_health_check(blueprint_id, content_type)
    
    # Step 2: Execute resilient LLM operation with fallback
    llm_result = await execute_resilient_llm_call("edit_content", {
        "content": original_content,
        "instruction": edit_instruction
    })
    
    # Step 3: Validate content quality
    validated_result = await validate_content_quality(llm_result["result"])
    
    # Step 4: Apply changes with rollback protection
    updated_content = await apply_changes_with_rollback(
        blueprint_id, 
        llm_result["result"],
        validated_result
    )
    
    return {
        "original_content": original_content,
        "edited_content": updated_content,
        "llm_service_used": llm_result["service"],
        "fallback_used": llm_result["fallback_used"],
        "fallback_reason": llm_result.get("fallback_reason"),
        "resilience_features": ["circuit_breaker", "retry", "fallback"]
    }
```

### Content Validation Strategy
- **Schema validation** using Pydantic models for each content type
- **Relationship validation** ensuring referential integrity
- **Business rule validation** (e.g., section depth limits, primitive uniqueness)
- **Content quality validation** (length, format, required fields)
- **Resilient validation** with circuit breakers and automatic fallbacks

### Performance Considerations
- **Bulk operations** for multiple items (e.g., bulk question creation)
- **Database indexing** optimization for common queries
- **Connection pooling** for database operations
- **Async operations** for non-blocking performance
- **Caching** for frequently accessed content
- **Multi-layer caching** with semantic similarity matching
- **Circuit breaker optimization** for LLM service calls

### Testing Strategy
- **Unit tests**: Individual method testing with real LLM calls (mocked only for Core API data)
- **Integration tests**: Full database operation testing with real AI operations
- **Performance tests**: Large dataset operations, concurrent access, LLM response time tracking
- **Stress tests**: High-volume operations, memory usage, LLM fallback scenarios
- **Edge case tests**: Invalid data, relationship conflicts, rollback scenarios, LLM error handling
- **LangGraph Workflow Testing**: Test complex workflows with state persistence and checkpointing
- **REAL LLM INTEGRATION**: All AI operations use actual Gemini 2.5 Flash calls with OpenRouter fallback
- **Resilience Testing**: Circuit breaker, retry mechanisms, and fallback scenarios
- **Performance Benchmarking**: Caching effectiveness, circuit breaker performance, and scalability testing

---

## IV. Technical Requirements

### Dependencies
- Prisma client for database operations
- Pydantic for data validation
- Async support for performance
- Comprehensive logging for debugging
- Error tracking and monitoring
- **LangGraph Integration:**
    - LangGraph for workflow orchestration and state management
    - Functional API with @task and @entrypoint decorators
    - InMemorySaver for workflow persistence and checkpointing
    - StateGraph for complex workflow definitions
- **LLM Integration Dependencies:**
    - Google Gemini API (gemini-2.5-flash model)
    - OpenRouter API (fallback service)
    - Real-time LLM response tracking
    - Service fallback monitoring and logging
- **Modern Agentic System Dependencies:**
    - Circuit breaker implementation (e.g., pybreaker or custom implementation)
    - Retry mechanisms with exponential backoff
    - Health check and service discovery libraries
    - Multi-layer caching (Redis, in-memory, semantic)

### Database Operations
- Proper transaction management
- Relationship cascade handling
- Index optimization for performance
- Connection pooling and management

### Error Handling
- Detailed error messages for debugging
- Proper HTTP status codes
- Rollback mechanisms for failed operations
- Logging of all operations and errors

---

## V. Success Metrics

### Performance Targets
- **Simple operations** (add/edit single item): <200ms
- **Complex operations** (bulk operations): <2s for 100 items
- **Database queries**: <50ms average
- **Memory usage**: <100MB for large operations

### Quality Targets
- **Test coverage**: >90%
- **Error rate**: <1% for valid inputs
- **Data integrity**: 100% relationship consistency
- **API response time**: <500ms for all endpoints
- **LLM Testing Quality:**
    - **100% real LLM calls** - no mocked AI responses
    - **Zero false positives** - tests only pass on genuine success
    - **Complete content verification** - OLD vs NEW content display
    - **Service transparency** - clear Gemini/OpenRouter usage tracking
    - **Fallback accountability** - documented reasons for service switches
- **Resilience Quality:**
    - **Circuit breaker effectiveness**: <5% of calls trigger circuit breaker
    - **Fallback success rate**: >95% successful fallback to OpenRouter
    - **Retry efficiency**: <3 retries on average for failed operations

### Scalability Targets
- **Concurrent users**: Support 100+ simultaneous operations
- **Large datasets**: Handle 10,000+ items efficiently
- **Memory efficiency**: Linear scaling with dataset size
- **Database performance**: Maintain performance with 1M+ records

---

## VI. Risk Assessment

### Technical Risks
- **Database performance** with large datasets
- **Concurrent operation conflicts** leading to data inconsistency
- **Complex relationship validation** causing performance bottlenecks
- **Transaction rollback complexity** for multi-step operations
- **LLM Integration Risks:**
    - **Gemini API failures** requiring OpenRouter fallback
    - **LLM response quality** affecting edit success rates
    - **API rate limiting** during high-volume testing
    - **LLM response parsing** failures leading to test inconsistencies

### Mitigation Strategies
- **Performance testing** with realistic datasets
- **Optimistic locking** for concurrent operations
- **Efficient validation algorithms** with early termination
- **Comprehensive testing** of rollback scenarios
- **LLM Risk Mitigation:**
    - **Robust fallback system** with OpenRouter integration
    - **LLM response validation** to ensure edit quality
    - **Rate limiting management** with exponential backoff
    - **Comprehensive error handling** for LLM failures
    - **Real-time monitoring** of LLM service health

---

## VII. Next Steps After Completion

1. **Performance monitoring** in production environment
2. **User feedback collection** on usability and performance
3. **Advanced features** like content templates and bulk import
4. **Integration** with existing AI services for content enhancement
5. **Analytics** on content editing patterns and usage
6. **LLM Performance Optimization:**
    - **Gemini vs OpenRouter** performance comparison
    - **Edit success rate** analysis by LLM service
    - **Response time optimization** for different content types
    - **Fallback frequency** monitoring and optimization
7. **System Integration:**
    - **Note creation system** integration testing
    - **Blueprint lifecycle** synchronization
    - **User experience** optimization across both systems
    - **Performance benchmarking** between granular editing and note creation
8. **BlockNote Ecosystem Integration:**
    - **Real-time collaboration** features using Yjs
    - **Advanced block types** and custom schemas
    - **Export/import** to other note-taking platforms
    - **Mobile optimization** for BlockNote editing
    - **Offline editing** capabilities with sync

---

## VII. Multi-Primitive Criteria Enhancement Impact

### **Enhanced System Capabilities**
The sprint has been updated to support the new many-to-many relationship system between MasteryCriteria and KnowledgePrimitives, enabling:

1. **Sophisticated Learning Progression**
   - **UNDERSTAND Stage**: 1-2 primitives for basic comprehension
   - **USE Stage**: 2-4 primitives for application and synthesis
   - **EXPLORE Stage**: 4+ primitives for advanced integration and creation

2. **Enhanced Content Editing Operations**
   - Multi-primitive criterion creation and management
   - UUE stage complexity validation
   - Conceptual cluster identification and management
   - Prerequisite chain validation for complex relationships

3. **Advanced Question Generation**
   - Synthesis questions testing multiple concept integration
   - Questions that prepare students for real-world problem-solving
   - Complexity-appropriate assessment generation

### **New Service Methods Added**
- `create_multi_primitive_criterion()` - Multi-primitive criterion creation
- `validate_criterion_complexity()` - UUE stage complexity validation
- `generate_synthesis_questions()` - Multi-concept integration questions

### **New LangGraph Tasks Added**
- `validate_multi_primitive_criterion_llm()` - LLM-powered complexity validation
- `generate_synthesis_questions_llm()` - LLM-powered synthesis question generation

### **Enhanced Testing Requirements**
- Multi-primitive criterion validation testing
- UUE stage complexity validation testing
- Conceptual cluster management testing
- Prerequisite chain validation testing
- Synthesis question generation testing

---

**Sprint Status:** [To be determined after Antonio's review]

### BlockNote Integration and Models
```python
from typing import TypedDict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, validator

# BlockNote Block Structure (following BlockNote schema)
class BlockNoteBlock(BaseModel):
    id: str
    type: str
    props: dict = Field(default_factory=dict)
    content: Optional[Union[List['InlineContent'], 'TableContent']] = None
    children: List['BlockNoteBlock'] = Field(default_factory=list)

# Inline Content Types
class StyledText(BaseModel):
    type: Literal["text"] = "text"
    text: str
    styles: dict = Field(default_factory=dict)  # bold, italic, underline, etc.

class Link(BaseModel):
    type: Literal["link"] = "link"
    content: List[StyledText]
    href: str

class InlineContent(BaseModel):
    __root__: Union[StyledText, Link]

# Table Content
class TableContent(BaseModel):
    type: Literal["tableContent"] = "tableContent"
    rows: List[dict] = Field(default_factory=list)

# Note Editing Models
class NoteEditRequest(BaseModel):
    note_id: int
    edit_type: Literal["content", "block_add", "block_update", "block_delete", "block_move", "format_convert"]
    edit_data: dict
    position: Optional[int] = None
    parent_block_id: Optional[str] = None
    target_format: Optional[str] = None

class NoteEditResponse(BaseModel):
    success: bool
    original_content: dict
    edited_content: dict
    llm_service_used: str
    fallback_reason: Optional[str] = None
    validation_errors: List[str] = Field(default_factory=list)

# BlockNote Validation
class BlockNoteValidator:
    """Validates BlockNote JSON structure and content"""
    
    @staticmethod
    def validate_block_structure(block: dict) -> bool:
        """Validate individual block structure"""
        required_fields = ["id", "type"]
        return all(field in block for field in required_fields)
    
    @staticmethod
    def validate_block_hierarchy(blocks: List[dict]) -> bool:
        """Validate block hierarchy and relationships"""
        block_ids = set()
        parent_refs = set()
        
        for block in blocks:
            if not BlockNoteValidator.validate_block_structure(block):
                return False
            
            if block["id"] in block_ids:
                return False  # Duplicate ID
            block_ids.add(block["id"])
            
            # Check children recursively
            if "children" in block and block["children"]:
                if not BlockNoteValidator.validate_block_hierarchy(block["children"]):
                    return False
        
        return True
    
    @staticmethod
    def validate_inline_content(content: List[dict]) -> bool:
        """Validate inline content structure"""
        for item in content:
            if "type" not in item:
                return False
            if item["type"] == "text" and "text" not in item:
                return False
            if item["type"] == "link" and ("content" not in item or "href" not in item):
                return False
        return True

# BlockNote Operations
class BlockNoteOperations:
    """Core operations for manipulating BlockNote content"""
    
    @staticmethod
    def insert_block(blocks: List[dict], new_block: dict, position: int) -> List[dict]:
        """Insert block at specific position"""
        if position < 0 or position > len(blocks):
            raise ValueError("Invalid position")
        
        result = blocks.copy()
        result.insert(position, new_block)
        return result
    
    @staticmethod
    def update_block(blocks: List[dict], block_id: str, updates: dict) -> List[dict]:
        """Update specific block by ID"""
        def update_recursive(block_list: List[dict]) -> List[dict]:
            result = []
            for block in block_list:
                if block["id"] == block_id:
                    updated_block = {**block, **updates}
                    if "children" in updated_block and updated_block["children"]:
                        updated_block["children"] = update_recursive(updated_block["children"])
                    result.append(updated_block)
                else:
                    if "children" in block and block["children"]:
                        block["children"] = update_recursive(block["children"])
                    result.append(block)
            return result
        
        return update_recursive(blocks)
    
    @staticmethod
    def delete_block(blocks: List[dict], block_id: str) -> List[dict]:
        """Delete block by ID"""
        def delete_recursive(block_list: List[dict]) -> List[dict]:
            result = []
            for block in block_list:
                if block["id"] != block_id:
                    if "children" in block and block["children"]:
                        block["children"] = delete_recursive(block["children"])
                    result.append(block)
            return result
        
        return delete_recursive(blocks)
    
    @staticmethod
    def move_block(blocks: List[dict], block_id: str, new_position: int, new_parent_id: Optional[str] = None) -> List[dict]:
        """Move block to new position or parent"""
        # Find and remove block
        block_to_move = None
        blocks_without_target = BlockNoteOperations.delete_block(blocks, block_id)
        
        # Find the block that was removed
        def find_block(block_list: List[dict]) -> Optional[dict]:
            for block in block_list:
                if block["id"] == block_id:
                    return block
                if "children" in block and block["children"]:
                    found = find_block(block["children"])
                    if found:
                        return found
            return None
        
        # Find in original blocks
        block_to_move = find_block(blocks)
        if not block_to_move:
            raise ValueError(f"Block with ID {block_id} not found")
        
        if new_parent_id:
            # Move to new parent
            return BlockNoteOperations.insert_block_into_parent(blocks_without_target, block_to_move, new_parent_id, new_position)
        else:
            # Move within same level
            return BlockNoteOperations.insert_block(blocks_without_target, block_to_move, new_position)
    
    @staticmethod
    def insert_block_into_parent(blocks: List[dict], block_to_insert: dict, parent_id: str, position: int) -> List[dict]:
        """Insert block into specific parent block"""
        def insert_recursive(block_list: List[dict]) -> List[dict]:
            result = []
            for block in block_list:
                if block["id"] == parent_id:
                    # Found parent, insert block into children
                    children = block.get("children", [])
                    children = BlockNoteOperations.insert_block(children, block_to_insert, position)
                    block["children"] = children
                    result.append(block)
                else:
                    if "children" in block and block["children"]:
                        block["children"] = insert_recursive(block["children"])
                    result.append(block)
            return result
        
        return insert_recursive(blocks)
```
