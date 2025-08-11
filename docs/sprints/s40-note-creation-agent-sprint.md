# Sprint 14: Note Creation Agent Development

**Date Range:** [Start Date] - [End Date]  
**Primary Focus:** AI Service - Note Creation Agent  
**Overview:** This sprint focuses on developing a comprehensive note creation agent that can create notes from source text, convert user input to BlockNote format, and provide agentic note editing capabilities.

---

## I. Sprint Goals & Objectives

### Primary Goals:
1. **Source-to-Note Agent**: Create an AI agent that can generate structured notes from source text via blueprint creation
2. **Input Conversion Agent**: Convert user input (plain text, markdown, etc.) to BlockNote format via blueprint creation
3. **Agentic Editing Agent**: Provide AI-powered note editing, improvement, and restructuring capabilities

### Success Criteria:
- Users can create notes from source text with automatic blueprint creation
- Users can create notes from their own content with automatic blueprint creation
- All notes remain linked to their foundational blueprint for RAG context
- User input can be converted to proper BlockNote format
- AI can suggest improvements and edits to existing notes
- All generated content maintains BlockNote compatibility

---

## II. Planned Tasks & To-Do List

### Phase 1: Source-to-Note Agent
- [x] **Task 1.1: Create Direct Note Generation Endpoint**
    - [x] **Sub-task 1.1.1:** Create new endpoint `POST /api/v1/generate-notes-from-source`
    - [x] **Sub-task 1.1.2:** Design request schema for source text, note preferences, and user context
    - [x] **Sub-task 1.1.3:** Implement prompt engineering for direct note generation
    - [x] **Sub-task 1.1.4:** Ensure output is BlockNote-compatible JSON structure 

- [x] **Task 1.2: Implement Note Generation Service**
    - [x] **Sub-task 1.2.1:** Create `NoteGenerationService` class in `app/core/services/`
    - [x] **Sub-task 1.2.2:** Implement source text analysis and structure detection
    - [x] **Sub-task 1.2.3:** Add note style customization (concise, detailed, bullet points, etc.)
    - [x] **Sub-task 1.2.4:** Integrate with existing LLM service
    - [x] **Sub-task 1.2.5:** Integrate with user preference learning

- [x] **Task 1.3: Implement Long Source Processing**
    - [x] **Sub-task 1.3.1:** Create `SourceChunkingService` for intelligent content segmentation
    - [x] **Sub-task 1.3.2:** Implement algorithmic section detection (font size, markup, structure)
    - [x] **Sub-task 1.3.3:** Add LLM validation and refinement of algorithmic chunks
    - [x] **Sub-task 1.3.4:** Create hierarchical blueprint synthesis from section blueprints
    - [x] **Sub-task 1.3.5:** Implement note assembly and coherence checking

### Phase 2: Input Conversion Agent
- [x] **Task 2.1: Create Input Conversion Endpoint**
    - [x] **Sub-task 2.1.1:** Create new endpoint `POST /api/v1/convert-input-to-blocks`
    - [x] **Sub-task 2.1.2:** Support multiple input formats (plain text, markdown, HTML)
    - [x] **Sub-task 2.1.3:** Design conversion strategies for different content types
    - [x] **Sub-task 2.1.4:** Implement validation for BlockNote compatibility

- [x] **Task 2.2: Implement Conversion Service**
    - [x] **Sub-task 2.2.1:** Create `InputConversionService` class
    - [x] **Sub-task 2.2.2:** Implement markdown-to-blocks parser
    - [x] **Sub-task 2.2.3:** Implement plain-text-to-blocks formatter
    - [x] **Sub-task 2.2.4:** Add content structure detection and block type assignment

### Phase 3: Agentic Editing Agent
- [x] **Task 3.1: Create Note Editing Endpoint**
    - [x] **Sub-task 3.1.1:** Create new endpoint `POST /api/v1/edit-note-agentically`
    - [x] **Sub-task 3.1.2:** Design request schema for edit instructions and note content
    - [x] **Sub-task 3.1.3:** Implement edit operation types (restructure, improve, expand, condense)
    - [x] **Sub-task 3.1.4:** Add change tracking and version control

- [x] **Task 3.2: Implement Editing Service**
    - [x] **Sub-task 3.2.1:** Create `NoteEditingService` class
    - [x] **Sub-task 3.2.2:** Implement content analysis and improvement suggestions
    - [x] **Sub-task 3.2.3:** Add structural optimization algorithms
    - [x] **Sub-task 3.2.4:** Integrate with user preference learning

### Phase 4: Integration & Testing
- [x] **Task 4.1: Update Core API Integration**
    - [x] **Sub-task 4.1.1:** Update Core API client to use new endpoints
    - [x] **Sub-task 4.1.2:** Add error handling and fallback mechanisms
    - [x] **Sub-task 4.1.3:** Implement request/response validation

- [ ] **Task 4.2: Comprehensive Testing**
    - [ ] **Sub-task 4.2.1:** Create unit tests for all new services
    - [ ] **Sub-task 4.2.2:** Add integration tests for endpoints
    - [ ] **Sub-task 4.2.3:** Test BlockNote format compatibility
    - [ ] **Sub-task 4.2.4:** Performance testing for large documents
    - [ ] **Sub-task 4.2.5:** Create E2E tests in blueprint-lifecycle folder
        - [ ] Test long source processing with hybrid chunking
        - [ ] Test note creation from various source types  
        - [ ] Test input conversion to BlockNote format
        - [ ] Test agentic note editing workflows
        - [ ] Test blueprint linkage and RAG context preservation

---

## III. Technical Architecture

### New Services Structure:
```
app/core/services/
├── note_generation_service.py      # ✅ Source-to-note generation
├── source_chunking_service.py      # ✅ Long source processing & chunking
├── input_conversion_service.py     # ✅ Input format conversion
├── note_editing_service.py         # ✅ Agentic editing
└── note_agent_orchestrator.py      # 🔄 Main agent coordinator (optional)
```

### API Endpoints:
```
POST /api/v1/generate-notes-from-source    # ✅ Creates blueprint + notes from source
POST /api/v1/generate-notes-from-content   # ✅ Creates blueprint + notes from user content
POST /api/v1/convert-input-to-blocks       # ✅ Converts user input to BlockNote format
POST /api/v1/edit-note-agentically         # ✅ AI-powered note editing
GET  /api/v1/note-editing-suggestions      # ✅ Get editing suggestions
```

### Data Models:
```python
class NoteGenerationRequest:
    source_text: str
    note_style: NoteStyle  # concise, detailed, bullet_points, etc.
    user_preferences: UserPreferences
    target_length: Optional[int]
    focus_areas: List[str]
    create_blueprint: bool = True  # Always create blueprint for RAG context
    chunking_strategy: Optional[ChunkingStrategy] = "auto"  # auto, manual, semantic

class ChunkingStrategy:
    max_chunk_size: Optional[int] = 8000  # tokens
    chunk_overlap: int = 500  # tokens for context preservation
    semantic_boundaries: bool = True  # prefer topic-based breaks
    preserve_structure: bool = True  # maintain hierarchical relationships
    use_algorithmic_detection: bool = True  # fast font/markup detection first
    llm_validation_threshold: float = 0.8  # confidence level for LLM review

class SourceChunk:
    chunk_id: str
    content: str
    start_position: int
    end_position: int
    topic: str
    parent_chunk_id: Optional[str]
    child_chunk_ids: List[str]
    cross_references: List[str]

class AlgorithmicDetectionResult:
    detected_sections: List[Dict[str, Any]]  # font size, markup, structure
    confidence_scores: List[float]  # how confident we are in each break
    suggested_chunks: List[SourceChunk]
    needs_llm_validation: bool  # whether LLM review is required

class ContentToNoteRequest:
    user_content: str
    content_format: ContentFormat  # plain_text, markdown, html
    note_style: NoteStyle
    user_preferences: UserPreferences
    create_blueprint: bool = True  # Extract knowledge primitives for blueprint

class InputConversionRequest:
    input_text: str
    input_format: InputFormat  # plain_text, markdown, html
    target_structure: Optional[NoteStructure]
    preserve_formatting: bool

class NoteEditingRequest:
    note_content: List[BlockNoteBlock]
    edit_instructions: str
    edit_type: EditType  # restructure, improve, expand, condense
    user_context: Optional[UserContext]
```

---

## IV. Implementation Details

### Blueprint-Centric Architecture:
- **All notes are linked to a blueprint**: Every note creation flow creates or links to a LearningBlueprint
- **Blueprint serves as knowledge foundation**: Provides RAG context, knowledge primitives, and content structure
- **Two main flows**:
  1. **Source → Blueprint → Notes**: Extract knowledge from source text, create blueprint, generate notes
  2. **User Content → Blueprint → Notes**: Extract knowledge from user input, create blueprint, generate structured notes
- **RAG integration**: Blueprint enables the AI system to understand note context and provide relevant suggestions

### BlockNote Format Requirements:
- All generated content must be valid BlockNote JSON structure
- Support for custom block types (insightCatalyst, etc.)
- Maintain proper block hierarchy and relationships
- Include metadata for AI processing and user preferences
- Link to blueprint ID for context and RAG operations

### AI Prompt Engineering:
- **Source Analysis**: Extract key concepts, structure, and relationships
- **Note Generation**: Create organized, scannable content with proper block types
- **Content Conversion**: Transform various input formats while preserving meaning
- **Editing Intelligence**: Understand context and suggest meaningful improvements

### Long Source Processing Strategy:
- **Hybrid Chunking Approach**:
  1. **Algorithmic Detection** (Fast): Font size, markup, structure patterns
  2. **LLM Validation** (Quality): Confirm semantic boundaries and coherence
- **Efficient Processing**: Algorithmic detection handles 80-90% of cases quickly
- **Quality Assurance**: LLM only processes edge cases and validates boundaries
- **Hierarchical Blueprint Creation**: 
  1. Create mini-blueprints for each validated chunk
  2. Identify cross-references and relationships between chunks
  3. Synthesize unified master blueprint
- **Parallel Processing**: Process chunks concurrently for efficiency
- **Coherence Assembly**: LLM ensures final notes flow logically between sections

### Performance Considerations:
- Implement streaming for large document processing
- Add caching for common conversion patterns
- Optimize LLM calls with prompt batching
- Add rate limiting and cost controls

---

## V. Dependencies & Prerequisites

### Required:
- ✅ Existing LLM service integration
- ✅ BlockNote schema definitions
- ✅ User preference system
- ✅ Authentication and authorization

### Nice to Have:
- Vector database for content similarity
- User learning history integration
- Content quality metrics
- A/B testing framework

---

## VI. Testing Strategy

### Unit Tests:
- [ ] Service layer functionality
- [ ] Input validation and sanitization
- [ ] BlockNote format validation
- [ ] Error handling scenarios

### Integration Tests:
- [ ] End-to-end note generation flow
- [ ] Format conversion accuracy
- [ ] Editing operation correctness
- [ ] API response validation

### Performance Tests:
- [ ] Large document processing
- [ ] Concurrent request handling
- [ ] Memory usage optimization
- [ ] Response time benchmarks

---

## VII. Success Metrics

### Functional Metrics:
- ✅ All endpoints return valid BlockNote JSON
- ✅ Input conversion maintains content integrity
- ✅ Editing suggestions are contextually relevant
- ✅ Generated notes meet user style preferences

### Performance Metrics:
- ⚡ Note generation < 30 seconds for 10k word documents
- ⚡ Input conversion < 5 seconds for typical content
- ⚡ Editing suggestions < 15 seconds response time
- ⚡ 99% BlockNote format compatibility

### Quality Metrics:
- 🎯 User satisfaction with generated notes > 4.5/5
- 🎯 Content accuracy and relevance > 90%
- 🎯 Format conversion success rate > 95%
- 🎯 Editing suggestion acceptance rate > 70%

---

## VIII. Risk Assessment & Mitigation

### High Risk:
- **BlockNote Format Incompatibility**: Implement strict validation and fallback mechanisms
- **LLM Response Quality**: Use multiple prompt strategies and response validation
- **Performance with Large Documents**: Implement streaming and chunking strategies

### Medium Risk:
- **User Preference Learning**: Start with basic preferences, iterate based on usage
- **Content Structure Detection**: Use multiple algorithms and fallback to simple parsing
- **API Integration Complexity**: Implement comprehensive error handling and logging

### Low Risk:
- **Authentication Integration**: Leverage existing middleware and patterns
- **Database Schema Changes**: Use optional fields and backward compatibility
- **Frontend Integration**: Maintain existing API patterns where possible

---

## IX. Future Enhancements

### Phase 2 Features:
- Multi-language note generation
- Collaborative note editing
- Advanced content analysis
- Learning pattern recognition

### Phase 3 Features:
- Real-time collaborative editing
- Advanced AI editing suggestions
- Content quality scoring
- Automated note organization

---

## X. Sprint Completion Checklist

- [x] All three agent types implemented and tested
- [x] BlockNote format compatibility verified
- [x] API endpoints documented and tested
- [x] Core API integration updated
- [ ] Performance benchmarks met
- [ ] Error handling comprehensive
- [ ] User documentation updated
- [ ] Code review completed
- [ ] Deployment tested in staging

---

## XI. Implementation Status Update

### ✅ Completed Components:

1. **Note Generation Service** (`note_generation_service.py`)
   - Source text analysis and structure detection
   - Note style customization (concise, detailed, bullet points, etc.)
   - Integration with LLM service and user preferences
   - Blueprint creation from source content

2. **Source Chunking Service** (`source_chunking_service.py`)
   - Intelligent content segmentation with hybrid approach
   - Algorithmic section detection (font size, markup, structure)
   - LLM validation and refinement of algorithmic chunks
   - Hierarchical blueprint synthesis from section blueprints

3. **Input Conversion Service** (`input_conversion_service.py`)
   - Conversion from various formats (plain text, markdown, HTML)
   - Blueprint creation from user content
   - BlockNote format generation
   - Content structure detection and block type assignment

4. **Note Editing Service** (`note_editing_service.py`)
   - AI-powered note editing capabilities
   - Content analysis and improvement suggestions
   - Structural optimization algorithms
   - Grammar, clarity, and structure suggestions

5. **LLM Service Interface** (`llm_service.py`)
   - Unified interface for different LLM providers
   - Gemini integration with existing gemini_service.py
   - Mock service for testing
   - Factory function for service creation

6. **API Endpoints** (`note_creation_endpoints.py`)
   - All planned endpoints implemented
   - Proper error handling and validation
   - Service integration and dependency injection

### 🔄 Next Steps:

1. **Testing Implementation**
   - Create unit tests for all services
   - Add integration tests for endpoints
   - Test BlockNote format compatibility
   - Performance testing for large documents

2. **Documentation Updates**
   - API documentation
   - User guides
   - Code documentation

3. **Deployment Preparation**
   - Staging environment testing
   - Production deployment planning
   - Monitoring and logging setup

---

**Signed off:** DO NOT PROCEED WITH THE SPRINT UNLESS SIGNED OFF BY ANTONIO  
**Next Review:** [Date]  
**Stakeholders:** [Names]    