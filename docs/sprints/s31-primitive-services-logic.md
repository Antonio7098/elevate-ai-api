# Sprint 31: Primitive-Centric AI API - Core Services & Logic

**Signed off** DO NOT PROCEED UNLESS SIGNED OFF BY ANTONIO
**Date Range:** [Start Date] - [End Date]
**Primary Focus:** AI API - Core Service Implementation & Business Logic
**Overview:** This sprint focuses on implementing core services that generate Core API Prisma-compatible primitive data during blueprint creation. This includes enhancing blueprint deconstruction to produce KnowledgePrimitive and MasteryCriterion data matching the exact Prisma schema, and creating transformation services for seamless Core API integration.

---

## I. Planned Tasks & To-Do List

- [ ] **Task 1: Enhanced Blueprint Creation with Primitive Generation**
    - *Sub-task 1.1:* Update `app/core/deconstruction.py` to generate primitives with mastery criteria
        ```python
        async def generate_primitives_with_criteria_from_source(
            source_content: str, 
            source_type: str,
            user_preferences: Optional[Dict] = None
        ) -> LearningBlueprint:
            """
            Generate blueprint with primitives and mastery criteria from source content.
            
            Uses LLM to:
            1. Analyze source content and identify discrete knowledge units
            2. Generate mastery criteria for each primitive during creation
            3. Assign UEE levels and importance weights
            4. Ensure comprehensive coverage across all source content
            """
        ```
    - *Sub-task 1.2:* Enhance LLM prompts to generate mastery criteria alongside primitives
    - *Sub-task 1.3:* Add user preference integration (learning style, focus areas)
    - *Sub-task 1.4:* Create primitive quality validation during generation
    - *Sub-task 1.5:* Add UEE level distribution optimization

- [ ] **Task 2: Core API Compatible Mastery Criteria Generation**
    - *Sub-task 2.1:* Implement `generate_prisma_compatible_criteria()` function
        ```python
        async def generate_prisma_compatible_criteria(
            primitive_content: str, 
            primitive_type: str,
            context: Dict[str, Any]
        ) -> List[Dict[str, Any]]:
            """
            Generate mastery criteria matching Core API Prisma schema exactly.
            
            Returns criteria with:
            - criterionId: str (unique identifier)
            - title: str (criterion title)
            - description: Optional[str] (criterion description)
            - ueeLevel: "UNDERSTAND" | "USE" | "EXPLORE"
            - weight: float (importance weight)
            - difficulty: float (difficulty level)
            - isRequired: bool (required for progression)
            """
        ```
    - *Sub-task 2.2:* Create UEE level mapping to Prisma enum values ("UNDERSTAND", "USE", "EXPLORE")
    - *Sub-task 2.3:* Implement float weight and difficulty assignment matching Prisma schema
    - *Sub-task 2.4:* Add criterionId generation strategy for Core API compatibility
    - *Sub-task 2.5:* Create validation against actual Prisma MasteryCriterion model

- [ ] **Task 3: Core API Compatible Question Generation**
    - *Sub-task 3.1:* Implement `generate_core_api_questions()` service
        ```python
        async def generate_core_api_questions(
            primitiveId: str, 
            criterionId: str, 
            criterion_data: Dict[str, Any],
            count: int
        ) -> List[Dict[str, Any]]:
            """
            Generate questions with Core API compatible references.
            
            Returns questions with:
            - criterionId: str (matches Prisma MasteryCriterion.criterionId)
            - primitiveId: str (matches Prisma KnowledgePrimitive.primitiveId)
            - ueeLevel: "UNDERSTAND" | "USE" | "EXPLORE"
            - Question content optimized for Core API Question model
            """
        ```
    - *Sub-task 3.2:* Create question type mapping based on Prisma ueeLevel enum
    - *Sub-task 3.3:* Implement question referencing using actual Core API IDs
    - *Sub-task 3.4:* Add validation against Core API Question model requirements
    - *Sub-task 3.5:* Create question batch generation for multiple criteria efficiently

- [ ] **Task 4: Question-Criterion Mapping Service**
    - *Sub-task 4.1:* Create `app/core/criterion_mapping.py` service
        ```python
        async def map_questions_to_criteria(
            questions: List[QuestionDto], 
            criteria: List[MasteryCriterionDto]
        ) -> Dict[str, str]:
            """
            Intelligently map generated questions to appropriate mastery criteria.
            
            Uses semantic analysis to:
            1. Analyze question cognitive requirements
            2. Match to criterion learning objectives
            3. Ensure balanced coverage across all criteria
            4. Optimize for spaced repetition effectiveness
            """
        ```
    - *Sub-task 4.2:* Implement semantic similarity analysis for question-criterion matching
    - *Sub-task 4.3:* Create coverage optimization algorithms
    - *Sub-task 4.4:* Add mapping validation and quality assurance
    - *Sub-task 4.5:* Implement automated remapping for improved coverage

- [ ] **Task 5: Core API Prisma Data Synchronization Services**
    - *Sub-task 5.1:* Create `app/core/prisma_sync.py` for exact Prisma schema compliance
        ```python
        async def format_for_core_api_storage(
            blueprint: LearningBlueprint,
            userId: int,
            blueprintId: int
        ) -> Dict[str, Any]:
            """Format blueprint data for exact Prisma KnowledgePrimitive/MasteryCriterion storage."""
            
        async def generate_prisma_primitive_records(
            blueprint: LearningBlueprint,
            userId: int,
            blueprintId: int
        ) -> List[Dict[str, Any]]:
            """Generate Core API KnowledgePrimitive records matching Prisma schema exactly."""
        ```
    - *Sub-task 5.2:* Implement ID generation for primitiveId and criterionId (String, unique)
    - *Sub-task 5.3:* Add Prisma relationship mapping (userId, blueprintId references)
    - *Sub-task 5.4:* Create TrackingIntensity enum mapping (DENSE, NORMAL, SPARSE)
    - *Sub-task 5.5:* Implement Prisma field validation before Core API transmission

- [ ] **Task 6: AI Service Integration with Prisma Schema Compliance**
    - *Sub-task 6.1:* Create LLM prompts for Core API Prisma-compatible primitive generation
        ```python
        PRISMA_COMPATIBLE_BLUEPRINT_PROMPT = """
        Analyze the following educational content and create primitives matching Core API Prisma schema.
        
        For each KnowledgePrimitive, generate:
        1. title: str (clear, concise)
        2. description: Optional[str] (detailed description)
        3. primitiveType: str ("fact", "concept", "process")
        4. difficultyLevel: str ("beginner", "intermediate", "advanced")
        5. estimatedTimeMinutes: Optional[int]
        
        For each MasteryCriterion, generate:
        1. title: str (criterion title)
        2. description: Optional[str] (what needs to be mastered)
        3. ueeLevel: "UNDERSTAND" | "USE" | "EXPLORE"
        4. weight: float (importance weight)
        5. difficulty: float (difficulty level)
        6. isRequired: bool (required for progression)
        
        Source Content: {content}
        Source Type: {source_type}
        """
        ```
    - *Sub-task 6.2:* Implement cost-optimized LLM call strategies
    - *Sub-task 6.3:* Add response caching and intelligent reuse
    - *Sub-task 6.4:* Create batch processing for multiple primitives
    - *Sub-task 6.5:* Implement quality gates and automatic retry logic

- [ ] **Task 7: Performance & Scalability**
    - *Sub-task 7.1:* Implement async processing for large blueprint analysis
    - *Sub-task 7.2:* Add result caching with intelligent invalidation
    - *Sub-task 7.3:* Create batch processing APIs for bulk operations
    - *Sub-task 7.4:* Implement progress tracking for long-running operations
    - *Sub-task 7.5:* Add resource usage monitoring and optimization

- [ ] **Task 8: Error Handling & Resilience**
    - *Sub-task 8.1:* Implement comprehensive error handling for LLM failures
    - *Sub-task 8.2:* Add graceful degradation for partial results
    - *Sub-task 8.3:* Create retry mechanisms with exponential backoff
    - *Sub-task 8.4:* Implement circuit breakers for external service failures
    - *Sub-task 8.5:* Add detailed logging and monitoring for troubleshooting

---

## II. Agent's Implementation Summary & Notes

*Instructions for AI Agent (Cascade): For each planned task you complete from Section I, please provide a summary below. If multiple tasks are done in one go, you can summarize them together but reference the task numbers.*

**Regarding Task 1: [To be filled during implementation]**
* **Summary of Implementation:**
    * [Agent will describe the primitive extraction service implementation]
* **Key Files Modified/Created:**
    * `app/core/primitive_generation.py`
* **Notes/Challenges Encountered (if any):**
    * [Agent will note any implementation challenges]

---

## III. Overall Sprint Summary & Review (To be filled out by Antonio after work is done)

**Sprint Completion Status:** [To be filled]
**Key Deliverables Achieved:** [To be filled]
**Technical Debt Introduced:** [To be filled]
**Next Sprint Preparation:** [To be filled]

---

## IV. Enterprise Readiness Checklist

- [ ] **Service Architecture**
    - [ ] Services follow single responsibility principle
    - [ ] Proper dependency injection and inversion
    - [ ] Clean separation of concerns
    - [ ] Testable and mockable interfaces

- [ ] **Performance & Scalability**
    - [ ] Async/await patterns used throughout
    - [ ] Caching strategy implemented and tested
    - [ ] Batch processing capabilities
    - [ ] Resource usage monitoring
    - [ ] Load testing completed

- [ ] **Error Handling & Resilience**
    - [ ] Comprehensive exception handling
    - [ ] Graceful degradation patterns
    - [ ] Circuit breaker implementation
    - [ ] Retry mechanisms with backoff
    - [ ] Dead letter queue for failed operations

- [ ] **AI/LLM Integration**
    - [ ] Cost optimization strategies implemented
    - [ ] Response quality validation
    - [ ] Prompt engineering best practices
    - [ ] Rate limiting and quota management
    - [ ] Fallback strategies for service failures

- [ ] **Monitoring & Observability**
    - [ ] Structured logging throughout
    - [ ] Performance metrics collection
    - [ ] Business metrics tracking
    - [ ] Health check endpoints
    - [ ] Alerting for critical failures

- [ ] **Security & Compliance**
    - [ ] Input validation and sanitization
    - [ ] Output filtering and safety checks
    - [ ] Rate limiting implementation
    - [ ] Audit logging for sensitive operations
    - [ ] Data privacy compliance

---

## V. Quality Gates

- [ ] **Code Quality**
    - [ ] Code coverage > 90% for all new services
    - [ ] Pylint score > 9.0
    - [ ] No critical security vulnerabilities
    - [ ] Performance benchmarks within SLA

- [ ] **AI Quality**
    - [ ] Primitive extraction accuracy > 95%
    - [ ] Criteria generation relevance > 90%
    - [ ] Question-criterion mapping precision > 95%
    - [ ] End-to-end quality validation passes

- [ ] **Integration Quality**
    - [ ] All service interfaces tested
    - [ ] Core API integration tests pass
    - [ ] Backward compatibility maintained
    - [ ] API response times < 2s for single operations

- [ ] **Production Readiness**
    - [ ] Deployment scripts tested
    - [ ] Monitoring dashboards configured
    - [ ] Rollback procedures documented
    - [ ] Capacity planning completed
