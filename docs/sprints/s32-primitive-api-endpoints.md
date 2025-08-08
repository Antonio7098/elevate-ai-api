# Sprint 32: Primitive-Centric AI API - Endpoints & Integration

**Signed off** DO NOT PROCEED UNLESS SIGNED OFF BY ANTONIO
**Date Range:** [Start Date] - [End Date]
**Primary Focus:** AI API - API Endpoints & Core API Integration
**Overview:** This sprint focuses on implementing API endpoints that provide Core API Prisma-compatible primitive data and integrate seamlessly with the Core API's KnowledgePrimitive and MasteryCriterion storage. All endpoints ensure perfect field name and type alignment with the Prisma schema to eliminate any integration issues.

---

## I. Planned Tasks & To-Do List

- [ ] **Task 1: Blueprint Primitive Data Access Endpoints**
    - *Sub-task 1.1:* Implement `GET /blueprints/{blueprint_id}/primitives` endpoint in `app/api/endpoints.py`
        ```python
        @router.get("/blueprints/{blueprint_id}/primitives", response_model=BlueprintPrimitivesResponse)
        async def get_blueprint_primitives_endpoint(blueprint_id: str):
            """
            Get formatted primitive data from existing blueprint for Core API storage.
            
            Request Flow:
            1. Validate blueprint_id exists in vector store
            2. Retrieve blueprint JSON from storage
            3. Format primitives for Core API schema
            4. Include mastery criteria with proper structure
            5. Return structured data for Core API import
            """
        ```
    - *Sub-task 1.2:* Add comprehensive blueprint validation and error handling
    - *Sub-task 1.3:* Implement primitive data transformation and formatting
    - *Sub-task 1.4:* Add metadata preservation and relationship mapping
    - *Sub-task 1.5:* Create batch processing for multiple blueprint primitive requests

- [ ] **Task 2: Core API Compatible Question Generation Endpoints**
    - *Sub-task 2.1:* Implement `POST /questions/criterion-specific` endpoint
        ```python
        @router.post("/questions/criterion-specific", response_model=CoreApiQuestionResponse)
        async def generate_core_api_questions_endpoint(request: CoreApiQuestionRequest):
            """
            Generate questions with Core API Prisma-compatible references.
            
            Core Features:
            - Uses actual Core API criterionId and primitiveId
            - Returns ueeLevel in "UNDERSTAND"|"USE"|"EXPLORE" format
            - Question content optimized for Core API Question model
            - Direct compatibility with Core API question storage
            - Validates against Prisma Question schema requirements
            """
        ```
    - *Sub-task 2.2:* Implement batch question generation using Core API IDs
    - *Sub-task 2.3:* Add Prisma Question model validation
    - *Sub-task 2.4:* Create Core API ID verification before question generation
    - *Sub-task 2.5:* Implement Core API question format caching

- [ ] **Task 3: Enhanced Blueprint Creation Endpoints**
    - *Sub-task 3.1:* Update `POST /deconstruct` endpoint to generate primitives with mastery criteria
        ```python
        @router.post("/deconstruct", response_model=EnhancedDeconstructResponse)
        async def enhanced_deconstruct_endpoint(
            request: EnhancedDeconstructRequest
        ):
            """
            Enhanced blueprint creation with primitive mastery criteria.
            
            Enhanced Features:
            - Generate primitives with mastery criteria during creation
            - Include UEE level assignment and weight distribution
            - Optimize for Core API primitive storage format
            - Provide primitive relationship mapping
            - Enable immediate Core API synchronization
            """
        ```
    - *Sub-task 3.2:* Add enhanced request schema with primitive generation options
    - *Sub-task 3.3:* Implement automatic primitive formatting for Core API
    - *Sub-task 3.4:* Create primitive relationship preservation and mapping
    - *Sub-task 3.5:* Add immediate Core API synchronization option

- [ ] **Task 4: Core API Prisma-Compatible Answer Evaluation**
    - *Sub-task 4.1:* Update `POST /api/ai/evaluate-answer` for Prisma criterion evaluation
        ```python
        class PrismaCriterionEvaluationDto(BaseModel):
            criterionId: str = Field(..., description="Core API criterion ID (matches Prisma)")
            primitiveId: str = Field(..., description="Core API primitive ID (matches Prisma)")
            ueeLevel: Literal["UNDERSTAND", "USE", "EXPLORE"] = Field(..., description="Prisma enum value")
            weight: float = Field(..., description="Criterion weight (Prisma Float)")
            difficulty: float = Field(..., description="Criterion difficulty (Prisma Float)")
            isRequired: bool = Field(..., description="Required for progression (Prisma Boolean)")
            
        class PrismaEvaluateAnswerDto(BaseModel):
            questionContext: QuestionContextDto = Field(..., description="Question context")
            userAnswer: str = Field(..., description="User's answer")
            criterionContext: Optional[PrismaCriterionEvaluationDto] = Field(None, description="Prisma criterion data")
            userId: int = Field(..., description="Core API user ID (Prisma Int)")
        ```
    - *Sub-task 4.2:* Implement feedback using Core API Prisma field references
    - *Sub-task 4.3:* Add mastery assessment compatible with Core API UserCriterionMastery model
    - *Sub-task 4.4:* Create progression recommendations using Prisma enum values
    - *Sub-task 4.5:* Implement evaluation results formatted for Core API storage

- [ ] **Task 5: Core API Integration Layer**
    - *Sub-task 5.1:* Create `app/core/core_api_integration.py` service
        ```python
        class CoreAPIIntegration:
            async def send_primitives_to_core(
                self, 
                primitives: List[KnowledgePrimitiveDto], 
                blueprint_id: str,
                user_id: str
            ) -> Dict[str, Any]:
                """Send generated primitives to Core API for storage."""
                
            async def validate_primitive_storage(
                self, 
                primitive_ids: List[str]
            ) -> bool:
                """Verify primitives were successfully stored in Core API."""
                
            async def sync_primitive_updates(
                self, 
                blueprint_id: str
            ) -> Dict[str, Any]:
                """Sync primitive updates between AI API and Core API."""
        ```
    - *Sub-task 5.2:* Implement bidirectional data synchronization
    - *Sub-task 5.3:* Add conflict resolution for concurrent modifications
    - *Sub-task 5.4:* Create integration health monitoring and alerting
    - *Sub-task 5.5:* Implement transaction-like behavior for multi-step operations

- [ ] **Task 6: Vector Store Enhancement for Primitives**
    - *Sub-task 6.1:* Update `app/core/vector_store.py` with primitive-aware metadata
        ```python
        # Enhanced TextNode metadata structure
        primitive_metadata = {
            "primitive_id": str,
            "primitive_type": str,  # proposition, entity, process, etc.
            "criterion_ids": List[str],
            "uee_levels": List[str],
            "criterion_weights": List[int],
            "tracking_intensity": str,
            "mastery_threshold": float,
            "learning_pathway": str
        }
        ```
    - *Sub-task 6.2:* Implement primitive-based search and filtering
    - *Sub-task 6.3:* Add criterion-aware similarity search
    - *Sub-task 6.4:* Create primitive relationship traversal
    - *Sub-task 6.5:* Implement primitive clustering and grouping

- [ ] **Task 7: Enhanced RAG with Primitive Context**
    - *Sub-task 7.1:* Update `app/core/chat.py` for primitive-aware responses
    - *Sub-task 7.2:* Implement criterion-specific context assembly
    - *Sub-task 7.3:* Add mastery-level appropriate response generation
    - *Sub-task 7.4:* Create progressive disclosure based on user mastery
    - *Sub-task 7.5:* Implement personalized learning recommendations

- [ ] **Task 8: API Versioning & Backward Compatibility**
    - *Sub-task 8.1:* Implement API versioning strategy (v1, v2 endpoints)
    - *Sub-task 8.2:* Create compatibility layer for existing integrations
    - *Sub-task 8.3:* Add deprecation warnings and migration guides
    - *Sub-task 8.4:* Implement feature flags for gradual rollout
    - *Sub-task 8.5:* Create automated compatibility testing

- [ ] **Task 9: Performance Optimization & Caching**
    - *Sub-task 9.1:* Implement intelligent caching for primitive generation
    - *Sub-task 9.2:* Add request deduplication and result reuse
    - *Sub-task 9.3:* Create async processing for long-running operations
    - *Sub-task 9.4:* Implement connection pooling and resource management
    - *Sub-task 9.5:* Add response compression and optimization

---

## II. Agent's Implementation Summary & Notes

*Instructions for AI Agent (Cascade): For each planned task you complete from Section I, please provide a summary below. If multiple tasks are done in one go, you can summarize them together but reference the task numbers.*

**Regarding Task 1: [To be filled during implementation]**
* **Summary of Implementation:**
    * [Agent will describe the primitive generation endpoints implementation]
* **Key Files Modified/Created:**
    * `app/api/endpoints.py`
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

- [ ] **API Design & Standards**
    - [ ] RESTful API design principles followed
    - [ ] Consistent error response format
    - [ ] Proper HTTP status codes used
    - [ ] OpenAPI/Swagger documentation complete
    - [ ] API versioning strategy implemented

- [ ] **Performance & Scalability**
    - [ ] Response times < 2s for standard operations
    - [ ] Batch processing for bulk operations
    - [ ] Intelligent caching strategy
    - [ ] Connection pooling implemented
    - [ ] Resource usage monitoring

- [ ] **Security & Authentication**
    - [ ] Input validation on all endpoints
    - [ ] Rate limiting implemented
    - [ ] Authentication/authorization checks
    - [ ] SQL injection prevention
    - [ ] XSS and CSRF protection

- [ ] **Error Handling & Resilience**
    - [ ] Comprehensive error handling
    - [ ] Graceful degradation patterns
    - [ ] Circuit breaker implementation
    - [ ] Retry mechanisms with backoff
    - [ ] Timeout handling

- [ ] **Integration Quality**
    - [ ] Core API integration tested
    - [ ] Data consistency validation
    - [ ] Transaction integrity
    - [ ] Conflict resolution mechanisms
    - [ ] Rollback capabilities

- [ ] **Monitoring & Observability**
    - [ ] Request/response logging
    - [ ] Performance metrics collection
    - [ ] Business metrics tracking
    - [ ] Health check endpoints
    - [ ] Alerting for failures

---

## V. Integration Testing Strategy

- [ ] **Core API Integration Tests**
    - [ ] Primitive generation → Core API storage
    - [ ] Question generation → Core API question creation
    - [ ] Answer evaluation → Core API feedback storage
    - [ ] Blueprint updates → Primitive synchronization

- [ ] **End-to-End Workflow Tests**
    - [ ] Blueprint creation → Primitive extraction → Question generation
    - [ ] User review → Answer evaluation → Mastery tracking
    - [ ] Primitive updates → Question regeneration → User notification

- [ ] **Performance & Load Tests**
    - [ ] Concurrent primitive generation requests
    - [ ] Large blueprint processing
    - [ ] Bulk question generation
    - [ ] Vector store query optimization

- [ ] **Failure & Recovery Tests**
    - [ ] Core API unavailability scenarios
    - [ ] Partial data processing failures
    - [ ] Network timeout handling
    - [ ] Data consistency recovery

---

## VI. Production Deployment Checklist

- [ ] **Environment Configuration**
    - [ ] Production environment variables
    - [ ] Database connection strings
    - [ ] External service configurations
    - [ ] Security certificates and keys

- [ ] **Monitoring Setup**
    - [ ] Application performance monitoring
    - [ ] Error tracking and alerting
    - [ ] Business metrics dashboards
    - [ ] Log aggregation and analysis

- [ ] **Backup & Recovery**
    - [ ] Data backup procedures
    - [ ] Disaster recovery plan
    - [ ] Rollback procedures
    - [ ] Data migration scripts

- [ ] **Documentation**
    - [ ] API documentation published
    - [ ] Integration guides updated
    - [ ] Troubleshooting guides
    - [ ] Operational runbooks
