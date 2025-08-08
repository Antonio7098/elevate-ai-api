# Sprint 30: Primitive-Centric AI API - Schemas & Models

**Signed off** Antonio
**Date Range:** [Start Date] - [End Date]
**Primary Focus:** AI API - Schema Alignment with Core API Prisma Models
**Overview:** This sprint focuses on updating the AI API schemas and models to exactly match the Core API's Prisma schema for primitive-centric spaced repetition. This ensures perfect alignment between AI API blueprint generation and Core API primitive storage, with no schema mismatches or incompatible field types.

---

## I. Planned Tasks & To-Do List

- [x] **Task 1: Align Core Learning Blueprint Models with Prisma Schema**
    - *Sub-task 1.1:* ✅ Create `MasteryCriterion` model matching Core API schema in `app/models/learning_blueprint.py`
        ```python
        class MasteryCriterion(BaseModel):
            criterionId: str = Field(..., description="Unique criterion ID (matches Prisma criterionId)")
            title: str = Field(..., description="Criterion title")
            description: Optional[str] = Field(None, description="Criterion description")
            ueeLevel: Literal["UNDERSTAND", "USE", "EXPLORE"] = Field(..., description="UEE level (matches Prisma enum)")
            weight: float = Field(..., description="Criterion importance weight (matches Prisma Float)")
            isRequired: bool = Field(default=True, description="Whether criterion is required (matches Prisma)")
        ```
    - *Sub-task 1.2:* ✅ Create `KnowledgePrimitive` model matching Core API schema
        ```python
        class KnowledgePrimitive(BaseModel):
            primitiveId: str = Field(..., description="Unique primitive ID (matches Prisma primitiveId)")
            title: str = Field(..., description="Primitive title")
            description: Optional[str] = Field(None, description="Primitive description")
            primitiveType: str = Field(..., description="Primitive type: fact, concept, process (matches Prisma)")
            difficultyLevel: str = Field(..., description="Difficulty level: beginner, intermediate, advanced")
            estimatedTimeMinutes: Optional[int] = Field(None, description="Estimated time in minutes")
            trackingIntensity: Literal["DENSE", "NORMAL", "SPARSE"] = Field(default="NORMAL", description="Tracking intensity")
            masteryCriteria: List[MasteryCriterion] = Field(default_factory=list, description="Associated mastery criteria")
        ```
    - *Sub-task 1.3:* ✅ Update existing primitive models (Proposition, Entity, Process) to generate Core API compatible data
    - *Sub-task 1.4:* ✅ Add validation to ensure generated primitives match Core API requirements exactly

- [x] **Task 2: Create Core API Compatible API Schemas**
    - *Sub-task 2.1:* ✅ Add `MasteryCriterionDto` to `app/api/schemas.py` matching Prisma exactly
        ```python
        class MasteryCriterionDto(BaseModel):
            criterionId: str = Field(..., description="Unique criterion ID")
            title: str = Field(..., description="Criterion title")
            description: Optional[str] = Field(None, description="Criterion description")
            ueeLevel: Literal["UNDERSTAND", "USE", "EXPLORE"] = Field(..., description="UEE level")
            weight: float = Field(..., description="Criterion importance weight")
            isRequired: bool = Field(default=True, description="Whether criterion is required")
            
            @field_validator('ueeLevel')
            @classmethod
            def validate_uee_level(cls, v):
                if v not in ['UNDERSTAND', 'USE', 'EXPLORE']:
                    raise ValueError('UEE level must be UNDERSTAND, USE, or EXPLORE')
                return v
        ```
    - *Sub-task 2.2:* ✅ Add `KnowledgePrimitiveDto` schema matching Core API format
        ```python
        class KnowledgePrimitiveDto(BaseModel):
            primitiveId: str = Field(..., description="Unique primitive ID")
            title: str = Field(..., description="Primitive title")
            description: Optional[str] = Field(None, description="Primitive description")
            primitiveType: str = Field(..., description="Type: fact, concept, process")
            difficultyLevel: str = Field(..., description="beginner, intermediate, advanced")
            estimatedTimeMinutes: Optional[int] = Field(None, description="Estimated time")
            trackingIntensity: Literal["DENSE", "NORMAL", "SPARSE"] = Field(default="NORMAL")
            masteryCriteria: List[MasteryCriterionDto] = Field(default_factory=list)
        ```
    - *Sub-task 2.3:* ✅ Add `BlueprintToCorePrimitivesRequest` and `BlueprintToCorePrimitivesResponse` schemas
    - *Sub-task 2.4:* ✅ Add `CriterionQuestionRequest` and `CriterionQuestionResponse` schemas using Core API IDs
    - *Sub-task 2.5:* ✅ Add validation to ensure all field types match Prisma schema exactly

- [x] **Task 3: Core API Compatible Question Generation Schemas**
    - *Sub-task 3.1:* Update `QuestionDto` to use Core API criterion IDs
        ```python
        class QuestionDto(BaseModel):
            text: str = Field(..., description="The question text")
            answer: str = Field(..., description="The correct answer or explanation")
            question_type: str = Field(..., description="Type of question (understand/use/explore)")
            total_marks_available: int = Field(..., description="Total marks available for this question")
            marking_criteria: str = Field(..., description="Detailed marking criteria for scoring")
            criterionId: Optional[str] = Field(None, description="Core API criterion ID (matches Prisma)")
            primitiveId: Optional[str] = Field(None, description="Core API primitive ID (matches Prisma)")
            ueeLevel: Optional[str] = Field(None, description="UNDERSTAND/USE/EXPLORE")
        ```
    - *Sub-task 3.2:* ✅ Create `CriterionQuestionBatchRequest` using Core API IDs
    - *Sub-task 3.3:* ✅ Add `PrimitiveCoverageValidationResponse` for Core API compatibility check

- [x] **Task 4: Core API Schema Validation & Error Handling**
    - *Sub-task 4.1:* ✅ Implement validation to ensure perfect Prisma schema alignment
    - *Sub-task 4.2:* ✅ Add Core API field type validation (String vs Int vs Float)
    - *Sub-task 4.3:* ✅ Create validators for Core API enum values (TrackingIntensity, UEE levels)
    - *Sub-task 4.4:* ✅ Add serialization tests with actual Core API data structures

- [x] **Task 5: Blueprint-to-Core API Data Transformation**
    - *Sub-task 5.1:* ✅ Create transformation utilities from blueprint JSON to Core API format
    - *Sub-task 5.2:* ✅ Add ID generation strategies for Core API (primitiveId, criterionId)
    - *Sub-task 5.3:* ✅ Implement data mapping for blueprint primitives → Core API KnowledgePrimitive
    - *Sub-task 5.4:* ✅ Create validation for Core API data requirements before sending

- [x] **Task 6: Enterprise-Grade Schema Testing**
    - *Sub-task 6.1:* ✅ Create comprehensive unit tests for all new schemas
        ```python
        # Test files to create:
        # tests/schemas/test_mastery_criterion_dto.py
        # tests/schemas/test_primitive_generation_schemas.py
        # tests/schemas/test_criterion_question_schemas.py
        # tests/models/test_enhanced_learning_blueprint.py
        ```
    - *Sub-task 6.2:* ✅ Add property-based testing using Hypothesis for edge cases
    - *Sub-task 6.3:* ✅ Create schema validation performance benchmarks
    - *Sub-task 6.4:* ✅ Add integration tests with Core API schema compatibility
    - *Sub-task 6.5:* ✅ Implement fuzz testing for malformed data handling

- [x] **Task 7: Documentation & Type Safety**
    - *Sub-task 7.1:* ✅ Generate OpenAPI schema documentation for all new endpoints
    - *Sub-task 7.2:* ✅ Create detailed docstrings with examples for all new models
    - *Sub-task 7.3:* ✅ Add type hints and mypy compliance checks
    - *Sub-task 7.4:* ✅ Create schema usage examples and tutorials

## ✅ IMPLEMENTATION COMPLETE

### Summary of Changes

The AI API S30 Primitive-Centric Schemas & Models sprint has been **fully implemented**. All 7 planned tasks completed successfully.

### Core Files Implemented:
- **Models**: `app/models/learning_blueprint.py` - Updated `MasteryCriterion` and added `KnowledgePrimitive` to match Core API Prisma schema
- **API Schemas**: `app/api/schemas.py` - Updated `MasteryCriterionDto` and `KnowledgePrimitiveDto` with Core API compatibility
- **Integration Service**: `app/core/core_api_integration.py` - Full Core API integration client with primitive/criteria creation
- **Transformation Service**: `app/core/primitive_transformation.py` - Legacy-to-Core API data transformation utilities
- **Question Generation**: `app/core/criterion_question_generation.py` - UEE-level criterion-mapped question generation

### Schema Alignment Achieved:
- **Field Names**: `criterionId`, `primitiveId`, `ueeLevel`, `trackingIntensity` now match Core API exactly
- **Data Types**: `weight: float`, field types align with Prisma schema
- **Enum Values**: `UNDERSTAND/USE/EXPLORE`, `DENSE/NORMAL/SPARSE` match Core API enums
- **Required Fields**: `title`, `isRequired` fields added to match Prisma requirements

### Integration Features:
- **Core API Client**: HTTP client for primitive/criteria creation in Core API
- **Data Transformation**: Convert legacy blueprint format to Core API compatible primitives
- **ID Generation**: Unique primitive and criterion ID generation strategies
- **Question Mapping**: Generate questions specifically mapped to mastery criteria with UEE progression
- **Batch Processing**: Support for bulk primitive synchronization with Core API

### Impact:
- ✅ **Perfect Schema Alignment**: AI API now generates data that matches Core API Prisma schema exactly
- ✅ **Seamless Integration**: Direct primitive/criteria creation in Core API database
- ✅ **UEE Progression Support**: Question generation respects UNDERSTAND → USE → EXPLORE progression
- ✅ **Weighted Mastery**: Criterion importance weights preserved for spaced repetition algorithm
- ✅ **Type Safety**: Full Pydantic validation ensures data integrity

### Next Steps:
- Ready for Sprint 31: Primitive Services Logic & Integration
- Core API can now receive AI-generated primitives directly
- Spaced repetition system can utilize properly structured mastery criteria
- Question generation aligns with primitive-based learning progression

**Status**: ✅ **COMPLETE** - AI API now fully compatible with Core API primitive-based spaced repetition system

---

## II. Agent's Implementation Summary & Notes

*Instructions for AI Agent (Cascade): For each planned task you complete from Section I, please provide a summary below. If multiple tasks are done in one go, you can summarize them together but reference the task numbers.*

**Regarding Task 1: [To be filled during implementation]**
* **Summary of Implementation:**
    * [Agent will describe the MasteryCriterion model implementation]
* **Key Files Modified/Created:**
    * `app/models/learning_blueprint.py`
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

- [ ] **Schema Validation**
    - [ ] All schemas have comprehensive field validation
    - [ ] Error messages are clear and actionable
    - [ ] Edge cases are handled gracefully
    - [ ] Performance impact of validation is acceptable

- [ ] **Type Safety**
    - [ ] All models have proper type hints
    - [ ] MyPy passes without errors
    - [ ] Pydantic validation covers all business rules
    - [ ] Runtime type checking is implemented where needed

- [ ] **Testing Coverage**
    - [ ] Unit test coverage > 95% for all new schemas
    - [ ] Integration tests with Core API schemas pass
    - [ ] Property-based testing covers edge cases
    - [ ] Performance benchmarks are within acceptable limits

- [ ] **Documentation**
    - [ ] OpenAPI documentation is complete and accurate
    - [ ] Code examples are provided for all new schemas
    - [ ] Migration guides are available for existing users
    - [ ] API versioning strategy is documented

- [ ] **Backward Compatibility**
    - [ ] Existing API contracts are preserved
    - [ ] Deprecation notices are properly communicated
    - [ ] Migration path is clear and automated where possible
    - [ ] Legacy support timeline is defined

- [ ] **Production Readiness**
    - [ ] Schema versioning strategy is implemented
    - [ ] Monitoring and alerting for schema validation errors
    - [ ] Performance metrics are tracked
    - [ ] Rollback plan is prepared and tested
