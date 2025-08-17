# Sprint 57: AI API Multi-Primitive Mastery Criteria Generation

**Signed off** DO NOT PROCEED UNLESS SIGNED OFF BY ANTONIO
**Date Range:** [Start Date] - [End Date]
**Primary Focus:** AI API - Enhance AI generation system for multi-primitive mastery criteria
**Overview:** Transform the AI generation system to create sophisticated mastery criteria that link to multiple knowledge primitives, enabling complex, interconnected learning objectives that scale with UUE stage complexity.

---

## I. Sprint Goals & Objectives

### Primary Goals:
1. Enhance AI generation prompts for multi-primitive mastery criteria
2. Implement relationship validation and quality assurance
3. Update generation workflows for complex criteria
4. Add AI-powered relationship suggestion engine
5. Ensure backward compatibility with existing single-primitive generation

### Success Criteria:
- AI generates mastery criteria with appropriate primitive counts per UUE stage
- Relationship validation prevents circular dependencies and invalid connections
- Generation quality maintains high standards for complex criteria
- AI suggests meaningful relationships between primitives
- Existing single-primitive generation continues to work

---

## II. Planned Tasks & To-Do List

### **Task 1: Enhanced AI Generation Prompts**
- [ ] **Sub-task 1.1:** Update mastery criteria generation prompts
  - Modify prompts to generate multi-primitive criteria
  - Add UUE stage complexity requirements
  - Implement relationship type generation
  - Add primitive count validation

- [ ] **Sub-task 1.2:** Enhance primitive relationship generation
  - Create prompts for relationship type assignment
  - Add relationship strength calculation
  - Implement weight distribution logic
  - Add relationship validation rules

- [ ] **Sub-task 1.3:** Implement complexity-aware generation
  - Add UUE stage-specific generation rules
  - Implement primitive count scaling
  - Add complexity scoring generation
  - Create stage progression validation

- [ ] **Sub-task 1.4:** Add relationship quality assurance
  - Implement semantic similarity validation
  - Add content overlap detection
  - Create relationship strength validation
  - Add relationship conflict detection

### **Task 2: Generation Workflow Updates**
- [ ] **Sub-task 2.1:** Update sequential generation workflow
  - Modify mastery criteria generation step
  - Add relationship validation step
  - Implement complexity scoring step
  - Add quality assurance step

- [ ] **Sub-task 2.2:** Enhance parallel generation system
  - Update section processing for multi-primitive criteria
  - Add relationship cross-validation
  - Implement dependency-aware generation
  - Add relationship consistency checks

- [ ] **Sub-task 2.3:** Implement relationship generation orchestration
  - Create relationship generation coordinator
  - Add relationship validation pipeline
  - Implement relationship optimization
  - Add relationship quality scoring

- [ ] **Sub-task 2.4:** Add generation quality monitoring
  - Implement quality metrics tracking
  - Add relationship validation reporting
  - Create generation performance monitoring
  - Add quality improvement suggestions

### **Task 3: AI-Powered Relationship Engine**
- [ ] **Sub-task 3.1:** Create relationship suggestion engine
  - Implement semantic similarity analysis
  - Add content overlap detection
  - Create learning pattern analysis
  - Add user behavior analysis

- [ ] **Sub-task 3.2:** Implement relationship validation AI
  - Create circular dependency detection
  - Add prerequisite chain validation
  - Implement relationship strength validation
  - Add relationship conflict resolution

- [ ] **Sub-task 3.3:** Add relationship optimization AI
  - Implement relationship strength optimization
  - Add relationship type optimization
  - Create weight distribution optimization
  - Add complexity balance optimization

- [ ] **Sub-task 3.4:** Create relationship quality scoring
  - Implement semantic coherence scoring
  - Add learning effectiveness scoring
  - Create relationship strength scoring
  - Add overall quality scoring

### **Task 4: Quality Assurance and Validation**
- [ ] **Sub-task 4.1:** Implement comprehensive validation
  - Add relationship validation rules
  - Implement complexity validation
  - Add UUE stage validation
  - Create quality threshold validation

- [ ] **Sub-task 4.2:** Add relationship conflict detection
  - Implement circular dependency detection
  - Add prerequisite chain validation
  - Create relationship strength validation
  - Add relationship type validation

- [ ] **Sub-task 4.3:** Create quality improvement system
  - Implement quality feedback loops
  - Add generation improvement suggestions
  - Create relationship optimization recommendations
  - Add quality trend analysis

- [ ] **Sub-task 4.4:** Implement backward compatibility
  - Ensure single-primitive generation works
  - Add compatibility mode for existing workflows
  - Implement gradual migration support
  - Add legacy format support

---

## III. Technical Details

### Enhanced Generation Prompts

#### **Multi-Primitive Mastery Criteria Generation**
```python
MULTI_PRIMITIVE_CRITERIA_PROMPT = """
You are an expert educational content creator. Your task is to generate mastery criteria that test understanding of multiple interconnected concepts.

CONTEXT:
- Source content: {source_text}
- Blueprint section: {section_title}
- Available primitives: {available_primitives}
- Target UUE stage: {target_uue_stage}

UUE STAGE REQUIREMENTS:
- UNDERSTAND: 1-2 primitives (basic comprehension)
- USE: 2-4 primitives (application and synthesis)
- EXPLORE: 4+ primitives (advanced integration and creation)

TASK:
Generate {criterion_count} mastery criteria that:
1. Link to the appropriate number of primitives for the UUE stage
2. Test understanding of how concepts interconnect
3. Have clear, measurable learning objectives
4. Scale complexity appropriately for the stage

For each criterion, specify:
- Title: Clear, concise learning objective
- Description: Detailed explanation of what mastery entails
- Linked primitives: List of primitive IDs with relationship types
- Relationship types: PRIMARY (core), SECONDARY (supporting), CONTEXTUAL (background)
- Weights: Importance of each primitive (1.0-5.0 scale)
- UUE stage: UNDERSTAND, USE, or EXPLORE
- Complexity score: 1-10 scale based on primitive count and relationships

OUTPUT FORMAT:
```json
{
  "criteria": [
    {
      "title": "Criterion title",
      "description": "Detailed description",
      "uue_stage": "UNDERSTAND|USE|EXPLORE",
      "complexity_score": 5,
      "linked_primitives": [
        {
          "primitive_id": "primitive_001",
          "relationship_type": "PRIMARY",
          "weight": 3.0,
          "strength": 0.9
        }
      ]
    }
  ]
}
```

QUALITY REQUIREMENTS:
- Each criterion must test meaningful concept integration
- Primitive counts must match UUE stage requirements
- Relationships must be logical and educationally sound
- No circular dependencies between criteria
- All primitives must be relevant to the learning objective
"""
```

#### **Relationship Validation Prompt**
```python
RELATIONSHIP_VALIDATION_PROMPT = """
You are an expert educational content validator. Your task is to validate the relationships between mastery criteria and knowledge primitives.

CONTEXT:
- Mastery criteria: {mastery_criteria}
- Knowledge primitives: {knowledge_primitives}
- Blueprint structure: {blueprint_structure}

VALIDATION TASKS:
1. Check for circular dependencies in prerequisite chains
2. Validate relationship strength scores (0.0-1.0)
3. Ensure relationship types are appropriate
4. Verify primitive counts match UUE stage requirements
5. Check for semantic coherence between related concepts

VALIDATION RULES:
- No criterion should be a prerequisite for itself (direct or indirect)
- Relationship strengths should reflect semantic similarity
- PRIMARY relationships should be core to the learning objective
- SECONDARY relationships should support the main concept
- CONTEXTUAL relationships should provide background context
- UUE stage primitive counts: UNDERSTAND (1-2), USE (2-4), EXPLORE (4+)

OUTPUT FORMAT:
```json
{
  "validation_results": [
    {
      "criterion_id": "criterion_001",
      "is_valid": true,
      "issues": [],
      "suggestions": [],
      "relationship_quality_score": 0.85
    }
  ],
  "overall_validation": {
    "is_valid": true,
    "total_issues": 0,
    "average_quality_score": 0.87,
    "recommendations": []
  }
}
```

ISSUE TYPES:
- CIRCULAR_DEPENDENCY: Criterion creates circular prerequisite chain
- INVALID_RELATIONSHIP_STRENGTH: Strength score outside valid range
- INAPPROPRIATE_RELATIONSHIP_TYPE: Relationship type doesn't match content
- PRIMITIVE_COUNT_MISMATCH: Primitive count doesn't match UUE stage
- SEMANTIC_INCOHERENCE: Related concepts lack logical connection
"""
```

### Enhanced Generation Workflows

#### **Multi-Primitive Generation Orchestrator**
```python
class MultiPrimitiveGenerationOrchestrator:
    def __init__(self, llm_service, validation_service):
        self.llm_service = llm_service
        self.validation_service = validation_service
        self.relationship_engine = RelationshipSuggestionEngine()
    
    async def generate_multi_primitive_criteria(
        self,
        section: BlueprintSection,
        primitives: List[KnowledgePrimitive],
        target_uue_stage: UueStage
    ) -> List[GeneratedMasteryCriterion]:
        """Generate mastery criteria with multiple primitive relationships."""
        
        # Step 1: Generate initial criteria
        initial_criteria = await self._generate_initial_criteria(
            section, primitives, target_uue_stage
        )
        
        # Step 2: Generate primitive relationships
        criteria_with_relationships = await self._add_primitive_relationships(
            initial_criteria, primitives
        )
        
        # Step 3: Validate relationships
        validation_results = await self._validate_relationships(
            criteria_with_relationships
        )
        
        # Step 4: Optimize relationships
        optimized_criteria = await self._optimize_relationships(
            criteria_with_relationships, validation_results
        )
        
        # Step 5: Final quality check
        final_criteria = await self._final_quality_check(optimized_criteria)
        
        return final_criteria
    
    async def _generate_initial_criteria(
        self,
        section: BlueprintSection,
        primitives: List[KnowledgePrimitive],
        target_uue_stage: UueStage
    ) -> List[Dict]:
        """Generate initial mastery criteria without relationships."""
        
        prompt = MULTI_PRIMITIVE_CRITERIA_PROMPT.format(
            source_text=section.source_text,
            section_title=section.title,
            available_primitives=[p.title for p in primitives],
            target_uue_stage=target_uue_stage.value,
            criterion_count=self._get_criterion_count_for_stage(target_uue_stage)
        )
        
        response = await self.llm_service.generate(
            prompt=prompt,
            model="gemini-2.5-flash",
            temperature=0.7,
            max_tokens=2000
        )
        
        return self._parse_criteria_response(response)
    
    async def _add_primitive_relationships(
        self,
        criteria: List[Dict],
        primitives: List[KnowledgePrimitive]
    ) -> List[GeneratedMasteryCriterion]:
        """Add primitive relationships to criteria."""
        
        enhanced_criteria = []
        
        for criterion in criteria:
            # Get AI suggestions for relationships
            relationship_suggestions = await self.relationship_engine.suggest_relationships(
                criterion, primitives
            )
            
            # Create enhanced criterion with relationships
            enhanced_criterion = GeneratedMasteryCriterion(
                title=criterion["title"],
                description=criterion["description"],
                uue_stage=UueStage(criterion["uue_stage"]),
                complexity_score=criterion["complexity_score"],
                linked_primitives=relationship_suggestions
            )
            
            enhanced_criteria.append(enhanced_criterion)
        
        return enhanced_criteria
    
    async def _validate_relationships(
        self,
        criteria: List[GeneratedMasteryCriterion]
    ) -> ValidationResults:
        """Validate all relationships in criteria."""
        
        validation_prompt = RELATIONSHIP_VALIDATION_PROMPT.format(
            mastery_criteria=[c.dict() for c in criteria],
            knowledge_primitives=[p.dict() for p in self.primitives],
            blueprint_structure=self.blueprint_structure
        )
        
        response = await self.llm_service.generate(
            prompt=validation_prompt,
            model="gemini-2.5-flash",
            temperature=0.3,
            max_tokens=1500
        )
        
        return self._parse_validation_response(response)
    
    async def _optimize_relationships(
        self,
        criteria: List[GeneratedMasteryCriterion],
        validation_results: ValidationResults
    ) -> List[GeneratedMasteryCriterion]:
        """Optimize relationships based on validation results."""
        
        optimized_criteria = []
        
        for criterion, validation in zip(criteria, validation_results.results):
            if not validation.is_valid:
                # Fix validation issues
                optimized_criterion = await self._fix_validation_issues(
                    criterion, validation
                )
            else:
                # Optimize for quality
                optimized_criterion = await self._optimize_for_quality(criterion)
            
            optimized_criteria.append(optimized_criterion)
        
        return optimized_criteria
    
    def _get_criterion_count_for_stage(self, uue_stage: UueStage) -> int:
        """Get appropriate criterion count for UUE stage."""
        stage_counts = {
            UueStage.UNDERSTAND: 3,
            UueStage.USE: 4,
            UueStage.EXPLORE: 5
        }
        return stage_counts.get(uue_stage, 3)
```

### Relationship Suggestion Engine

#### **AI-Powered Relationship Suggestions**
```python
class RelationshipSuggestionEngine:
    def __init__(self, llm_service, embedding_service):
        self.llm_service = llm_service
        self.embedding_service = embedding_service
    
    async def suggest_relationships(
        self,
        criterion: Dict,
        available_primitives: List[KnowledgePrimitive]
    ) -> List[PrimitiveRelationship]:
        """Suggest primitive relationships for a mastery criterion."""
        
        # Calculate semantic similarities
        semantic_scores = await self._calculate_semantic_similarities(
            criterion, available_primitives
        )
        
        # Get AI-powered relationship suggestions
        ai_suggestions = await self._get_ai_relationship_suggestions(
            criterion, available_primitives, semantic_scores
        )
        
        # Combine semantic and AI suggestions
        combined_suggestions = self._combine_suggestions(
            semantic_scores, ai_suggestions
        )
        
        # Filter and rank suggestions
        ranked_suggestions = self._rank_suggestions(
            combined_suggestions, criterion
        )
        
        # Convert to relationship objects
        relationships = self._create_relationships(ranked_suggestions)
        
        return relationships
    
    async def _calculate_semantic_similarities(
        self,
        criterion: Dict,
        primitives: List[KnowledgePrimitive]
    ) -> Dict[str, float]:
        """Calculate semantic similarity between criterion and primitives."""
        
        criterion_embedding = await self.embedding_service.get_embedding(
            criterion["title"] + " " + criterion["description"]
        )
        
        similarities = {}
        
        for primitive in primitives:
            primitive_embedding = await self.embedding_service.get_embedding(
                primitive.title + " " + (primitive.description or "")
            )
            
            similarity = self._cosine_similarity(
                criterion_embedding, primitive_embedding
            )
            
            similarities[primitive.primitive_id] = similarity
        
        return similarities
    
    async def _get_ai_relationship_suggestions(
        self,
        criterion: Dict,
        primitives: List[KnowledgePrimitive],
        semantic_scores: Dict[str, float]
    ) -> List[Dict]:
        """Get AI-powered relationship suggestions."""
        
        prompt = RELATIONSHIP_SUGGESTION_PROMPT.format(
            criterion_title=criterion["title"],
            criterion_description=criterion["description"],
            uue_stage=criterion["uue_stage"],
            available_primitives=[p.title for p in primitives],
            semantic_scores=semantic_scores
        )
        
        response = await self.llm_service.generate(
            prompt=prompt,
            model="gemini-2.5-flash",
            temperature=0.6,
            max_tokens=1000
        )
        
        return self._parse_relationship_suggestions(response)
    
    def _combine_suggestions(
        self,
        semantic_scores: Dict[str, float],
        ai_suggestions: List[Dict]
    ) -> List[Dict]:
        """Combine semantic and AI suggestions."""
        
        combined = []
        
        for primitive_id, semantic_score in semantic_scores.items():
            ai_suggestion = next(
                (s for s in ai_suggestions if s["primitive_id"] == primitive_id),
                None
            )
            
            combined_score = self._calculate_combined_score(
                semantic_score, ai_suggestion
            )
            
            combined.append({
                "primitive_id": primitive_id,
                "semantic_score": semantic_score,
                "ai_suggestion": ai_suggestion,
                "combined_score": combined_score
            })
        
        return combined
    
    def _rank_suggestions(
        self,
        suggestions: List[Dict],
        criterion: Dict
    ) -> List[Dict]:
        """Rank suggestions based on multiple factors."""
        
        for suggestion in suggestions:
            # Calculate relationship type score
            suggestion["type_score"] = self._calculate_type_score(
                suggestion, criterion
            )
            
            # Calculate weight score
            suggestion["weight_score"] = self._calculate_weight_score(
                suggestion, criterion
            )
            
            # Calculate final ranking score
            suggestion["ranking_score"] = (
                suggestion["combined_score"] * 0.4 +
                suggestion["type_score"] * 0.3 +
                suggestion["weight_score"] * 0.3
            )
        
        # Sort by ranking score
        suggestions.sort(key=lambda x: x["ranking_score"], reverse=True)
        
        return suggestions
    
    def _create_relationships(
        self,
        ranked_suggestions: List[Dict]
    ) -> List[PrimitiveRelationship]:
        """Create relationship objects from ranked suggestions."""
        
        relationships = []
        
        for suggestion in ranked_suggestions:
            relationship = PrimitiveRelationship(
                primitive_id=suggestion["primitive_id"],
                relationship_type=self._determine_relationship_type(suggestion),
                weight=self._calculate_weight(suggestion),
                strength=suggestion["combined_score"]
            )
            
            relationships.append(relationship)
        
        return relationships
```

---

## IV. Quality Assurance System

### **Comprehensive Validation Pipeline**
```python
class MultiPrimitiveValidationPipeline:
    def __init__(self):
        self.validators = [
            CircularDependencyValidator(),
            RelationshipStrengthValidator(),
            UueStageValidator(),
            SemanticCoherenceValidator(),
            ComplexityValidator()
        ]
    
    async def validate_criteria(
        self,
        criteria: List[GeneratedMasteryCriterion]
    ) -> ValidationResults:
        """Run comprehensive validation on all criteria."""
        
        validation_results = []
        
        for criterion in criteria:
            criterion_validation = await self._validate_criterion(criterion)
            validation_results.append(criterion_validation)
        
        overall_validation = self._aggregate_validation_results(validation_results)
        
        return ValidationResults(
            results=validation_results,
            overall=overall_validation
        )
    
    async def _validate_criterion(
        self,
        criterion: GeneratedMasteryCriterion
    ) -> CriterionValidationResult:
        """Validate a single criterion."""
        
        validation_issues = []
        
        for validator in self.validators:
            try:
                validator_result = await validator.validate(criterion)
                if not validator_result.is_valid:
                    validation_issues.extend(validator_result.issues)
            except Exception as e:
                validation_issues.append(f"Validation error: {str(e)}")
        
        return CriterionValidationResult(
            criterion_id=criterion.id,
            is_valid=len(validation_issues) == 0,
            issues=validation_issues,
            quality_score=self._calculate_quality_score(criterion, validation_issues)
        )
```

---

## V. Testing Strategy

### **Generation Quality Testing**
- Test multi-primitive criteria generation
- Validate relationship quality
- Test UUE stage complexity requirements
- Verify backward compatibility

### **Performance Testing**
- Test generation speed with multiple primitives
- Validate memory usage for complex criteria
- Test relationship validation performance
- Monitor API response times

### **Quality Assurance Testing**
- Test relationship validation accuracy
- Validate circular dependency detection
- Test semantic coherence validation
- Verify quality scoring accuracy

---

## VI. Risk Assessment & Mitigation

### **High Risk Items**
1. **Generation Quality**: Complex criteria may be lower quality
   - *Mitigation*: Enhanced prompts, validation, quality scoring

2. **Performance Impact**: Multi-primitive generation may be slower
   - *Mitigation*: Optimization, caching, parallel processing

3. **Validation Complexity**: Complex validation may have edge cases
   - *Mitigation*: Comprehensive testing, validation rules, error handling

### **Medium Risk Items**
1. **Relationship Quality**: AI may suggest poor relationships
   - *Mitigation*: Semantic validation, quality scoring, human review

2. **Backward Compatibility**: Changes may break existing workflows
   - *Mitigation*: Compatibility mode, gradual migration, testing

---

## VII. Success Metrics

### **Generation Quality Metrics**
- [ ] Multi-primitive criteria quality score > 0.8
- [ ] Relationship validation accuracy > 95%
- [ ] UUE stage complexity compliance > 98%
- [ ] Semantic coherence score > 0.85

### **Performance Metrics**
- [ ] Generation time within 20% of baseline
- [ ] Memory usage remains stable
- [ ] API response times acceptable
- [ ] Validation performance meets targets

### **Quality Metrics**
- [ ] No critical validation failures
- [ ] Relationship quality scores > 0.8
- [ ] User acceptance testing passed
- [ ] Quality improvement trends positive

---

## VIII. Dependencies & Blockers

### **Dependencies**
- Backend multi-primitive API endpoints
- Updated data models and validation rules
- Enhanced LLM service capabilities
- Relationship validation services

### **Blockers**
- None identified at this time

---

## IX. Next Steps

### **Immediate Next Steps (Next Sprint)**
1. Integration with frontend multi-primitive interface
2. Advanced relationship optimization algorithms
3. Quality monitoring and improvement systems
4. User feedback integration

### **Future Considerations**
1. Machine learning for relationship quality prediction
2. Advanced semantic analysis for better relationships
3. Real-time relationship optimization
4. Integration with external AI services

---

**Sprint Status:** [To be filled out by Antonio after work is done]
**Completion Date:** [To be filled out by Antonio after work is done]
**Notes:** [To be filled out by Antonio after work is done]
