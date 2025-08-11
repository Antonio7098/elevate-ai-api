# Sprint 33: Performance Testing with pytest-benchmark

import pytest
import asyncio
import time
from typing import List, Dict, Any
from unittest.mock import AsyncMock, patch, Mock
import json

from app.core.deconstruction import DeconstructionService
from app.core.mastery_criteria_service import MasteryCriteriaService
from app.core.question_generation_service import QuestionGenerationService
from app.core.question_mapping_service import QuestionMappingService
from app.core.core_api_sync_service import CoreAPISyncService
from app.core.performance_integration import PerformanceIntegrationService
from app.api.schemas import (
    MasteryCriterionDto, 
    KnowledgePrimitiveDto,
    CriterionQuestionDto
)


class TestPrimitiveGenerationPerformance:
    """Performance tests for primitive generation services."""

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for performance testing."""
        mock_service = AsyncMock()
        
        # Fast mock responses
        mock_service.generate_primitives.return_value = [
            {
                "primitive_id": f"perf_prim_{i:03d}",
                "title": f"Performance Primitive {i}",
                "description": f"Performance test primitive {i}",
                "content": f"Content for performance primitive {i}" * 10,
                "primitive_type": "concept",
                "tags": ["performance", "test"]
            }
            for i in range(5)
        ]
        
        mock_service.generate_mastery_criteria.return_value = [
            {
                "criterion_id": f"perf_crit_{i:03d}",
                "title": f"Performance Criterion {i}",
                "description": f"Performance test criterion {i}",
                "uee_level": ["UNDERSTAND", "USE", "EXPLORE"][i % 3],
                "weight": 3.0,
                "is_required": True
            }
            for i in range(10)
        ]
        
        mock_service.generate_questions.return_value = [
            {
                "question_id": f"perf_q_{i:03d}",
                "question_text": f"Performance question {i}?",
                "question_type": "short_answer",
                "correct_answer": f"Answer {i}",
                "options": [],
                "explanation": f"Explanation {i}",
                "total_marks": 10
            }
            for i in range(3)
        ]
        
        return mock_service

    @pytest.fixture
    def sample_blueprint_small(self):
        """Small blueprint for performance testing."""
        return {
            "title": "Small Performance Blueprint",
            "description": "Small blueprint for performance testing",
            "content": "Short content for performance testing." * 50,
            "sections": [
                {
                    "title": "Section 1",
                    "content": "Section content." * 20
                }
            ]
        }

    @pytest.fixture
    def sample_blueprint_medium(self):
        """Medium blueprint for performance testing."""
        return {
            "title": "Medium Performance Blueprint", 
            "description": "Medium blueprint for performance testing",
            "content": "Medium content for performance testing." * 200,
            "sections": [
                {
                    "title": f"Section {i}",
                    "content": f"Section {i} content." * 50
                }
                for i in range(5)
            ]
        }

    @pytest.fixture
    def sample_blueprint_large(self):
        """Large blueprint for performance testing."""
        return {
            "title": "Large Performance Blueprint",
            "description": "Large blueprint for performance testing", 
            "content": "Large content for performance testing." * 1000,
            "sections": [
                {
                    "title": f"Section {i}",
                    "content": f"Section {i} content." * 100
                }
                for i in range(20)
            ]
        }

    def test_primitive_extraction_small_blueprint(self, benchmark, mock_llm_service, sample_blueprint_small):
        """Benchmark primitive extraction for small blueprints."""
        service = DeconstructionService()
        
        with patch.object(service, 'llm_service', mock_llm_service):
            result = benchmark(
                asyncio.run,
                service.generate_primitives(sample_blueprint_small, {})
            )
        
        assert len(result) > 0
        assert benchmark.stats.stats.mean < 0.1  # Should complete in <100ms

    def test_primitive_extraction_medium_blueprint(self, benchmark, mock_llm_service, sample_blueprint_medium):
        """Benchmark primitive extraction for medium blueprints."""
        service = DeconstructionService()
        
        with patch.object(service, 'llm_service', mock_llm_service):
            result = benchmark(
                asyncio.run,
                service.generate_primitives(sample_blueprint_medium, {})
            )
        
        assert len(result) > 0
        assert benchmark.stats.stats.mean < 0.5  # Should complete in <500ms

    def test_primitive_extraction_large_blueprint(self, benchmark, mock_llm_service, sample_blueprint_large):
        """Benchmark primitive extraction for large blueprints."""
        service = DeconstructionService()
        
        with patch.object(service, 'llm_service', mock_llm_service):
            result = benchmark(
                asyncio.run,
                service.generate_primitives(sample_blueprint_large, {})
            )
        
        assert len(result) > 0
        assert benchmark.stats.stats.mean < 2.0  # Should complete in <2s

    def test_mastery_criteria_generation_performance(self, benchmark, mock_llm_service):
        """Benchmark mastery criteria generation."""
        service = MasteryCriteriaService()
        
        # Sample primitive
        primitive = KnowledgePrimitiveDto(
            primitive_id="perf_test_prim",
            title="Performance Test Primitive",
            description="A primitive for performance testing",
            content="Performance test content " * 100,
            primitive_type="concept",
            tags=["performance"],
            mastery_criteria=[]
        )
        
        with patch.object(service, 'llm_service', mock_llm_service):
            result = benchmark(
                asyncio.run,
                service.generate_mastery_criteria(primitive, {})
            )
        
        assert len(result) > 0
        assert benchmark.stats.stats.mean < 0.3  # Should complete in <300ms

    def test_question_generation_performance(self, benchmark, mock_llm_service):
        """Benchmark question generation."""
        service = QuestionGenerationService()
        
        # Sample criterion
        criterion = MasteryCriterionDto(
            criterion_id="perf_test_crit",
            primitive_id="perf_test_prim",
            title="Performance Test Criterion",
            description="A criterion for performance testing",
            uee_level="UNDERSTAND",
            weight=3.0,
            is_required=True
        )
        
        primitive = KnowledgePrimitiveDto(
            primitive_id="perf_test_prim",
            title="Performance Test Primitive",
            description="A primitive for performance testing",
            content="Performance test content " * 50,
            primitive_type="concept",
            tags=["performance"],
            mastery_criteria=[criterion]
        )
        
        with patch.object(service, 'llm_service', mock_llm_service):
            result = benchmark(
                asyncio.run,
                service.generate_criterion_questions(criterion, primitive, {})
            )
        
        assert len(result) > 0
        assert benchmark.stats.stats.mean < 0.4  # Should complete in <400ms

    def test_batch_primitive_generation_performance(self, benchmark, mock_llm_service):
        """Benchmark batch primitive generation."""
        service = DeconstructionService()
        
        # Multiple small blueprints
        blueprints = [
            {
                "title": f"Batch Blueprint {i}",
                "description": f"Batch blueprint {i}",
                "content": f"Batch content {i} " * 30,
                "sections": [{"title": f"Section {i}", "content": f"Section content {i} " * 20}]
            }
            for i in range(5)
        ]
        
        async def batch_generate():
            tasks = []
            for blueprint in blueprints:
                task = service.generate_primitives(blueprint, {})
                tasks.append(task)
            return await asyncio.gather(*tasks)
        
        with patch.object(service, 'llm_service', mock_llm_service):
            results = benchmark(asyncio.run, batch_generate())
        
        assert len(results) == 5
        assert all(len(result) > 0 for result in results)
        assert benchmark.stats.stats.mean < 1.0  # Should complete in <1s

    def test_question_mapping_performance(self, benchmark):
        """Benchmark question-criterion mapping."""
        service = QuestionMappingService()
        
        # Sample questions and criteria
        questions = [
            CriterionQuestionDto(
                question_id=f"q_{i:03d}",
                criterion_id=f"crit_{i%3:03d}",
                primitive_id="prim_001",
                question_text=f"What is concept {i}?",
                question_type="short_answer",
                correct_answer=f"Answer {i}",
                options=[],
                explanation=f"Explanation {i}",
                total_marks=10,
                similarity_score=0.8
            )
            for i in range(20)
        ]
        
        criteria = [
            MasteryCriterionDto(
                criterion_id=f"crit_{i:03d}",
                primitive_id="prim_001",
                title=f"Criterion {i}",
                description=f"Test criterion {i}",
                uee_level="UNDERSTAND",
                weight=3.0,
                is_required=True
            )
            for i in range(3)
        ]
        
        result = benchmark(
            asyncio.run,
            service.map_questions_to_criteria(questions, criteria)
        )
        
        assert len(result) > 0
        assert benchmark.stats.stats.mean < 0.2  # Should complete in <200ms


class TestAPIEndpointPerformance:
    """Performance tests for API endpoints."""

    @pytest.fixture
    def mock_services(self):
        """Mock all services for API performance testing."""
        mocks = {}
        
        # Mock primitive generation
        mocks['primitive_service'] = AsyncMock()
        mocks['primitive_service'].generate_primitives.return_value = [
            KnowledgePrimitiveDto(
                primitive_id=f"api_prim_{i:03d}",
                title=f"API Primitive {i}",
                description=f"API primitive {i}",
                content=f"API content {i}",
                primitive_type="concept",
                tags=["api"],
                mastery_criteria=[]
            )
            for i in range(3)
        ]
        
        # Mock question generation
        mocks['question_service'] = AsyncMock()
        mocks['question_service'].generate_criterion_questions.return_value = [
            CriterionQuestionDto(
                question_id=f"api_q_{i:03d}",
                criterion_id="api_crit_001",
                primitive_id="api_prim_001",
                question_text=f"API question {i}?",
                question_type="short_answer",
                correct_answer=f"API answer {i}",
                options=[],
                explanation=f"API explanation {i}",
                total_marks=10,
                similarity_score=0.9
            )
            for i in range(2)
        ]
        
        # Mock Core API sync
        mocks['sync_service'] = AsyncMock()
        mocks['sync_service'].sync_primitives.return_value = {
            "success": True,
            "synced_count": 3,
            "failed_count": 0
        }
        
        return mocks

    @pytest.mark.asyncio
    async def test_blueprint_primitives_endpoint_performance(self, benchmark, mock_services):
        """Benchmark blueprint primitives endpoint."""
        from app.api.endpoints import router
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        # Mock the services in the endpoint
        with patch('app.api.endpoints.DeconstructionService') as mock_decon, \
             patch('app.api.endpoints.MasteryCriteriaService') as mock_criteria, \
             patch('app.api.endpoints.CoreAPISyncService') as mock_sync:
            
            mock_decon.return_value = mock_services['primitive_service']
            mock_criteria.return_value = AsyncMock()
            mock_sync.return_value = mock_services['sync_service']
            
            def make_request():
                return client.post(
                    "/api/v1/blueprints/test_blueprint_001/primitives",
                    json={
                        "user_preferences": {
                            "primitive_count": 3,
                            "detail_level": "medium"
                        }
                    }
                )
            
            response = benchmark(make_request)
            
            assert response.status_code == 200
            assert benchmark.stats.stats.mean < 0.5  # Should respond in <500ms

    @pytest.mark.asyncio 
    async def test_question_generation_endpoint_performance(self, benchmark, mock_services):
        """Benchmark question generation endpoint."""
        from app.api.endpoints import router
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        with patch('app.api.endpoints.QuestionGenerationService') as mock_question:
            mock_question.return_value = mock_services['question_service']
            
            def make_request():
                return client.post(
                    "/api/v1/primitives/test_prim_001/questions",
                    json={
                        "criterion_id": "test_crit_001",
                        "question_count": 2,
                        "user_preferences": {
                            "difficulty_level": "medium"
                        }
                    }
                )
            
            response = benchmark(make_request)
            
            assert response.status_code == 200
            assert benchmark.stats.stats.mean < 0.3  # Should respond in <300ms


class TestConcurrentPerformance:
    """Performance tests for concurrent operations."""

    @pytest.fixture
    def mock_async_services(self):
        """Mock async services for concurrent testing."""
        async def mock_primitive_gen(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing time
            return [
                KnowledgePrimitiveDto(
                    primitive_id="concurrent_prim_001",
                    title="Concurrent Primitive",
                    description="Concurrent test primitive",
                    content="Concurrent content",
                    primitive_type="concept",
                    tags=["concurrent"],
                    mastery_criteria=[]
                )
            ]
        
        async def mock_question_gen(*args, **kwargs):
            await asyncio.sleep(0.05)  # Simulate processing time
            return [
                CriterionQuestionDto(
                    question_id="concurrent_q_001",
                    criterion_id="concurrent_crit_001",
                    primitive_id="concurrent_prim_001", 
                    question_text="Concurrent question?",
                    question_type="short_answer",
                    correct_answer="Concurrent answer",
                    options=[],
                    explanation="Concurrent explanation",
                    total_marks=10,
                    similarity_score=0.8
                )
            ]
        
        return {
            'primitive_gen': mock_primitive_gen,
            'question_gen': mock_question_gen
        }

    def test_concurrent_primitive_generation(self, benchmark, mock_async_services):
        """Test concurrent primitive generation performance."""
        async def concurrent_generation():
            tasks = []
            for i in range(10):
                task = mock_async_services['primitive_gen']()
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        results = benchmark(asyncio.run, concurrent_generation())
        
        assert len(results) == 10
        assert all(len(result) > 0 for result in results)
        # Should be faster than sequential due to concurrency
        assert benchmark.stats.stats.mean < 0.5

    def test_concurrent_question_generation(self, benchmark, mock_async_services):
        """Test concurrent question generation performance."""
        async def concurrent_question_gen():
            tasks = []
            for i in range(20):
                task = mock_async_services['question_gen']()
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        results = benchmark(asyncio.run, concurrent_question_gen())
        
        assert len(results) == 20
        assert all(len(result) > 0 for result in results)
        # Should benefit from concurrency
        assert benchmark.stats.stats.mean < 0.3


class TestMemoryUsagePerformance:
    """Performance tests focusing on memory usage."""

    def test_large_blueprint_memory_usage(self, benchmark):
        """Test memory usage for large blueprint processing."""
        import psutil
        import os
        
        # Create very large blueprint
        large_blueprint = {
            "title": "Memory Test Blueprint",
            "description": "Very large blueprint for memory testing",
            "content": "Large content " * 10000,  # ~120KB of text
            "sections": [
                {
                    "title": f"Large Section {i}",
                    "content": f"Large section content {i} " * 1000
                }
                for i in range(50)  # 50 large sections
            ]
        }
        
        def memory_intensive_operation():
            # Simulate primitive extraction processing
            sections = large_blueprint["sections"]
            processed_sections = []
            
            for section in sections:
                # Simulate text processing
                processed_content = section["content"].split()
                processed_sections.append({
                    "title": section["title"],
                    "word_count": len(processed_content),
                    "content_preview": " ".join(processed_content[:100])
                })
            
            return processed_sections
        
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        result = benchmark(memory_intensive_operation)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert len(result) == 50
        assert memory_increase < 100  # Should not increase by more than 100MB
        assert benchmark.stats.stats.mean < 1.0  # Should complete in <1s

    def test_batch_processing_memory_efficiency(self, benchmark):
        """Test memory efficiency of batch processing."""
        import gc
        
        def batch_processing_operation():
            # Simulate processing many small items
            items = []
            for i in range(1000):
                item = {
                    "id": f"item_{i:04d}",
                    "data": f"Item data {i} " * 100,
                    "metadata": {
                        "processed": True,
                        "timestamp": time.time()
                    }
                }
                items.append(item)
            
            # Process in batches to manage memory
            results = []
            batch_size = 100
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_result = [
                    {
                        "id": item["id"],
                        "summary": f"Processed {item['id']}"
                    }
                    for item in batch
                ]
                results.extend(batch_result)
                
                # Simulate memory cleanup
                del batch
                if i % 500 == 0:
                    gc.collect()
            
            return results
        
        result = benchmark(batch_processing_operation)
        
        assert len(result) == 1000
        assert benchmark.stats.stats.mean < 0.5  # Should complete in <500ms


# Performance test configuration
pytestmark = [pytest.mark.performance, pytest.mark.benchmark]
