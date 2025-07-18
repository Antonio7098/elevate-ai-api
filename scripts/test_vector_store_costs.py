#!/usr/bin/env python3
"""
Test script to measure vector store costs for blueprint ingestion operations.

This script tests the cost implications of:
1. Google Gemini embedding API calls
2. ChromaDB storage operations
3. Vector search operations
4. Batch processing efficiency

Usage:
    python scripts/test_vector_store_costs.py
"""

import asyncio
import json
import time
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

from app.core.blueprint_parser import BlueprintParser
from app.core.embeddings import GoogleEmbeddingService
from app.core.vector_store import ChromaDBVectorStore
from app.core.indexing_pipeline import IndexingPipeline
from app.core.search_service import SearchService
from app.models.learning_blueprint import LearningBlueprint


@dataclass
class CostMetrics:
    """Metrics for cost analysis."""
    operation: str
    text_nodes_count: int
    total_characters: int
    total_words: int
    embedding_calls: int
    processing_time: float
    estimated_cost: float
    cost_per_word: float
    cost_per_node: float


class VectorStoreCostTester:
    """Test vector store operations and estimate costs."""
    
    # Google Gemini API pricing (as of 2024)
    GEMINI_EMBEDDING_COST_PER_1K_CHARS = 0.00001  # $0.00001 per 1K characters
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY', 'test-key')
        self.embedding_service = GoogleEmbeddingService(self.api_key)
        self.vector_store = ChromaDBVectorStore()
        self.blueprint_parser = BlueprintParser()
        self.search_service = SearchService(self.vector_store, self.embedding_service)
        self.indexing_pipeline = IndexingPipeline()
        
        self.results: List[CostMetrics] = []
    
    def generate_test_blueprint(self, complexity_level: str = "medium") -> LearningBlueprint:
        """Generate test blueprint with different complexity levels."""
        
        if complexity_level == "simple":
            sections = [
                {
                    'section_id': 'intro',
                    'section_name': 'Introduction',
                    'description': 'Basic introduction to the topic',
                    'parent_section_id': None
                }
            ]
            propositions = [
                {
                    'id': 'prop-1',
                    'statement': 'This is a simple concept to understand',
                    'supporting_evidence': ['Basic evidence'],
                    'sections': ['intro']
                }
            ]
        elif complexity_level == "medium":
            sections = [
                {
                    'section_id': 'intro',
                    'section_name': 'Introduction',
                    'description': 'Introduction to machine learning concepts',
                    'parent_section_id': None
                },
                {
                    'section_id': 'algorithms',
                    'section_name': 'Core Algorithms',
                    'description': 'Fundamental machine learning algorithms',
                    'parent_section_id': 'intro'
                },
                {
                    'section_id': 'applications',
                    'section_name': 'Applications',
                    'description': 'Real-world applications of machine learning',
                    'parent_section_id': 'algorithms'
                }
            ]
            propositions = [
                {
                    'id': 'prop-1',
                    'statement': 'Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.',
                    'supporting_evidence': ['Research papers', 'Industry applications', 'Academic consensus'],
                    'sections': ['intro']
                },
                {
                    'id': 'prop-2',
                    'statement': 'Supervised learning algorithms learn from labeled training data to make predictions on new, unseen data.',
                    'supporting_evidence': ['Classification examples', 'Regression analysis', 'Cross-validation studies'],
                    'sections': ['algorithms']
                },
                {
                    'id': 'prop-3',
                    'statement': 'Deep learning networks use multiple layers of artificial neurons to model complex patterns in data.',
                    'supporting_evidence': ['Neural network research', 'Image recognition breakthroughs', 'Natural language processing advances'],
                    'sections': ['algorithms']
                }
            ]
        else:  # complex
            sections = [
                {
                    'section_id': f'section-{i}',
                    'section_name': f'Section {i}',
                    'description': f'This is a detailed section about topic {i} with comprehensive coverage of multiple subtopics and in-depth analysis.',
                    'parent_section_id': f'section-{i-1}' if i > 1 else None
                }
                for i in range(1, 11)  # 10 sections
            ]
            propositions = [
                {
                    'id': f'prop-{i}',
                    'statement': f'This is a complex proposition {i} that requires detailed explanation and covers multiple interconnected concepts that build upon each other in a hierarchical manner. The proposition involves sophisticated reasoning and multiple layers of abstraction that demonstrate the complexity of the subject matter.',
                    'supporting_evidence': [
                        f'Research study {i}A showing empirical evidence',
                        f'Theoretical framework {i}B providing conceptual foundation',
                        f'Experimental validation {i}C confirming practical applications',
                        f'Meta-analysis {i}D synthesizing multiple studies'
                    ],
                    'sections': [f'section-{(i-1) % 10 + 1}']
                }
                for i in range(1, 21)  # 20 propositions
            ]
        
        return LearningBlueprint(
            source_id=f'test-blueprint-{complexity_level}',
            source_title=f'Test Blueprint - {complexity_level.title()} Complexity',
            source_type='educational_content',
            source_summary={
                'core_thesis_or_main_argument': f'Testing vector store costs with {complexity_level} complexity blueprint',
                'inferred_purpose': 'Cost analysis and performance testing'
            },
            sections=sections,
            knowledge_primitives={
                'key_propositions_and_facts': propositions,
                'key_entities_and_definitions': [
                    {
                        'id': 'entity-1',
                        'entity': 'Test Entity',
                        'definition': 'An entity used for testing purposes in cost analysis',
                        'category': 'Concept',
                        'sections': [sections[0]['section_id']]
                    }
                ],
                'described_processes_and_steps': [
                    {
                        'id': 'process-1',
                        'process_name': 'Test Process',
                        'description': 'A process used for testing vector store operations',
                        'steps': ['Step 1: Initialize', 'Step 2: Process', 'Step 3: Validate'],
                        'sections': [sections[0]['section_id']]
                    }
                ],
                'identified_relationships': [],
                'implicit_and_open_questions': []
            }
        )
    
    def count_text_stats(self, text_nodes) -> Tuple[int, int]:
        """Count total characters and words in text nodes."""
        total_chars = sum(len(node.content) for node in text_nodes)
        total_words = sum(len(node.content.split()) for node in text_nodes)
        return total_chars, total_words
    
    def estimate_embedding_cost(self, total_chars: int, embedding_calls: int) -> float:
        """Estimate cost for embedding operations."""
        # Google Gemini embedding cost per 1K characters
        cost = (total_chars / 1000) * self.GEMINI_EMBEDDING_COST_PER_1K_CHARS
        return cost
    
    async def test_blueprint_parsing(self, complexity_level: str) -> CostMetrics:
        """Test blueprint parsing costs."""
        print(f"\n🔍 Testing blueprint parsing - {complexity_level} complexity...")
        
        start_time = time.time()
        
        # Generate test blueprint
        blueprint = self.generate_test_blueprint(complexity_level)
        
        # Parse blueprint
        text_nodes = self.blueprint_parser.parse_blueprint(blueprint)
        
        processing_time = time.time() - start_time
        
        # Calculate stats
        total_chars, total_words = self.count_text_stats(text_nodes)
        
        # Parsing itself doesn't use embeddings, so no API cost
        metrics = CostMetrics(
            operation=f"Blueprint Parsing ({complexity_level})",
            text_nodes_count=len(text_nodes),
            total_characters=total_chars,
            total_words=total_words,
            embedding_calls=0,
            processing_time=processing_time,
            estimated_cost=0.0,
            cost_per_word=0.0,
            cost_per_node=0.0
        )
        
        self.results.append(metrics)
        
        print(f"✓ Parsed {len(text_nodes)} nodes in {processing_time:.3f}s")
        print(f"  Total characters: {total_chars:,}")
        print(f"  Total words: {total_words:,}")
        
        return metrics
    
    async def test_embedding_generation(self, complexity_level: str) -> CostMetrics:
        """Test embedding generation costs."""
        print(f"\n🔍 Testing embedding generation - {complexity_level} complexity...")
        
        start_time = time.time()
        
        # Generate and parse blueprint
        blueprint = self.generate_test_blueprint(complexity_level)
        text_nodes = self.blueprint_parser.parse_blueprint(blueprint)
        
        # Generate embeddings for each node
        embedding_calls = 0
        if self.api_key != 'test-key':  # Only if we have a real API key
            try:
                await self.embedding_service.initialize()
                for node in text_nodes:
                    await self.embedding_service.embed_text(node.content)
                    embedding_calls += 1
            except Exception as e:
                print(f"⚠️  Embedding generation failed (using dummy API key): {e}")
                embedding_calls = len(text_nodes)  # Estimate
        else:
            embedding_calls = len(text_nodes)  # Estimate
        
        processing_time = time.time() - start_time
        
        # Calculate stats
        total_chars, total_words = self.count_text_stats(text_nodes)
        estimated_cost = self.estimate_embedding_cost(total_chars, embedding_calls)
        
        metrics = CostMetrics(
            operation=f"Embedding Generation ({complexity_level})",
            text_nodes_count=len(text_nodes),
            total_characters=total_chars,
            total_words=total_words,
            embedding_calls=embedding_calls,
            processing_time=processing_time,
            estimated_cost=estimated_cost,
            cost_per_word=estimated_cost / total_words if total_words > 0 else 0,
            cost_per_node=estimated_cost / len(text_nodes) if len(text_nodes) > 0 else 0
        )
        
        self.results.append(metrics)
        
        print(f"✓ Generated embeddings for {len(text_nodes)} nodes in {processing_time:.3f}s")
        print(f"  Embedding calls: {embedding_calls}")
        print(f"  Estimated cost: ${estimated_cost:.6f}")
        print(f"  Cost per word: ${metrics.cost_per_word:.8f}")
        
        return metrics
    
    async def test_vector_indexing(self, complexity_level: str) -> CostMetrics:
        """Test vector indexing costs."""
        print(f"\n🔍 Testing vector indexing - {complexity_level} complexity...")
        
        start_time = time.time()
        
        # Generate and parse blueprint
        blueprint = self.generate_test_blueprint(complexity_level)
        text_nodes = self.blueprint_parser.parse_blueprint(blueprint)
        
        # Index nodes (this includes embedding generation)
        embedding_calls = len(text_nodes)  # Each node requires one embedding call
        
        # Note: We're not actually indexing to avoid API costs during testing
        # In real usage, this would call: await self.indexing_pipeline.index_nodes(text_nodes)
        
        processing_time = time.time() - start_time
        
        # Calculate stats
        total_chars, total_words = self.count_text_stats(text_nodes)
        estimated_cost = self.estimate_embedding_cost(total_chars, embedding_calls)
        
        metrics = CostMetrics(
            operation=f"Vector Indexing ({complexity_level})",
            text_nodes_count=len(text_nodes),
            total_characters=total_chars,
            total_words=total_words,
            embedding_calls=embedding_calls,
            processing_time=processing_time,
            estimated_cost=estimated_cost,
            cost_per_word=estimated_cost / total_words if total_words > 0 else 0,
            cost_per_node=estimated_cost / len(text_nodes) if len(text_nodes) > 0 else 0
        )
        
        self.results.append(metrics)
        
        print(f"✓ Estimated indexing cost for {len(text_nodes)} nodes")
        print(f"  Estimated cost: ${estimated_cost:.6f}")
        print(f"  Cost per node: ${metrics.cost_per_node:.6f}")
        
        return metrics
    
    async def test_search_operations(self) -> CostMetrics:
        """Test search operation costs."""
        print(f"\n🔍 Testing search operations...")
        
        start_time = time.time()
        
        # Test queries
        test_queries = [
            "machine learning algorithms",
            "deep learning neural networks",
            "supervised learning classification",
            "artificial intelligence applications",
            "data science methodology"
        ]
        
        embedding_calls = len(test_queries)  # Each query requires one embedding call
        total_chars = sum(len(query) for query in test_queries)
        total_words = sum(len(query.split()) for query in test_queries)
        
        # Note: We're not actually searching to avoid API costs during testing
        # In real usage, this would call: await self.search_service.search(query)
        
        processing_time = time.time() - start_time
        
        estimated_cost = self.estimate_embedding_cost(total_chars, embedding_calls)
        
        metrics = CostMetrics(
            operation="Search Operations",
            text_nodes_count=0,  # N/A for search
            total_characters=total_chars,
            total_words=total_words,
            embedding_calls=embedding_calls,
            processing_time=processing_time,
            estimated_cost=estimated_cost,
            cost_per_word=estimated_cost / total_words if total_words > 0 else 0,
            cost_per_node=0.0  # N/A for search
        )
        
        self.results.append(metrics)
        
        print(f"✓ Estimated search cost for {len(test_queries)} queries")
        print(f"  Estimated cost: ${estimated_cost:.6f}")
        print(f"  Cost per query: ${estimated_cost / len(test_queries):.6f}")
        
        return metrics
    
    def generate_cost_report(self) -> Dict:
        """Generate comprehensive cost analysis report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_operations": len(self.results),
                "total_estimated_cost": sum(r.estimated_cost for r in self.results),
                "total_embedding_calls": sum(r.embedding_calls for r in self.results),
                "total_processing_time": sum(r.processing_time for r in self.results)
            },
            "operations": []
        }
        
        for result in self.results:
            report["operations"].append({
                "operation": result.operation,
                "text_nodes_count": result.text_nodes_count,
                "total_characters": result.total_characters,
                "total_words": result.total_words,
                "embedding_calls": result.embedding_calls,
                "processing_time": result.processing_time,
                "estimated_cost": result.estimated_cost,
                "cost_per_word": result.cost_per_word,
                "cost_per_node": result.cost_per_node
            })
        
        return report
    
    def print_cost_summary(self):
        """Print a formatted cost summary."""
        print("\n" + "=" * 80)
        print("VECTOR STORE COST ANALYSIS SUMMARY")
        print("=" * 80)
        
        total_cost = sum(r.estimated_cost for r in self.results)
        total_calls = sum(r.embedding_calls for r in self.results)
        total_time = sum(r.processing_time for r in self.results)
        
        print(f"📊 Overall Statistics:")
        print(f"  Total Operations: {len(self.results)}")
        print(f"  Total Estimated Cost: ${total_cost:.6f}")
        print(f"  Total Embedding Calls: {total_calls}")
        print(f"  Total Processing Time: {total_time:.3f}s")
        
        print(f"\n📈 Cost Breakdown by Operation:")
        for result in self.results:
            print(f"  {result.operation:30} | ${result.estimated_cost:.6f} | {result.embedding_calls:3} calls | {result.processing_time:.3f}s")
        
        print(f"\n💡 Cost Insights:")
        if total_cost > 0:
            avg_cost_per_call = total_cost / total_calls if total_calls > 0 else 0
            print(f"  Average cost per embedding call: ${avg_cost_per_call:.8f}")
            
            # Find most expensive operation
            most_expensive = max(self.results, key=lambda r: r.estimated_cost)
            print(f"  Most expensive operation: {most_expensive.operation} (${most_expensive.estimated_cost:.6f})")
            
            # Estimate costs for different scales
            print(f"\n📊 Scaling Estimates:")
            print(f"  Cost for 100 blueprints (medium): ${total_cost * 33:.4f}")  # Rough estimate
            print(f"  Cost for 1,000 blueprints (medium): ${total_cost * 333:.4f}")
            print(f"  Cost for 10,000 blueprints (medium): ${total_cost * 3333:.4f}")
        
        print(f"\n⚠️  Note: These are estimates based on Google Gemini embedding pricing.")
        print(f"   Actual costs may vary based on API usage, caching, and pricing changes.")


async def main():
    """Main test execution."""
    print("🚀 Vector Store Cost Analysis")
    print("=" * 50)
    
    tester = VectorStoreCostTester()
    
    # Test different complexity levels
    complexity_levels = ["simple", "medium", "complex"]
    
    for level in complexity_levels:
        await tester.test_blueprint_parsing(level)
        await tester.test_embedding_generation(level)
        await tester.test_vector_indexing(level)
    
    # Test search operations
    await tester.test_search_operations()
    
    # Generate and display results
    tester.print_cost_summary()
    
    # Save detailed report
    report = tester.generate_cost_report()
    with open("vector_store_cost_analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: vector_store_cost_analysis.json")


if __name__ == "__main__":
    asyncio.run(main())
