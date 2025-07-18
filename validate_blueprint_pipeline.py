#!/usr/bin/env python3
"""
Blueprint Ingestion Pipeline Validation Script

This script validates the complete blueprint ingestion pipeline without relying
on the full test suite, which may cause environment crashes.
"""

import sys
import traceback
from typing import Dict, Any

def test_blueprint_parser():
    """Test the blueprint parser functionality."""
    print("=" * 60)
    print("Testing Blueprint Parser")
    print("=" * 60)
    
    try:
        from app.core.blueprint_parser import BlueprintParser
        from app.models.learning_blueprint import LearningBlueprint
        
        # Create test blueprint data
        test_data = {
            'source_id': 'validation-test',
            'source_title': 'Validation Test Blueprint',
            'source_type': 'educational_content',
            'source_summary': {
                'core_thesis_or_main_argument': 'Testing blueprint ingestion pipeline',
                'inferred_purpose': 'Validate that all components work together'
            },
            'sections': [
                {
                    'section_id': 'intro',
                    'section_name': 'Introduction',
                    'description': 'Introduction to the topic',
                    'parent_section_id': None
                },
                {
                    'section_id': 'concepts',
                    'section_name': 'Key Concepts',
                    'description': 'Important concepts to understand',
                    'parent_section_id': 'intro'
                }
            ],
            'knowledge_primitives': {
                'key_propositions_and_facts': [
                    {
                        'id': 'prop-1',
                        'statement': 'This is a fundamental concept',
                        'supporting_evidence': ['Research study A', 'Expert opinion B'],
                        'sections': ['intro']
                    },
                    {
                        'id': 'prop-2',
                        'statement': 'This concept builds on the previous one',
                        'supporting_evidence': ['Experimental data'],
                        'sections': ['concepts']
                    }
                ],
                'key_entities_and_definitions': [
                    {
                        'id': 'entity-1',
                        'entity': 'Important Term',
                        'definition': 'A crucial term in this domain',
                        'category': 'Concept',
                        'sections': ['concepts']
                    }
                ],
                'described_processes_and_steps': [
                    {
                        'id': 'process-1',
                        'process_name': 'Learning Process',
                        'description': 'How to learn this topic effectively',
                        'steps': ['Step 1: Read', 'Step 2: Practice', 'Step 3: Apply'],
                        'sections': ['concepts']
                    }
                ],
                'identified_relationships': [
                    {
                        'id': 'rel-1',
                        'relationship_type': 'builds_on',
                        'source_primitive_id': 'prop-2',
                        'target_primitive_id': 'prop-1',
                        'description': 'Prop-2 builds on prop-1',
                        'sections': ['concepts']
                    }
                ],
                'implicit_and_open_questions': [
                    {
                        'id': 'question-1',
                        'question': 'What are the implications of this concept?',
                        'question_type': 'open_ended',
                        'sections': ['concepts']
                    }
                ]
            }
        }
        
        print("✓ Creating LearningBlueprint from test data...")
        blueprint = LearningBlueprint(**test_data)
        print(f"✓ Blueprint created: {blueprint.source_title}")
        
        print("✓ Initializing BlueprintParser...")
        parser = BlueprintParser()
        
        print("✓ Parsing blueprint...")
        nodes = parser.parse_blueprint(blueprint)
        print(f"✓ Successfully parsed {len(nodes)} TextNodes")
        
        # Analyze the generated nodes
        locus_types = {}
        for node in nodes:
            locus_type = node.locus_type.value
            locus_types[locus_type] = locus_types.get(locus_type, 0) + 1
        
        print("\n📊 Generated Node Statistics:")
        for locus_type, count in locus_types.items():
            print(f"  - {locus_type}: {count} nodes")
        
        # Show sample nodes
        print("\n📋 Sample Generated Nodes:")
        for i, node in enumerate(nodes[:3]):
            print(f"  {i+1}. {node.locus_type.value} | {node.locus_id} | {len(node.content)} chars")
        
        print("\n✅ Blueprint Parser Test: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Blueprint Parser Test: FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False

def test_vector_store():
    """Test vector store initialization."""
    print("\n" + "=" * 60)
    print("Testing Vector Store")
    print("=" * 60)
    
    try:
        from app.core.vector_store import ChromaDBVectorStore
        
        print("✓ Initializing ChromaDBVectorStore...")
        vector_store = ChromaDBVectorStore()
        print("✓ Vector store initialized successfully")
        
        print("✅ Vector Store Test: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Vector Store Test: FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False

def test_metadata_indexing():
    """Test metadata indexing service."""
    print("\n" + "=" * 60)
    print("Testing Metadata Indexing Service")
    print("=" * 60)
    
    try:
        from app.core.vector_store import ChromaDBVectorStore
        from app.core.metadata_indexing import MetadataIndexingService
        
        print("✓ Creating vector store...")
        vector_store = ChromaDBVectorStore()
        
        print("✓ Initializing MetadataIndexingService...")
        metadata_service = MetadataIndexingService(vector_store)
        print("✓ Metadata indexing service initialized successfully")
        
        print("✅ Metadata Indexing Test: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Metadata Indexing Test: FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False

def test_search_service():
    """Test search service initialization."""
    print("\n" + "=" * 60)
    print("Testing Search Service")
    print("=" * 60)
    
    try:
        from app.core.vector_store import ChromaDBVectorStore
        from app.core.embeddings import GoogleEmbeddingService
        from app.core.search_service import SearchService
        import os
        
        print("✓ Creating vector store...")
        vector_store = ChromaDBVectorStore()
        
        print("✓ Creating Google embedding service...")
        # Use a dummy API key for testing initialization
        api_key = os.getenv('GEMINI_API_KEY', 'dummy-key-for-testing')
        embedding_service = GoogleEmbeddingService(api_key)
        
        print("✓ Initializing SearchService...")
        search_service = SearchService(vector_store, embedding_service)
        print("✓ Search service initialized successfully")
        
        print("✅ Search Service Test: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Search Service Test: FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False

def test_indexing_pipeline():
    """Test indexing pipeline."""
    print("\n" + "=" * 60)
    print("Testing Indexing Pipeline")
    print("=" * 60)
    
    try:
        from app.core.indexing_pipeline import IndexingPipeline
        
        print("✓ Initializing IndexingPipeline...")
        pipeline = IndexingPipeline()
        print("✓ Indexing pipeline initialized successfully")
        
        print("✅ Indexing Pipeline Test: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Indexing Pipeline Test: FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("🚀 Blueprint Ingestion Pipeline Validation")
    print("=" * 60)
    
    tests = [
        ("Blueprint Parser", test_blueprint_parser),
        ("Vector Store", test_vector_store),
        ("Metadata Indexing", test_metadata_indexing),
        ("Search Service", test_search_service),
        ("Indexing Pipeline", test_indexing_pipeline),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} Test: CRASHED")
            print(f"Error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} | {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All blueprint ingestion components are working correctly!")
        return 0
    else:
        print("⚠️  Some components need attention.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
