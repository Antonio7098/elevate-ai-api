#!/usr/bin/env python3
"""
Test script for blueprint ingestion functionality.

This script tests the blueprint parsing and indexing pipeline without
requiring the full vector database setup.
"""

import json
import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.learning_blueprint import LearningBlueprint
from app.core.blueprint_parser import BlueprintParser, BlueprintParserError


def load_sample_blueprint(blueprint_file: str) -> LearningBlueprint:
    """Load a sample blueprint from JSON file."""
    try:
        with open(blueprint_file, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        
        # Handle both direct blueprint format and deconstruction result format
        if 'blueprint_json' in file_data:
            # This is a deconstruction result file
            blueprint_data = file_data['blueprint_json']
        else:
            # This is a direct blueprint file
            blueprint_data = file_data
        
        return LearningBlueprint(**blueprint_data)
    except Exception as e:
        print(f"Error loading blueprint from {blueprint_file}: {e}")
        raise


def test_blueprint_parsing(blueprint_file: str):
    """Test blueprint parsing functionality."""
    print(f"\n=== Testing Blueprint Parsing ===")
    print(f"Loading blueprint from: {blueprint_file}")
    
    try:
        # Load the blueprint
        blueprint = load_sample_blueprint(blueprint_file)
        print(f"✓ Successfully loaded blueprint: {blueprint.source_id}")
        print(f"  Title: {blueprint.source_title}")
        print(f"  Type: {blueprint.source_type}")
        
        # Get blueprint statistics
        parser = BlueprintParser()
        stats = parser.get_blueprint_stats(blueprint)
        
        print(f"\nBlueprint Statistics:")
        print(f"  Sections: {stats['sections_count']}")
        print(f"  Propositions: {stats['propositions_count']}")
        print(f"  Entities: {stats['entities_count']}")
        print(f"  Processes: {stats['processes_count']}")
        print(f"  Relationships: {stats['relationships_count']}")
        print(f"  Questions: {stats['questions_count']}")
        print(f"  Total Primitives: {stats['total_primitives']}")
        
        # Parse blueprint into TextNodes
        print(f"\nParsing blueprint into TextNodes...")
        nodes = parser.parse_blueprint(blueprint)
        
        print(f"✓ Successfully parsed {len(nodes)} TextNodes")
        
        # Analyze node types
        node_types = {}
        uue_stages = {}
        chunk_counts = {}
        
        for node in nodes:
            # Count by locus type
            if node.locus_type:
                locus_type = node.locus_type.value
                node_types[locus_type] = node_types.get(locus_type, 0) + 1
            
            # Count by UUE stage
            if node.uue_stage:
                uue_stage = node.uue_stage.value
                uue_stages[uue_stage] = uue_stages.get(uue_stage, 0) + 1
            
            # Count chunks per locus
            if node.locus_id:
                chunk_counts[node.locus_id] = chunk_counts.get(node.locus_id, 0) + 1
        
        print(f"\nTextNode Analysis:")
        print(f"  Locus Types: {node_types}")
        print(f"  UUE Stages: {uue_stages}")
        print(f"  Average chunks per locus: {sum(chunk_counts.values()) / len(chunk_counts) if chunk_counts else 0:.1f}")
        
        # Show sample nodes
        print(f"\nSample TextNodes:")
        for i, node in enumerate(nodes[:3]):  # Show first 3 nodes
            print(f"  {i+1}. ID: {node.id}")
            print(f"     Locus: {node.locus_id} ({node.locus_type.value if node.locus_type else 'None'})")
            print(f"     UUE Stage: {node.uue_stage.value if node.uue_stage else 'None'}")
            print(f"     Content: {node.content[:100]}...")
            print(f"     Word Count: {node.word_count}")
            print()
        
        return True
        
    except BlueprintParserError as e:
        print(f"✗ Blueprint parsing error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_content_chunking(blueprint_file: str):
    """Test content chunking functionality."""
    print(f"\n=== Testing Content Chunking ===")
    
    try:
        blueprint = load_sample_blueprint(blueprint_file)
        parser = BlueprintParser()
        
        # Test chunking with different content sizes
        test_contents = [
            "Short content",
            "This is a medium length content that should fit in one chunk without any issues.",
            "Long content " * 200,  # ~2000 words
            "Very long content " * 500  # ~5000 words
        ]
        
        for i, content in enumerate(test_contents):
            print(f"\nTest {i+1}: {len(content.split())} words")
            chunks = parser._chunk_content(content, f"test_locus_{i}")
            print(f"  Chunks: {len(chunks)}")
            print(f"  Total words: {sum(len(chunk.split()) for chunk in chunks)}")
            
            if len(chunks) > 1:
                print(f"  Chunk sizes: {[len(chunk.split()) for chunk in chunks]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Content chunking error: {e}")
        return False


def main():
    """Main test function."""
    print("Blueprint Ingestion Test Suite")
    print("=" * 50)
    
    # Find sample blueprint files
    deconstructions_dir = Path("deconstructions")
    if not deconstructions_dir.exists():
        print(f"✗ Deconstructions directory not found: {deconstructions_dir}")
        return False
    
    blueprint_files = list(deconstructions_dir.glob("*.json"))
    if not blueprint_files:
        print(f"✗ No blueprint files found in {deconstructions_dir}")
        return False
    
    print(f"Found {len(blueprint_files)} blueprint files")
    
    # Test with the first blueprint file
    test_file = blueprint_files[0]
    print(f"Using test file: {test_file}")
    
    # Run tests
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Blueprint parsing
    if test_blueprint_parsing(str(test_file)):
        tests_passed += 1
    
    # Test 2: Content chunking
    if test_content_chunking(str(test_file)):
        tests_passed += 1
    
    # Summary
    print(f"\n=== Test Summary ===")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! Blueprint ingestion functionality is working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 