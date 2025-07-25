#!/usr/bin/env python3
"""
Test script to verify the blueprint translation layer works correctly
and resolves the node_count: 0 indexing issue.
"""

import os
import sys
import asyncio
import json
from dotenv import load_dotenv

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Load environment variables
load_dotenv(override=True)

from app.utils.blueprint_translator import translate_blueprint, BlueprintTranslationError
from app.core.indexing_pipeline import IndexingPipeline
from app.core.services import initialize_services


def create_test_blueprint():
    """Create a test blueprint in the format from the integration layer."""
    return {
        "id": "test_blueprint_123",
        "title": "Machine Learning Fundamentals", 
        "description": "Introduction to machine learning concepts and algorithms",
        "user_id": "test_user_456",
        "learning_objectives": [
            "Understand supervised vs unsupervised learning",
            "Learn about neural networks and deep learning",
            "Apply machine learning algorithms to real problems"
        ],
        "content_sections": [
            {
                "id": "intro",
                "title": "Introduction to ML",
                "content": "Machine learning is a subset of artificial intelligence..."
            },
            {
                "id": "algorithms", 
                "title": "ML Algorithms",
                "content": "Common algorithms include linear regression, decision trees..."
            }
        ],
        "summary": "Comprehensive introduction to machine learning concepts",
        "created_at": "2025-01-20T10:00:00Z"
    }


async def test_translation_layer():
    """Test the blueprint translation layer."""
    print("🔄 Testing Blueprint Translation Layer...")
    
    # Test 1: Translation
    test_blueprint = create_test_blueprint()
    print(f"📝 Original blueprint keys: {list(test_blueprint.keys())}")
    
    try:
        translated = translate_blueprint(test_blueprint)
        print(f"✅ Translation successful!")
        print(f"📋 LearningBlueprint fields:")
        print(f"   - source_id: {translated.source_id}")
        print(f"   - source_title: {translated.source_title}")
        print(f"   - source_type: {translated.source_type}")
        print(f"   - sections: {len(translated.sections)} sections")
        print(f"   - knowledge_primitives: {len(translated.knowledge_primitives.key_propositions_and_facts)} propositions")
        
        return translated
        
    except BlueprintTranslationError as e:
        print(f"❌ Translation failed: {e}")
        return None


async def test_blueprint_indexing():
    """Test end-to-end blueprint indexing with translation."""
    print("\n🚀 Testing End-to-End Blueprint Indexing...")
    
    # Initialize services
    print("🔧 Initializing services...")
    try:
        await initialize_services()
        print("✅ Services initialized successfully")
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False
    
    # Test translation
    translated_blueprint = await test_translation_layer()
    if not translated_blueprint:
        return False
    
    # Test indexing
    print("\n📊 Testing indexing pipeline...")
    try:
        pipeline = IndexingPipeline()
        
        result = await pipeline.index_blueprint(translated_blueprint)
        
        print(f"✅ Indexing completed!")
        print(f"📈 Results:")
        node_count = result.get('processed_nodes', 0)
        processing_time = result.get('elapsed_seconds', 0.0)
        print(f"   - Node count: {node_count}")
        print(f"   - Processing time: {processing_time:.2f}s")
        print(f"   - Success: {node_count > 0}")
        
        return node_count > 0
        
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_api_format():
    """Test the API request format that would be sent from integration layer."""
    print("\n🌐 Testing API Request Format...")
    
    test_blueprint = create_test_blueprint()
    
    # Simulate the IndexBlueprintRequest format
    api_request = {
        "blueprint_json": test_blueprint,
        "force_reindex": True
    }
    
    print(f"📡 API request format:")
    print(f"   - blueprint_json keys: {list(api_request['blueprint_json'].keys())}")
    print(f"   - force_reindex: {api_request['force_reindex']}")
    
    # Test translation from API format
    try:
        translated = translate_blueprint(api_request["blueprint_json"])
        user_id = api_request["blueprint_json"].get('user_id') or api_request["blueprint_json"].get('userId', 'default')
        
        print(f"✅ API format translation successful!")
        print(f"   - Extracted user_id: {user_id}")
        print(f"   - Translated source_id: {translated.source_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ API format translation failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🧪 Blueprint Translation and Indexing Test Suite")
    print("=" * 60)
    
    # Test 1: Translation layer
    translation_success = await test_translation_layer() is not None
    
    # Test 2: API format compatibility  
    api_success = await test_api_format()
    
    # Test 3: End-to-end indexing
    indexing_success = await test_blueprint_indexing()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   ✅ Translation Layer: {'PASS' if translation_success else 'FAIL'}")
    print(f"   ✅ API Format Compatibility: {'PASS' if api_success else 'FAIL'}")
    print(f"   ✅ End-to-End Indexing: {'PASS' if indexing_success else 'FAIL'}")
    
    overall_success = translation_success and api_success and indexing_success
    print(f"\n🎯 Overall Result: {'SUCCESS' if overall_success else 'FAILURE'}")
    
    if overall_success:
        print("🎉 Blueprint indexing issue has been resolved!")
        print("   The translation layer successfully converts arbitrary JSON to LearningBlueprint format")
        print("   and enables vector creation in the Pinecone database.")
    else:
        print("⚠️  Some tests failed - further investigation needed.")
    
    return overall_success


if __name__ == "__main__":
    asyncio.run(main())
