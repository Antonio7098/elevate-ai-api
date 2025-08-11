import sys
sys.path.append('/home/antonio/programming/elevate/core_and_ai/elevate-ai-api')

import asyncio
import requests
import json
from app.core.indexing_pipeline import IndexingPipeline
from app.models.learning_blueprint import LearningBlueprint

async def test_indexing_with_debug():
    print("=== Creating new blueprint via Core API ===")
    
    # Auth
    auth_response = requests.post('http://localhost:3000/api/auth/login', json={
        'email': 'test@example.com',
        'password': 'password123'
    })
    token = auth_response.json()['token']
    
    # Create blueprint via /deconstruct
    deconstruct_response = requests.post(
        'http://localhost:3000/api/ai-rag/deconstruct',
        headers={'Authorization': f'Bearer {token}'},
        json={
            'sourceText': 'Python decorators are a way to modify or extend the functionality of functions without changing their structure. They use the @ symbol.'
        }
    )
    blueprint_id = deconstruct_response.json()['id']
    print(f"Created blueprint ID: {blueprint_id}")
    
    # Retrieve the blueprint
    blueprint_response = requests.get(f'http://localhost:3000/api/ai-rag/learning-blueprints/{blueprint_id}', 
                                      headers={'Authorization': f'Bearer {token}'})
    blueprint_data = blueprint_response.json()['blueprintJson']
    
    print(f"Blueprint sections: {len(blueprint_data.get('sections', []))}")
    print(f"Blueprint knowledge primitives: {len(blueprint_data.get('knowledge_primitives', {}).get('key_propositions_and_facts', []))}")
    
    # Test indexing pipeline directly
    print("\n=== Testing IndexingPipeline directly ===")
    pipeline = IndexingPipeline()
    
    # Convert to LearningBlueprint object
    learning_blueprint = LearningBlueprint(**blueprint_data)
    
    try:
        result = await pipeline.index_blueprint(learning_blueprint)
        print(f"Indexing result: {result}")
    except Exception as e:
        print(f"Indexing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_indexing_with_debug())
