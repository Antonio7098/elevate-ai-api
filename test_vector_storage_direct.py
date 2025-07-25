import sys
sys.path.append('/home/antonio/programming/elevate/core_and_ai/elevate-ai-api')

import asyncio
import requests
from app.core.indexing_pipeline import IndexingPipeline
from app.models.learning_blueprint import LearningBlueprint

async def test_vector_storage():
    print("=== Testing Vector Storage with Blueprint 78 ===")
    
    # Get auth token
    auth_response = requests.post('http://localhost:3000/api/auth/login', json={
        'email': 'test@example.com',
        'password': 'password123'
    })
    token = auth_response.json()['token']
    
    # Get blueprint 78 data (which we know has valid sections and knowledge primitives)
    blueprint_response = requests.get('http://localhost:3000/api/ai-rag/learning-blueprints/78', 
                                      headers={'Authorization': f'Bearer {token}'})
    blueprint_data = blueprint_response.json()['blueprintJson']
    
    print(f"Blueprint sections: {len(blueprint_data.get('sections', []))}")
    print(f"Knowledge primitives: {len(blueprint_data.get('knowledge_primitives', {}).get('key_propositions_and_facts', []))}")
    
    # Test indexing pipeline directly
    print("\n=== Testing IndexingPipeline ===")
    pipeline = IndexingPipeline()
    
    try:
        # Convert to LearningBlueprint object
        learning_blueprint = LearningBlueprint(**blueprint_data)
        print(f"Created LearningBlueprint object: {learning_blueprint.source_id}")
        
        # Index the blueprint and capture detailed output
        result = await pipeline.index_blueprint(learning_blueprint)
        
        print(f"\n=== Indexing Result ===")
        print(f"Status: {result.get('status', 'Unknown')}")
        print(f"Total nodes: {result.get('total_nodes', 0)}")
        print(f"Nodes processed: {result.get('nodes_processed', 0)}")
        print(f"Embeddings generated: {result.get('embeddings_generated', 0)}")
        print(f"Vectors stored: {result.get('vectors_stored', 0)}")
        print(f"Errors: {result.get('errors', [])}")
        
        if result.get('errors'):
            print(f"Error details: {result['errors']}")
            
    except Exception as e:
        print(f"Indexing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vector_storage())
