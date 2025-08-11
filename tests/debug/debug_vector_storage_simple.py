import sys
sys.path.append('/home/antonio/programming/elevate/core_and_ai/elevate-ai-api')

import asyncio
import requests
import traceback

print("=== Starting Vector Storage Debug ===")

async def simple_test():
    try:
        print("Step 1: Getting auth token...")
        # Get auth token
        auth_response = requests.post('http://localhost:3000/api/auth/login', json={
            'email': 'test@example.com',
            'password': 'password123'
        })
        print(f"Auth response status: {auth_response.status_code}")
        
        if auth_response.status_code != 200:
            print(f"Auth failed: {auth_response.text}")
            return
            
        token = auth_response.json()['token']
        print("Step 2: Auth successful")
        
        # Get blueprint 78 data
        print("Step 3: Getting blueprint 78...")
        blueprint_response = requests.get('http://localhost:3000/api/ai-rag/learning-blueprints/78', 
                                          headers={'Authorization': f'Bearer {token}'})
        print(f"Blueprint response status: {blueprint_response.status_code}")
        
        if blueprint_response.status_code != 200:
            print(f"Blueprint fetch failed: {blueprint_response.text}")
            return
            
        blueprint_data = blueprint_response.json()['blueprintJson']
        print(f"Step 4: Got blueprint data with {len(blueprint_data.get('sections', []))} sections")
        
        # Import IndexingPipeline
        print("Step 5: Importing IndexingPipeline...")
        from app.core.indexing_pipeline import IndexingPipeline
        
        print("Step 6: Importing LearningBlueprint...")
        from app.models.learning_blueprint import LearningBlueprint
        
        print("Step 7: Creating IndexingPipeline...")
        pipeline = IndexingPipeline()
        
        print("Step 8: Creating LearningBlueprint object...")
        learning_blueprint = LearningBlueprint(**blueprint_data)
        print(f"Created LearningBlueprint: {learning_blueprint.source_id}")
        
        print("Step 9: Starting indexing...")
        result = await pipeline.index_blueprint(learning_blueprint)
        
        print("Step 10: Indexing completed!")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(simple_test())
