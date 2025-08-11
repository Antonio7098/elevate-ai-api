import requests
import json
import time

def test_core_api_vs_direct_call():
    print("=== Debugging Core API vs Direct AI API Calls ===\n")
    
    # Step 1: Create a blueprint via Core API (like the E2E test)
    print("Step 1: Authenticate with Core API...")
    auth_response = requests.post('http://localhost:3000/api/auth/login', json={
        'email': 'test@example.com',
        'password': 'password123'
    })
    
    if auth_response.status_code != 200:
        print(f"❌ Authentication failed: {auth_response.text}")
        return
    
    token = auth_response.json()['token']
    print("✅ Authentication successful")
    
    # Step 2: Create blueprint and monitor logs
    print("\nStep 2: Creating blueprint via Core API (ID will be 80)...")
    blueprint_data = {
        'sourceText': 'Python decorators are a powerful feature that allows you to modify or enhance functions without permanently modifying them. They use the @ syntax.'
    }
    
    create_response = requests.post(
        'http://localhost:3000/api/ai-rag/deconstruct',
        headers={'Authorization': f'Bearer {token}'},
        json=blueprint_data
    )
    
    if create_response.status_code != 200:
        print(f"❌ Blueprint creation failed: {create_response.status_code} - {create_response.text}")
        return
    
    blueprint_id = create_response.json().get('id')
    print(f"✅ Blueprint created with ID: {blueprint_id}")
    
    # Step 3: Wait and check initial status
    time.sleep(3)
    status_response = requests.get(
        f'http://localhost:8000/api/v1/blueprints/{blueprint_id}/status',
        headers={'Authorization': 'Bearer test_api_key_123'}
    )
    
    if status_response.status_code == 200:
        status = status_response.json()
        print(f"Initial AI API status: {status.get('status')} (nodes: {status.get('node_count')})")
    
    # Step 4: Now test the same content with direct AI API call
    print(f"\nStep 3: Testing same content via direct AI API call...")
    
    # Get the blueprint details from Core API
    blueprint_response = requests.get(
        f'http://localhost:3000/api/ai-rag/learning-blueprints/{blueprint_id}',
        headers={'Authorization': f'Bearer {token}'}
    )
    
    if blueprint_response.status_code != 200:
        print(f"❌ Failed to get blueprint details: {blueprint_response.status_code}")
        return
    
    blueprint_full = blueprint_response.json()
    blueprint_json = blueprint_full.get('blueprintJson', {})
    
    print(f"Core API blueprint structure:")
    print(f"  - Source text: {'✅' if blueprint_full.get('sourceText') else '❌'}")
    print(f"  - Blueprint JSON: {'✅' if blueprint_json else '❌'}")
    print(f"  - Sections: {len(blueprint_json.get('sections', []))} sections")
    print(f"  - Knowledge primitives: {'✅' if blueprint_json.get('knowledge_primitives') else '❌'}")
    
    if not blueprint_json or not blueprint_json.get('sections'):
        print("❌ Blueprint JSON structure is invalid - this explains the indexing failure!")
        return
        
    # Direct AI API call using the same structure
    direct_payload = {
        'blueprint_id': f'{blueprint_id}-direct-test',
        'blueprint_json': {
            'source_text': blueprint_full.get('sourceText', ''),
            'source_summary': f"Learning blueprint from: {blueprint_full.get('sourceText', '')[:100]}",
            'sections': blueprint_json.get('sections', []),
            'knowledge_primitives': blueprint_json.get('knowledge_primitives', {})
        }
    }
    
    print(f"\nStep 4: Making direct AI API call with same data...")
    direct_response = requests.post(
        'http://localhost:8000/api/v1/index-blueprint',
        headers={
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test_api_key_123'
        },
        json=direct_payload
    )
    
    if direct_response.status_code == 200:
        direct_result = direct_response.json()
        print(f"✅ Direct AI API call successful:")
        print(f"  - Nodes processed: {direct_result.get('nodes_processed')}")
        print(f"  - Vectors stored: {direct_result.get('vectors_stored')}")
        print(f"  - Success rate: {direct_result.get('success_rate')}")
    else:
        print(f"❌ Direct AI API call failed: {direct_response.status_code}")
        print(f"Error: {direct_response.text}")
        
    print(f"\n=== COMPARISON SUMMARY ===")
    print(f"Core API blueprint creation: ✅ (ID: {blueprint_id})")
    print(f"Core API indexing: ❌ (not indexed)")
    print(f"Direct AI API call: ✅ (5+ nodes processed)" if direct_response.status_code == 200 else "❌ (failed)")
    
    if direct_response.status_code == 200 and blueprint_json:
        print(f"\n💡 ROOT CAUSE: Despite identical data, Core API indexing requests are not processed correctly by AI API.")
        print(f"   Next: Debug AI API request processing and logging for Core API vs direct calls.")

if __name__ == "__main__":
    test_core_api_vs_direct_call()
