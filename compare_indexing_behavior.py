import requests
import json
import time

def test_indexing_with_debug_logs():
    print("=== Testing Core API vs Direct Indexing with Debug Logs ===\n")
    
    # Step 1: Authenticate with Core API
    print("Step 1: Authenticating with Core API...")
    auth_response = requests.post('http://localhost:3000/api/auth/login', json={
        'email': 'test@example.com',
        'password': 'password123'
    })
    
    if auth_response.status_code != 200:
        print(f"❌ Authentication failed: {auth_response.text}")
        return
    
    token = auth_response.json()['token']
    print("✅ Authentication successful")
    
    # Step 2: Create a new blueprint via Core API (this should trigger AI API indexing)
    print("\nStep 2: Creating blueprint via Core API (will trigger indexing)...")
    blueprint_data = {
        'sourceText': 'Machine Learning algorithms can be categorized into supervised learning (with labeled data), unsupervised learning (without labels), and reinforcement learning (learning through interaction and rewards).'
    }
    
    create_response = requests.post(
        'http://localhost:3000/api/ai-rag/learning-blueprints',
        headers={'Authorization': f'Bearer {token}'},
        json=blueprint_data
    )
    
    if create_response.status_code != 201:
        print(f"❌ Blueprint creation failed: {create_response.status_code} - {create_response.text}")
        return
    
    blueprint_id = create_response.json().get('id')
    print(f"✅ Blueprint created with ID: {blueprint_id}")
    print("   (This should have triggered AI API indexing - check logs)")
    
    # Step 3: Wait a moment for indexing to process
    time.sleep(5)
    
    # Step 4: Make a direct API call with similar content
    print(f"\nStep 3: Making direct AI API call for comparison...")
    direct_payload = {
        'blueprint_id': f'{blueprint_id}-direct-comparison',
        'blueprint_json': {
            'source_text': blueprint_data['sourceText'],
            'source_summary': 'Machine Learning algorithm categories overview',
            'sections': [
                {
                    'section_id': 'ml_categories',
                    'section_name': 'Machine Learning Categories',
                    'description': blueprint_data['sourceText'],
                    'parent_section_id': None
                }
            ],
            'knowledge_primitives': {
                'key_propositions_and_facts': [
                    {'proposition': 'Machine Learning algorithms can be categorized into three main types'}
                ],
                'key_entities_and_definitions': [
                    {'entity': 'Supervised Learning', 'definition': 'Learning with labeled data'},
                    {'entity': 'Unsupervised Learning', 'definition': 'Learning without labels'},
                    {'entity': 'Reinforcement Learning', 'definition': 'Learning through interaction and rewards'}
                ],
                'described_processes_and_steps': [],
                'identified_relationships': [
                    {'relationship': 'Machine Learning algorithms are categorized by their learning approach'}
                ]
            }
        }
    }
    
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
        print(f"   - Nodes processed: {direct_result.get('nodes_processed')}")
        print(f"   - Vectors stored: {direct_result.get('vectors_stored')}")
        print(f"   - Success rate: {direct_result.get('success_rate')}")
    else:
        print(f"❌ Direct AI API call failed: {direct_response.status_code}")
        print(f"   Error: {direct_response.text}")
    
    # Step 5: Check Core API blueprint status
    print(f"\nStep 4: Checking Core API blueprint indexing status...")
    status_response = requests.get(
        f'http://localhost:8000/api/v1/blueprints/{blueprint_id}/status',
        headers={'Authorization': 'Bearer test_api_key_123'}
    )
    
    if status_response.status_code == 200:
        status = status_response.json()
        print(f"Core API blueprint status:")
        print(f"   - Status: {status.get('status')}")
        print(f"   - Is indexed: {status.get('is_indexed')}")
        print(f"   - Node count: {status.get('node_count')}")
    else:
        print(f"❌ Status check failed: {status_response.status_code}")
    
    print(f"\n=== ANALYSIS ===")
    print(f"🔍 Check the AI API server logs to see:")
    print(f"   1. Debug output from Core API indexing request (blueprint {blueprint_id})")
    print(f"   2. Debug output from direct API request ({blueprint_id}-direct-comparison)")
    print(f"   3. Compare the processing differences between the two")

if __name__ == "__main__":
    test_indexing_with_debug_logs()
