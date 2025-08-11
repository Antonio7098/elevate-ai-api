import sys
sys.path.append('/home/antonio/programming/elevate/core_and_ai/elevate-ai-api')

import json
import requests

def debug_database_blueprint():
    # Get auth token
    auth_response = requests.post('http://localhost:3000/api/auth/login', json={
        'email': 'test@example.com',
        'password': 'password123'
    })
    token = auth_response.json()['token']
    
    # Get blueprint 78 from Core API endpoint
    blueprint_response = requests.get('http://localhost:3000/api/ai-rag/learning-blueprints/78', 
                                      headers={'Authorization': f'Bearer {token}'})
    
    print("=== Core API Response ===")
    response_data = blueprint_response.json()
    print(f"Response status: {blueprint_response.status_code}")
    print(f"Response keys: {list(response_data.keys())}")
    print(f"blueprintJson type: {type(response_data.get('blueprintJson'))}")
    print(f"blueprintJson is null: {response_data.get('blueprintJson') is None}")
    
    if response_data.get('blueprintJson'):
        print(f"blueprintJson keys: {list(response_data['blueprintJson'].keys()) if isinstance(response_data['blueprintJson'], dict) else 'Not a dict'}")
    else:
        print("blueprintJson is null or falsy")
    
    print(f"Full response data: {json.dumps(response_data, indent=2)[:500]}...")

if __name__ == "__main__":
    debug_database_blueprint()
