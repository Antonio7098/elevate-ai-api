import requests
import json

# Get auth token
auth_response = requests.post('http://localhost:3000/api/auth/login', json={
    'email': 'test@example.com',
    'password': 'password123'
})

print(f"Auth status: {auth_response.status_code}")
if auth_response.status_code == 200:
    token = auth_response.json()['token']
    
    # Test deconstruct endpoint
    deconstruct_response = requests.post(
        'http://localhost:3000/api/ai-rag/deconstruct',
        headers={'Authorization': f'Bearer {token}'},
        json={
            'sourceText': 'Python decorators are a way to modify or extend the functionality of functions without changing their structure. They use the @ symbol.'
        }
    )
    
    print(f"Deconstruct status: {deconstruct_response.status_code}")
    print(f"Deconstruct headers: {deconstruct_response.headers}")
    print(f"Deconstruct content length: {len(deconstruct_response.content)}")
    print(f"Deconstruct content: {deconstruct_response.text[:500]}...")
else:
    print(f"Auth failed: {auth_response.text}")
