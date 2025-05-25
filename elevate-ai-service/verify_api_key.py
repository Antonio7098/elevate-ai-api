"""
Script to verify the Core API access key.
"""
import os
import sys
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("CORE_API_ACCESS_KEY")

def verify_api_key():
    """Verify the API key by making a request to the health endpoint."""
    print(f"Verifying API key: {API_KEY}")
    
    if not API_KEY:
        print("❌ Error: CORE_API_ACCESS_KEY is not set in environment variables")
        return False
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        # Test with health endpoint first (doesn't require auth)
        print(f"\nTesting connection to {BASE_URL}/health")
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code != 200:
            print("❌ Failed to connect to the server")
            return False
            
        # Test with a protected endpoint
        print("\nTesting protected endpoint with API key...")
        response = requests.get(
            f"{BASE_URL}/evaluate-answer",  # Any protected endpoint would work
            headers=headers,
            json={"test": "test"},  # Invalid request, but we just want to test auth
            timeout=5
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 401:
            print("❌ API key is invalid or not accepted")
            return False
        elif response.status_code == 400:
            print("✅ API key is valid (received 400 which is expected for invalid request data)")
            return True
        else:
            print(f"✅ API key is valid (status code: {response.status_code})")
            return True
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error making request: {e}")
        return False

if __name__ == "__main__":
    print("=== Core API Access Key Verification ===\n")
    
    if verify_api_key():
        print("\n✅ API key verification successful!")
        sys.exit(0)
    else:
        print("\n❌ API key verification failed!")
        sys.exit(1)
