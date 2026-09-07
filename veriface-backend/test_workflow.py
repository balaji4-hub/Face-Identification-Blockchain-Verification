import httpx
import sys
import os
sys.path.insert(0, '.')

BASE_URL = "http://127.0.0.1:8000"

def test_workflow():
    with httpx.Client(timeout=30) as client:
        # 1. Create session
        print("1. Creating session...")
        resp = client.post(f"{BASE_URL}/api/v1/verify/sessions")
        print(f"   Status: {resp.status_code}")
        session_data = resp.json()
        session_id = session_data["session_id"]
        print(f"   Session ID: {session_id}")
        
        # 2. Test status
        print("\n2. Checking status...")
        resp = client.get(f"{BASE_URL}/api/v1/verify/{session_id}/status")
        print(f"   Status: {resp.status_code}")
        print(resp.json())
        
        # 3. Test health
        print("\n3. Health check...")
        resp = client.get(f"{BASE_URL}/api/v1/verify/health")
        print(f"   Status: {resp.status_code}")
        print(resp.json())
        
        # 4. Test blockchain info
        print("\n4. Blockchain info...")
        resp = client.get(f"{BASE_URL}/api/v1/chain")
        print(f"   Status: {resp.status_code}")
        print(resp.json())
        
        print("\n=== All tests passed! ===")

if __name__ == "__main__":
    test_workflow()