import httpx
import sys
sys.path.insert(0, '.')

try:
    # Test health endpoint
    resp = httpx.get("http://127.0.0.1:8000/api/v1/verify/health", timeout=5)
    print(f"Health: {resp.status_code}")
    print(resp.json())
    
    # Test root endpoint
    resp = httpx.get("http://127.0.0.1:8000/", timeout=5)
    print(f"\nRoot: {resp.status_code}")
    print(resp.json())
    
except Exception as e:
    print(f"Error: {e}")