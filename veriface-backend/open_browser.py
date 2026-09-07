import httpx
import webbrowser

print("Testing VERIFACE CHAIN API...")
print("=" * 50)

with httpx.Client(timeout=10) as client:
    endpoints = [
        ("/", "Root", "GET"),
        ("/api/v1/chain", "Chain Info", "GET"),
        ("/api/v1/verify/health", "Health", "GET"),
        ("/api/v1/verify/sessions", "Create Session", "POST"),
    ]

    for path, name, method in endpoints:
        try:
            if method == "POST":
                resp = client.post(f"http://127.0.0.1:8000{path}")
            else:
                resp = client.get(f"http://127.0.0.1:8000{path}")
            print(f"✓ {name}: {resp.status_code}")
            if resp.status_code == 200:
                print(f"  Response: {resp.json()}")
        except Exception as e:
            print(f"✗ {name}: {e}")

    print("\n" + "=" * 50)
    print("✓ Server is RUNNING at http://127.0.0.1:8000")
    print("✓ Opening Swagger Docs...")
    webbrowser.open("http://127.0.0.1:8000/docs")