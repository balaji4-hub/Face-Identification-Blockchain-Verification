import sys
sys.path.insert(0, '.')
from veriface.main import app
from veriface.api.routers.verify import router

print("=== Router included routes ===")
print(f"Router prefix: {router.prefix}")
print(f"Router tags: {router.tags}")
print(f"Number of routes: {len(router.routes)}")

print("\n=== App Routes ===")
for route in app.routes:
    if hasattr(route, 'methods'):
        print(f"{list(route.methods)} {route.path}")
    elif hasattr(route, 'router'):
        print(f"INCLUDED ROUTER with {len(route.routes)} routes")