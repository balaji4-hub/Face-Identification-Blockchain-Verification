import uvicorn
import sys
sys.path.insert(0, '.')

uvicorn.run(
    "veriface.main:app",
    host="127.0.0.1",
    port=8000,
    reload=False,
    log_level="info"
)