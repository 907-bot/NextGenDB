print("DEBUG: app.py is being loaded...")
try:
    from backend.main import app
    print("DEBUG: app imported successfully from backend.main")
except Exception as e:
    print(f"DEBUG: Failed to import app from backend.main: {e}")
    import traceback
    traceback.print_exc()
    raise

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
