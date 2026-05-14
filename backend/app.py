print("DEBUG: backend/app.py is being loaded...")
try:
    from .main import app
    print("DEBUG: app imported successfully from .main")
except Exception as e:
    try:
        from main import app
        print("DEBUG: app imported successfully from main")
    except Exception as e2:
        print(f"DEBUG: Failed to import app: {e} | {e2}")
        import traceback
        traceback.print_exc()
        raise
