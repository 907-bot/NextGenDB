import sys
import os
from pathlib import Path

print("DEBUG: backend/app.py is being loaded...")

# Add the parent of the backend directory to sys.path
# This allows us to import 'backend' as a package even if we are running from inside it.
backend_dir = Path(__file__).resolve().parent
repo_root = backend_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    # We import 'backend.main' instead of just 'main' to preserve package context
    # This fixed the "ImportError: attempted relative import with no known parent package"
    from backend.main import app
    print("DEBUG: app imported successfully from backend.main")
except Exception as e:
    print(f"DEBUG: Failed to import app from backend.main: {e}")
    import traceback
    traceback.print_exc()
    raise
