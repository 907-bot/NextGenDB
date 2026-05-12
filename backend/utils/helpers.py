import json
import os

def load_config(path: str):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def format_timestamp(ts: float):
    from datetime import datetime
    return datetime.fromtimestamp(ts).isoformat()
