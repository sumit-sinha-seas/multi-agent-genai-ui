# agents/logger.py

import json
from datetime import datetime

LOG_FILE = "logs/interaction_log.jsonl"

def log_interaction(data: dict):
    data["timestamp"] = datetime.now().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")
