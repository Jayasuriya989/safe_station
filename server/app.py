import sys
import os

# ABSOLUTE TOP PRIORITY: Path Injection
# Ensure the root directory and server directory are in sys.path.
# We resolve the absolute real path to ensure the container can find graders.
try:
    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.abspath(os.path.join(current_file_dir, ".."))
    
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    if current_file_dir not in sys.path:
        sys.path.insert(0, current_file_dir)
except Exception as e:
    sys.stderr.write(f"[WARNING] Path injection failure: {e}\n")

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FastAPI application for the Safe Station Environment.
"""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    create_app = None

# Now import root modules - path injection ensures these are found
try:
    from models import SafeStationAction, SafeStationObservation
    from safe_station_environment import SafeStationEnvironment
except ImportError:
    # Final manual fallback if the first path injection is bypassed by the loader
    sys.path.append(os.getcwd())
    import models
    import safe_station_environment
    SafeStationAction = models.SafeStationAction
    SafeStationObservation = models.SafeStationObservation
    SafeStationEnvironment = safe_station_environment.SafeStationEnvironment

app = create_app(
    SafeStationEnvironment,
    SafeStationAction,
    SafeStationObservation,
    env_name="safe_station",
    max_concurrent_envs=1,
) if create_app else None

@app.get("/")
async def root():
    return {
        "status": "online",
        "environment": "Safe Station RL v1.0",
        "message": "Visit /docs for the interactive API explorer."
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

def main():
    import uvicorn
    if app:
        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
