# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FastAPI application for the Safe Station Environment.
"""

import sys
import os

# Robust path injection: Ensure the repository root is in sys.path
# This allows 'import models' and 'import safe_station_environment' to work 
# regardless of where the server is started from.
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    # Error will be raised during app creation if not installed
    create_app = None

try:
    from models import SafeStationAction, SafeStationObservation
    from safe_station_environment import SafeStationEnvironment
except ImportError:
    # Fallback for local development environments
    try:
        from .models import SafeStationAction, SafeStationObservation
        from .safe_station_environment import SafeStationEnvironment
    except ImportError:
        # Final fallback: manually import from root if everything else fails
        import models
        import safe_station_environment
        SafeStationAction = models.SafeStationAction
        SafeStationObservation = models.SafeStationObservation
        SafeStationEnvironment = safe_station_environment.SafeStationEnvironment


# Create the app with web interface and README integration
app = create_app(
    SafeStationEnvironment,
    SafeStationAction,
    SafeStationObservation,
    env_name="safe_station",
    max_concurrent_envs=1,
) if create_app else None

@app.get("/")
async def root():
    """Welcome page to satisfy Hugging Face health checks."""
    return {
        "status": "online",
        "environment": "Safe Station RL v1.0",
        "message": "Visit /docs for the interactive API explorer."
    }

@app.get("/health")
async def health():
    """Dedicated health check endpoint for Docker."""
    return {"status": "ok"}


def main():
    """Entry point for direct execution."""
    import uvicorn
    if app:
        uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
