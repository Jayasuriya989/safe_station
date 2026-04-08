# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Safe Station Environment.

This module creates an HTTP server that exposes the SafeStationEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import SafeStationAction, SafeStationObservation
    from .safe_station_environment import SafeStationEnvironment
except ImportError:
    from models import SafeStationAction, SafeStationObservation
    from server.safe_station_environment import SafeStationEnvironment


# Create the app with web interface and README integration
app = create_app(
    SafeStationEnvironment,
    SafeStationAction,
    SafeStationObservation,
    env_name="safe_station",
    max_concurrent_envs=1,
)

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


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
