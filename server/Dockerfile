# Meta PyTorch OpenEnv Hackathon 2026
# Optimized Dockerfile for Safe Station Environment

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the environment source code
# Assuming build context is the repository root
COPY . .

# Install dependencies
# We use the populated requirements.txt inside the package
RUN pip install --no-cache-dir -r safe_station/requirements.txt

# Set PYTHONPATH so that 'import safe_station' works correctly
ENV PYTHONPATH="/app"

# Expose the FastAPI port
EXPOSE 8000

# Health check to ensure the server is responsive
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the environment server
# Using the module path relative to PYTHONPATH=/app
CMD ["uvicorn", "safe_station.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
