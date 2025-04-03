FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies that are commonly used
RUN pip install --no-cache-dir \
    weave==0.51.32 \
    numpy \
    pandas \
    requests \
    rich \
    tqdm \
    pydantic

# Set working directory
WORKDIR /workspace

# We don't use a non-root user because we need to write files
# that can be read by the host system after the container exits

# Default entry point
CMD ["bash"] 