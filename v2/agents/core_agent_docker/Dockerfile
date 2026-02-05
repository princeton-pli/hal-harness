# Note: building on Silicon Macs should be done enforcing AMD64.
# Do this with:
#    `docker build --platform linux/amd64 -t hal-agent-runner:latest .`

FROM python:3.12-slim

# Detect if building natively on ARM64 and exit with helpful message
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ] && [ "$(uname -m)" = "aarch64" ]; then \
        echo ""; \
        echo "========================================"; \
        echo "ERROR: Building natively on ARM64/Apple Silicon"; \
        echo "========================================"; \
        echo ""; \
        echo "This image must be built for AMD64 to match Azure VMs."; \
        echo ""; \
        echo "Please rebuild with:"; \
        echo "  docker build --platform linux/amd64 -t hal-agent-runner:latest ."; \
        echo ""; \
        echo "This will use Rosetta 2 emulation on your Mac and ensure"; \
        echo "compatibility with Azure x86_64 VMs."; \
        echo "========================================"; \
        echo ""; \
        exit 1; \
    fi

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Anaconda (full version) with architecture detection
ARG TARGETARCH
RUN wget --quiet "https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh" -O /tmp/anaconda.sh && \
    /bin/bash /tmp/anaconda.sh -b -p /opt/conda && \
    rm /tmp/anaconda.sh && \
    /opt/conda/bin/conda clean -afy && \
    echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh

ENV PATH=/opt/conda/bin:$PATH

# Create conda environment
RUN conda create -y -n agent_env python=3.12

# Set working directory
WORKDIR /workspace

# Copy requirements.txt first for better layer caching
COPY requirements.txt /workspace/requirements.txt

# Install agent requirements (includes wandb, weave, and gql)
RUN conda run -n agent_env pip install -r /workspace/requirements.txt

# Copy in the core_agent code
COPY core_agent /workspace

# Copy entrypoint script
COPY entrypoint.py /workspace/entrypoint.py
RUN chmod +x /workspace/entrypoint.py

# Environment variables (these will be overridden at runtime)
ENV HAL_RUN_ID=""
ENV HAL_TASK_ID=""
ENV HAL_AGENT_MODULE=""
ENV HAL_AGENT_FUNCTION=""

# We don't use a non-root user because we need to write files
# that can be read by the host system after the container exits

# Default entry point
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "agent_env", "python", "/workspace/entrypoint.py"] 
