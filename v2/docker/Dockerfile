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

# Install Miniconda with architecture detection
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    fi && \
    wget --quiet "$MINICONDA_URL" -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy && \
    echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh

ENV PATH=/opt/conda/bin:$PATH

# Create conda environment
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y -n agent_env python=3.12

# Install weave and pin gql < 4 in agent_env
RUN conda run -n agent_env pip install weave==0.51.41 "gql<4"

# Set working directory
WORKDIR /workspace

# Copy entrypoint script
COPY entrypoint.py /usr/local/bin/entrypoint.py
RUN chmod +x /usr/local/bin/entrypoint.py

# Environment variables (these will be overridden at runtime)
ENV HAL_RUN_ID=""
ENV HAL_TASK_ID=""
ENV HAL_AGENT_MODULE=""
ENV HAL_AGENT_FUNCTION=""

# We don't use a non-root user because we need to write files
# that can be read by the host system after the container exits

# Default entry point
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "agent_env", "python", "/usr/local/bin/entrypoint.py"] 
