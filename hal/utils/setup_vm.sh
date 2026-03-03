#!/bin/bash
set -e  # Exit on error

# Redirect all output to log file
exec > /home/agent/setup_vm.log 2>&1

# Configuration
HOME_DIR="/home/agent"

echo "Starting setup for user: agent"

# Create and activate environment as the agent user with explicit output
echo "Creating conda environment..."
# FIXME: stop installing pinned dependencies like this
su - agent -c "bash -c '\
    source /home/agent/miniconda3/etc/profile.d/conda.sh && \
    echo \"Installing Python standard library modules...\" && \
    conda activate agent_env
    pip install --upgrade pip && \
    echo \"Checking for requirements.txt...\" && \
    if [ -f requirements.txt ]; then \
        echo \"Installing requirements...\" && \
        pip install -r requirements.txt && \
        echo \"Installing weave and gql pin...\" && \
        pip install weave==0.51.41 \"gql<4\" && \
        echo \"Installing Azure logging dependencies...\" && \
        pip install \"azure-monitor-ingestion>=1.0.3\" \"azure-identity>=1.12.0\" \"requests>=2.31.0\" && \
        echo \"Requirements installed\"; \
    else \
        echo \"No requirements.txt found\" && \
        exit 1; \
    fi'"

echo "Setup completed successfully"
