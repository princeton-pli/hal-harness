#!/bin/bash
set -e  # Exit on any error

apt-get update && apt-get install -y procps

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "appworld-server"; then
    echo "Creating appworld-server environment..."
    conda create -y -n appworld-server python=3.12
fi

# Activate the server environment
eval "$(conda shell.bash hook)"
conda activate appworld-server

# Install appworld in server environment
echo "Installing appworld in server environment..."
pip install appworld

# Install and download data
echo "Setting up appworld data..."
appworld install
appworld download data

# Start appworld server in the background
echo "Starting appworld environment server..."
appworld serve environment --port 8001 &

# Add a short pause to ensure the server has time to start
sleep 20

# Run additional commands
conda activate agent_env
appworld install
pip install "pydantic>=2.8.0"
pip install "sqlmodel>=0.0.15"

echo "Appworld setup complete"
