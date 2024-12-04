#!/bin/bash
set -e  # Exit on any error

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "appworld-server"; then
    echo "Creating appworld-server environment..."
    conda create -y -n appworld-server python=3.11
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

# Check if appworld server is already running
if ! pgrep -f "appworld serve environment" > /dev/null; then
    echo "Starting appworld environment server..."
    # Start the server in background and save PID
    nohup appworld serve environment > appworld_server.log 2>&1 &
    echo $! > appworld_server.pid
    
    # Wait a bit for server to start
    sleep 5
    
    # Check if server started successfully
    if ! pgrep -f "appworld serve environment" > /dev/null; then
        echo "Failed to start appworld server. Check appworld_server.log for details"
        exit 1
    fi
    echo "Appworld server started successfully"
else
    echo "Appworld server is already running"
fi

conda activate agent_env
pip install "pydantic>=2.8.0"
pip install "sqlmodel>=0.0.15"
appworld install
echo "Appworld setup complete"
