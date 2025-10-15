#!/bin/bash
set -e  # Exit on any error

apt-get update && apt-get install -y procps

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "appworld"; then
    echo "Creating appworld environment..."
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    conda create -y -n appworld python=3.12
fi

# Activate the appworld environment
eval "$(conda shell.bash hook)"
conda activate appworld

# Install appworld
echo "Installing appworld ..."
pip install appworld

# Install and download data
echo "Setting up appworld data..."
appworld install
appworld download data

echo "Appworld setup complete"
