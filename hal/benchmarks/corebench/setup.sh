#!/bin/bash
set -e

echo "Starting CORE-Bench-Hard minimal setup."
echo "---"

# Install R system dependencies needed for RMarkdown capsules
# Check if we're running as root or need sudo
if [ "$EUID" -eq 0 ]; then
    SUDO_CMD=""
    echo "[SETUP] Running as root, no sudo needed"
else
    SUDO_CMD="sudo"
    echo "[SETUP] Running as user, will use sudo"
fi

echo "[SETUP] Installing R system dependencies..."
export DEBIAN_FRONTEND=noninteractive
$SUDO_CMD apt-get update -y
$SUDO_CMD apt-get install -y --no-install-recommends \
    r-base r-base-dev \
    build-essential gfortran \
    pandoc \
    libssl-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    libcairo2-dev \
    libpng-dev \
    libjpeg-dev \
    libfreetype6-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libtiff5 \
    libxt6 \
    fontconfig \
    libudunits2-0 libudunits2-dev

echo "[SETUP] R base + pandoc + header libs (including udunits2) installed"

# Ensure conda-forge R + pandoc available inside agent_env
echo "Checking for minimal R + pandoc inside agent_env..."

echo "1. Checking interpreters:"
command -v Rscript >/dev/null && echo "   ✅ Rscript available" || echo "   ❌ Rscript missing"
command -v pandoc  >/dev/null && echo "   ✅ pandoc available"  || echo "   ❌ pandoc missing"

echo "---"
echo "2. Agent Responsibility:"
echo "   ❌ No Python ML libs preinstalled"
echo "   ❌ No R packages preinstalled"
echo "   The agent must install ALL project-level dependencies."

echo "CORE-Bench-Hard minimal setup complete."