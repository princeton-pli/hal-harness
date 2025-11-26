#!/bin/bash
set -e

echo "Starting CORE-Bench-Hard minimal setup."
echo "---"

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