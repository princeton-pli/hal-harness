#!/bin/bash
set -e  # Exit on any error

conda activate agent_env
git clone https://github.com/sierra-research/tau-bench && cd ./tau-bench && git checkout 14bf0ef52e595922d597a38f32d3e8c0dce3a8f8
pip install -e .
echo "TauBench setup complete"
