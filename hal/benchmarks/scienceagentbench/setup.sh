#!/bin/bash

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -y -n sci-agent-eval python=3.10 pip setuptools wheel
eval "$(conda shell.bash hook)"
conda activate sci-agent-eval
pip install pip-tools
conda deactivate

echo "ScienceAgentBench setup complete"
