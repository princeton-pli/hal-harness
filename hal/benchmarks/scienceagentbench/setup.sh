#!/bin/bash

conda create -y -n sci-agent-eval python=3.10 pip setuptools wheel
eval "$(conda shell.bash hook)"
conda activate sci-agent-eval
pip install pip-tools
conda deactivate

mkdir /workspace/pred_programs

echo "ScienceAgentBench setup complete"
