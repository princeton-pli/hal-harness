#!/bin/bash
set -e  # Exit on error

# Redirect all output to log file
exec > /home/agent/setup_vm.log 2>&1

# Configuration
HOME_DIR="/home/agent"

echo "Starting setup for user: agent"

# Grant passwordless sudo to the agent user
echo "agent ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/agent
chmod 440 /etc/sudoers.d/agent

# Create /workspace owned by agent user
mkdir -p /workspace
chown agent:agent /workspace

# Install Miniconda
echo "Installing Miniconda..."
MINICONDA_PATH="/home/agent/miniconda3"
curl -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p $MINICONDA_PATH
rm /tmp/miniconda.sh
echo "Miniconda installed"

# Set ownership
echo "Setting Miniconda ownership..."
chown -R agent:agent $MINICONDA_PATH

# Create conda initialization script
echo "Creating conda initialization script..."
cat > "$HOME_DIR/init_conda.sh" << EOF
#!/bin/bash
export PATH="$MINICONDA_PATH/bin:\$PATH"
eval "\$($MINICONDA_PATH/bin/conda shell.bash hook)"
conda init
if [ -f "\$HOME/.env" ]; then
    set -a
    source "\$HOME/.env"
    set +a
fi
EOF

# Make initialization script executable and set ownership
chmod +x "$HOME_DIR/init_conda.sh"
chown agent:agent "$HOME_DIR/init_conda.sh"


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
        echo \"Installing weave, wandb, gql pin, and Azure VM dependencies...\" && \
        pip install weave==0.51.41 wandb==0.23.0 \"gql<4\" && \
        pip install \"azure-identity>=1.12.0\" \"requests>=2.31.0\" && \
        echo \"Requirements installed\"; \
    else \
        echo \"No requirements.txt found\" && \
        exit 1; \
    fi'"

echo "Setup completed successfully"
