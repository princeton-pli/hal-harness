#!/bin/bash
set -e  # Exit on error

# Configuration
USERNAME="$1"  # Pass username as first argument
HOME_DIR="/home/$USERNAME"

echo "Starting setup for user: $USERNAME"

# Install system dependencies
# echo "Installing system dependencies..."
# apt-get update -y
# apt-get install -y curl wget build-essential
# echo "System dependencies installed"

# Install Miniconda
echo "Installing Miniconda..."
MINICONDA_PATH="/home/$USERNAME/miniconda3"
curl -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p $MINICONDA_PATH
rm /tmp/miniconda.sh
echo "Miniconda installed"

# Set ownership
echo "Setting Miniconda ownership..."
chown -R $USERNAME:$USERNAME $MINICONDA_PATH

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
chown $USERNAME:$USERNAME "$HOME_DIR/init_conda.sh"

# Create and activate environment as the agent user with explicit output
echo "Creating conda environment..."
su - $USERNAME -c "bash -c '\
    echo \"Initializing conda...\" && \
    source $HOME_DIR/init_conda.sh && \
    echo \"Creating agent_env...\" && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    if [ -f requirements.txt ]; then \
        PYTHON_VERSION=$(grep "^python==" requirements.txt | cut -d"=" -f3) && \
        if [ ! -z "$PYTHON_VERSION" ]; then \
            conda create -n agent_env python=$PYTHON_VERSION -y; \
        else \
            conda create -n agent_env python=3.12 -y; \
        fi \
    else \
        conda create -n agent_env python=3.12 -y; \
    fi && \
    echo \"Activating agent_env...\" && \
    conda activate agent_env && \
    echo \"Checking for requirements.txt...\" && \
    if [ -f requirements.txt ]; then \
        echo \"Installing requirements...\" && \
        pip install -r requirements.txt && \
        echo \"Installing weave and gql pin...\" && \
        pip install weave==0.51.41 \"gql<4\" && \
        echo \"Requirements installed\"; \
    else \
        echo \"No requirements.txt found\" && \
        exit 1; \
    fi'"

echo "Setup completed successfully"
