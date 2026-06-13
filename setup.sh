#!/bin/bash
set -e

echo "=== SWM RunPod Setup ==="

# Check if setup is completed
if [ -f /workspace/.setup_done ]; then
    echo "Already set up. Skipping."
    exit 0
fi

# swm-runpod repo clone
echo "[1/4] Cloning swm-runpod..."
cd /workspace
git clone https://github.com/Byungsooo/swm-runpod.git

# stable-worldmodel source code clone
echo "[2/4] Cloning stable-worldmodel..."
git clone https://github.com/galilai-group/stable-worldmodel.git

# Install in editable mode
echo "[3/4] Installing stable-worldmodel in editable mode..."
cd stable-worldmodel
pip install -e ".[all]" --quiet
cd /workspace

# S3 credentials setup
echo "[4/4] Configuring S3..."
mkdir -p /root/.aws
cat > /root/.aws/credentials << EOF
[default]
aws_access_key_id=${RUNPOD_S3_ACCESS_KEY}
aws_secret_access_key=${RUNPOD_S3_SECRET_KEY}
EOF

cat > /root/.aws/config << EOF
[default]
region=us-east-1
EOF

# Completion marking
touch /workspace/.setup_done

echo ""
echo "=== Setup Complete! ==="
echo "- stable-worldmodel: /workspace/stable-worldmodel"
echo "- swm-runpod:        /workspace/swm-runpod"
echo ""
echo "Run 'tmux new -s dev' to start a tmux session."
