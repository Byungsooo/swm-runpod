#!/bin/bash
# setup.sh - First-time initialization script for RunPod instance
# Run once after launching a new Pod:
#   export AWS_ACCESS_KEY_ID="..."
#   export AWS_SECRET_ACCESS_KEY="..."
#   curl -s https://raw.githubusercontent.com/Byungsooo/swm-runpod/main/setup.sh | bash
set -e

echo "=== SWM RunPod Setup ==="

# Skip if already initialized
if [ -f /workspace/.setup_done ]; then
    echo "Already set up. Skipping."
    exit 0
fi

# Clone this repo
echo "[1/4] Cloning swm-runpod..."
cd /workspace
git clone https://github.com/Byungsooo/swm-runpod.git

# Clone stable-worldmodel from your fork, with upstream pointing to the original repo
echo "[2/4] Cloning stable-worldmodel (your fork)..."
git clone https://github.com/Byungsooo/stable-worldmodel.git
cd stable-worldmodel
git remote add upstream https://github.com/galilai-group/stable-worldmodel.git
cd /workspace

# Install in editable mode so code changes are reflected immediately
echo "[3/4] Installing stable-worldmodel in editable mode..."
cd stable-worldmodel
pip install -e ".[all]" --quiet
cd /workspace

# Configure GitHub credentials for HTTPS push (requires GITHUB_TOKEN env var)
if [ -n "$GITHUB_TOKEN" ]; then
    git config --global credential.helper store
    echo "https://Byungsooo:${GITHUB_TOKEN}@github.com" > /root/.git-credentials
    chmod 600 /root/.git-credentials
    echo "GitHub credentials configured."
else
    echo "WARNING: GITHUB_TOKEN not set. Git push over HTTPS will require manual auth."
fi

# Configure AWS S3 credentials (requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars)
echo "[4/4] Configuring AWS S3..."
mkdir -p /root/.aws

if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "WARNING: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set."
    echo "S3 credentials not configured. Export them before running this script if S3 access is needed."
else
    cat > /root/.aws/credentials << CREDS
[default]
aws_access_key_id=${AWS_ACCESS_KEY_ID}
aws_secret_access_key=${AWS_SECRET_ACCESS_KEY}
CREDS

    cat > /root/.aws/config << CONFIG
[default]
region=us-east-1
CONFIG
    echo "AWS credentials configured."
fi

# Mark setup as complete
touch /workspace/.setup_done

echo ""
echo "=== Setup Complete ==="
echo "  stable-worldmodel : /workspace/stable-worldmodel"
echo "  swm-runpod        : /workspace/swm-runpod"
echo "  S3 bucket         : s3://swm-research"
echo ""
echo "Tip: run 'tmux new -s dev' to start a tmux session."
