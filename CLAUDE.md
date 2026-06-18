# CLAUDE.md

This file gives Claude Code context for working in this environment. The goal is research-oriented, hobby-scale open source contribution to `stable-worldmodel` (swm), with paper-writing as an eventual outcome.

## Environment Overview

- **Platform**: RunPod Pod, deployed from custom Docker image `b8k3/swm-dev:latest`
- **Base image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Image repo**: https://github.com/Byungsooo/swm-runpod (public)
- **GPU strategy**: cheap GPU (RTX 2000 Ada / RTX 4000 Ada / similar, ~$0.25/hr) for development and debugging; RTX 4090 (~$0.69/hr) only for actual training runs. Always check GPU availability per region before deploying — availability varies a lot and some regions (e.g. US-CA-2) have very limited stock of affordable GPUs.
- **No Network Volume** — decided against it because Network Volumes are pinned to a single datacenter region, and GPU availability in that region can dry up entirely. Storage is handled via S3 instead (see below).
- **Container disk**: ~20GB is enough since large data isn't stored locally long-term.

## Storage: AWS S3

- **Bucket**: `s3://swm-research` (region: us-east-1)
- **Purpose**: code backups, model checkpoints, experiment logs/results. NOT for bulk datasets.
- **Dataset strategy**: use existing public datasets (HuggingFace Hub, etc.) and pull them directly into the Pod. Do not route bulk dataset transfer through S3 — egress cost ($0.09/GB) makes that expensive. S3 storage itself is cheap (~$0.023/GB/month); egress is the cost driver to avoid.
- **Credentials**: `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` are injected as RunPod Pod template environment variables (not hardcoded anywhere, not passed manually per-session). `setup.sh` picks them up automatically and writes `/root/.aws/credentials`.

## First-time Pod Setup

After deploying a new Pod from the `swm-dev` template:

```bash
swm-setup
```

This alias (defined in `.bashrc` inside the Docker image) runs `setup.sh` from this repo, which:
1. Clones this repo (`swm-runpod`) into `/workspace`
2. Clones `stable-worldmodel` into `/workspace` and installs it in editable mode (`pip install -e ".[all]"`)
3. Configures AWS credentials from env vars
4. Marks setup complete so re-running is a no-op

If `swm-setup` isn't available (older image), run manually:
```bash
cd /workspace && curl -s https://raw.githubusercontent.com/Byungsooo/swm-runpod/main/setup.sh | bash
```

## Working with stable-worldmodel

- Installed in **editable mode** at `/workspace/stable-worldmodel` — code edits take effect immediately, no reinstall needed.
- Claude has full freedom to modify, refactor, and experiment with the swm codebase. No need to ask before making changes — this is a research sandbox, not production code. Prefer creating new branches/files for experimental work when changes are large, but small iterative edits in place are fine.
- swm provides the data pipeline, baseline models (DINO-WM, LeWM, TD-MPC2, PLDM), and evaluation protocols. The typical workflow is: pull/generate data via swm → train a model (custom or baseline) → evaluate using swm's protocols.
- Reference training scripts live in `scripts/train/` (e.g. `lewm.py`, `prejepa.py` for DINO-WM reproduction).

## Session Persistence

- **Always use tmux** for any long-running process (training, data generation). SSH/VS Code Remote connections can drop; tmux sessions survive that as long as the underlying Pod is still running.
  ```bash
  tmux new -s train   # start a named session
  tmux attach -t train # reattach after reconnecting
  ```
- tmux config (mouse scroll, larger history) is already baked into the Docker image.
- Remember: Pod stop/terminate wipes anything not in S3. Push checkpoints and important results to `s3://swm-research` before stopping a Pod for an extended period.

## Cost Awareness

- This is a hobby-budget project. Be mindful of GPU hours — stop/terminate Pods when not actively working.
- Rough monthly budget target: ~$25–50/mo combining cheap-GPU development time and occasional 4090 training runs.
- Don't default to the most expensive available GPU; check the cheap tier first unless a training run specifically needs more VRAM/throughput.

## Repo Structure (swm-runpod)

```
swm-runpod/
├── Dockerfile              # custom dev image definition
├── setup.sh                 # first-time Pod initialization script
├── CLAUDE.md                 # this file
└── .github/workflows/
    └── docker-build.yml     # auto-builds & pushes b8k3/swm-dev:latest on push to main
```

Changes to `Dockerfile` or the workflow file trigger an automatic rebuild via GitHub Actions and push to Docker Hub (`b8k3/swm-dev:latest`). After a rebuild, existing Pods need to be terminated and redeployed to pick up the new image — they don't auto-update.