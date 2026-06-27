#!/bin/bash
set -euo pipefail

TO="hamalove@gmail.com"
FROM="hamalove@gmail.com"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

# Exit quietly if AWS credentials absent (prevents log spam on misconfigured pods)
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    echo "[notify] AWS credentials not set, skipping."
    exit 0
fi

HOSTNAME_VAL=$(hostname)
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable")
UPTIME_VAL=$(uptime -p 2>/dev/null || uptime)
CURRENT_TIME=$(date -u '+%Y-%m-%d %H:%M UTC')

SUBJECT="[RunPod] Pod still running: ${HOSTNAME_VAL} @ ${CURRENT_TIME}"
BODY="Your RunPod GPU instance is still running.

Host:    ${HOSTNAME_VAL}
Time:    ${CURRENT_TIME}
Uptime:  ${UPTIME_VAL}

GPU:
${GPU_INFO}

---
Sent every 30 minutes while the pod is active.
"

aws ses send-email \
    --region "${REGION}" \
    --from "${FROM}" \
    --to "${TO}" \
    --subject "${SUBJECT}" \
    --text "${BODY}"

echo "[notify] Email sent at ${CURRENT_TIME}"
