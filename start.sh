#!/bin/bash
set -e

# SSH key injection
mkdir -p /root/.ssh && chmod 700 /root/.ssh
echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

# Fix: sshd does not forward Docker env vars to SSH sessions.
# Write to profile.d so all login shells (SSH, tmux, etc.) inherit them.
# This also fixes swm-setup silently skipping AWS credential config.
cat > /etc/profile.d/pod-env.sh <<ENV
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export GITHUB_TOKEN="${GITHUB_TOKEN:-}"
ENV
chmod 600 /etc/profile.d/pod-env.sh

# Write git credentials so HTTPS push works in SSH sessions without VS Code helper
if [ -n "${GITHUB_TOKEN:-}" ]; then
    git config --global credential.helper store
    printf "https://Byungsooo:%s@github.com\n" "${GITHUB_TOKEN}" > /root/.git-credentials
    chmod 600 /root/.git-credentials
fi

# Write credentials file for awscli (used by cron and any shell)
if [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    mkdir -p /root/.aws
    printf "[default]\naws_access_key_id=%s\naws_secret_access_key=%s\n" \
        "${AWS_ACCESS_KEY_ID}" "${AWS_SECRET_ACCESS_KEY}" > /root/.aws/credentials
    printf "[default]\nregion=%s\n" "${AWS_DEFAULT_REGION:-us-east-1}" > /root/.aws/config
fi

# Write env file cron can source (cron does NOT source /etc/profile.d/)
cat > /etc/pod_env <<ENV
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
ENV
chmod 600 /etc/pod_env

# Install crontab: every 30 minutes, source env then run notify.sh
(crontab -l 2>/dev/null; echo "*/30 * * * * . /etc/pod_env && /usr/local/bin/notify.sh >> /var/log/notify.log 2>&1") | crontab -

# Start cron daemon in background, then exec sshd as foreground process
service cron start
exec /usr/sbin/sshd -D
