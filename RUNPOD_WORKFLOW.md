# RunPod Pod Workflow: Resume vs. Restart

This guide covers the two ways to come back to development after stepping away from a Pod.

---

## 1. Pause → Resume (Stop / Start)

Use this when you want to come back soon (same day, next day) and don't mind paying a small storage holding cost in the meantime. The container disk is preserved — nothing is lost.

### Steps

1. **Start the Pod**
   RunPod console → Pods → select your stopped Pod → **Start**

2. **Get the new connection info**
   IP and port can change on restart. Check the **Connect** tab for the updated SSH command:
   ```
   ssh root@<NEW_IP> -p <NEW_PORT> -i ~/.ssh/id_ed25519
   ```

3. **Update `~/.ssh/config`**
   ```
   Host runpod-dev
       HostName <NEW_IP>
       User root
       Port <NEW_PORT>
       IdentityFile ~/.ssh/id_ed25519
   ```

4. **Reconnect via VS Code Remote-SSH**
   Bottom-left `><` icon → Connect to Host → `runpod-dev`

5. **Reattach to your tmux session**
   ```bash
   tmux attach -t train
   ```
   (or whatever session name you used)

### What's preserved
- `/workspace` contents (swm-runpod, stable-worldmodel, any data/checkpoints you put there)
- Installed packages, editable install of swm
- Running tmux sessions (processes inside them may have been killed when the Pod stopped, but the session itself reattaches)

### What you do NOT need to do
- No need to re-run `swm-setup` — `/workspace/.setup_done` marker is still there
- No need to re-clone anything
- No need to re-pull from S3 unless you specifically want fresher data

### Cost note
Stopped Pods still incur a small container disk holding cost. If you won't be back for more than a few days, terminating instead is usually cheaper — see below.

---

## 2. Terminate → New Pod

Use this when you're done with a Pod for an extended period, switching GPU types, or RunPod terminated it for you (e.g. preemption). The container disk is wiped — you're starting fresh.

### Steps

1. **Deploy a new Pod**
   RunPod console → Deploy → select the `swm-dev` template → choose GPU → Deploy

2. **Get connection info**
   Same as above — check Connect tab for IP/port.

3. **Update `~/.ssh/config`**
   Same as above.

4. **Connect via VS Code Remote-SSH**

5. **Run first-time setup**
   ```bash
   cd /workspace
   swm-setup
   ```
   This clones `swm-runpod` and `stable-worldmodel`, installs swm in editable mode, and configures AWS credentials (already injected via Pod template env vars).

6. **Restore anything you need from S3**
   ```bash
   aws s3 sync s3://swm-research/checkpoints /workspace/checkpoints
   ```
   Only needed if you had in-progress work backed up there. Don't sync large datasets through S3 — pull those from their original public source (e.g. HuggingFace) instead.

7. **Start a new tmux session**
   ```bash
   tmux new -s train
   ```

### What's lost
- Everything in `/workspace` from the previous Pod (code changes not pushed to git, local files not backed up to S3, in-progress checkpoints not synced)

### Before terminating an old Pod, make sure to:
- `git push` any code changes
- `aws s3 sync` any checkpoints/results you want to keep:
  ```bash
  aws s3 sync /workspace/checkpoints s3://swm-research/checkpoints
  ```

---

## Quick Comparison

| | Pause → Resume | Terminate → New Pod |
|---|---|---|
| `/workspace` contents | Preserved | Wiped |
| Run `swm-setup` again | No | Yes |
| Restore from S3 | No (unless desired) | Yes, if you had in-progress work |
| SSH config update | Yes, if IP/port changed | Yes, always |
| Holding cost while idle | Small (container disk) | None |
| Best for | Short breaks (hours–days) | Long breaks, GPU type switch, or after backing everything up |