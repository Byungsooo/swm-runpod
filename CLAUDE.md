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
2. Clones the **forked** `stable-worldmodel` (`Byungsooo/stable-worldmodel`) into `/workspace`, adds `galilai-group/stable-worldmodel` as the `upstream` remote, and installs in editable mode (`pip install -e ".[all]"`)
3. Configures AWS credentials from env vars
4. Marks setup complete so re-running is a no-op

If `swm-setup` isn't available (older image), run manually:
```bash
cd /workspace && curl -s https://raw.githubusercontent.com/Byungsooo/swm-runpod/main/setup.sh | bash
```

## Working with stable-worldmodel

- Installed in **editable mode** at `/workspace/stable-worldmodel` — code edits take effect immediately, no reinstall needed.
- Fork: `Byungsooo/stable-worldmodel` (origin). Upstream: `galilai-group/stable-worldmodel`. To sync upstream changes: `git fetch upstream && git merge upstream/main`.
- Claude has full freedom to modify, refactor, and experiment with the swm codebase. No need to ask before making changes — this is a research sandbox, not production code. Prefer creating new branches/files for experimental work when changes are large, but small iterative edits in place are fine.
- swm provides the data pipeline, baseline models (DINO-WM, LeWM, TD-MPC2, PLDM), and evaluation protocols. The typical workflow is: pull/generate data via swm → train a model (custom or baseline) → evaluate using swm's protocols.
- Reference training scripts live in `scripts/train/` (e.g. `lewm.py`, `prejepa.py` for DINO-WM reproduction).

## MuJoCo / Rendering

- `MUJOCO_GL=egl` is set globally in the Docker image — MuJoCo-based envs (e.g. `swm/OGBScene-v0`, `swm/OGBCube-v0`, Fetch robotics) render headlessly without any extra setup.
- Do **not** set `MUJOCO_GL` manually in notebooks; it is already inherited from the environment.
- The EGL fix required force-installing `libegl1_1.4.0-1` (Ubuntu jammy package) because the base image ships a stub `libEGL.so` without the full GLVND dispatch. This is baked into the Dockerfile.

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

## Lessons Learned: Running WM Experiments

Accumulated from a multi-session round of LeWM/DINO-WM/CJEPA data-quality and eval-harness debugging (see `CJEPA_PROJECT.md`'s session log for the full narratives). Organized by how likely each is to recur.

### Tier 1 — general engineering/methodology habits (apply to any WM work, not swm-specific)

- **Verify data quality directly; don't trust dataset names or docstrings.** `pusht_expert_train.lance` sounds like expert demonstrations. Querying its `terminated` column directly showed 0% real task success — it was collected with `WeakPolicy`, a non-goal-directed random-near-block policy, not an expert. Always check the actual ground-truth signal (success flags, distance-to-goal) in a dataset before trusting what it's named or how it's described.
- **Measure real throughput with a small smoke test before committing to a long run — don't linearly extrapolate from a different scale or batch size.** An early guess of ~15-25hr for a training run (extrapolated from a `batch_size=32` run) turned out to be ~11-12hr once actually measured at the intended `batch_size=128` — batch-size speedups aren't linear. A 200-step timed dry run is cheap insurance against a wildly wrong multi-hour estimate.
- **Expect real run-to-run noise in CEM/MPC eval success rates, even with a fixed seed, especially at small episode counts (n=50).** Rerunning the *identical* checkpoint and config produced 8% vs. 10%. Don't over-read a few-percentage-point difference as a real effect without repeating it; CJEPA's own project log independently flagged the same "candidate sampling isn't seeded independently of episode selection" phenomenon.
- **When something behaves strangely, seriously consider the bug is in the shared/upstream `stable-worldmodel` library, not just local script logic.** This project has repeatedly found and fixed real upstream bugs this way rather than working around symptoms (e.g. a numpy truth-value bug in `is_image_column`, a dataset-driven-eval `options` gap, a `Costable` dispatch issue, a `PreJEPA.encode()` missing-key crash) — see `SWM_CONTRIBUTION_IDEAS.md` for the running list, several of which are candidates for upstream PRs.

### Tier 2 — mechanisms specific to `swm`'s `World`/`eval_wm.py`/PushT pipeline, but guaranteed to recur in *any* experiment using this same harness (CJEPA included, since it shares the identical code path)

- **Environment variation-space settings used at data-collection time do not automatically carry over to eval.** `World.collect(..., options={'variation_values': {...}})` applies a custom override (e.g. block shape) only during collection. `World.evaluate(dataset=..., ...)` (dataset-driven mode) previously didn't even accept an `options` kwarg — it silently reset every episode to the *default* variation-space values, regardless of what was used to generate the training data. Concretely: training a model entirely on a square-block variant and then evaluating it against the harness's default T-shaped block gave a strict 0% success — not because the model failed, but because eval was silently testing an entirely different physical shape than it was trained on. Fixed in `World._evaluate_from_dataset()`/`World.evaluate()` to thread `options` through (on the `merge-upstream-planning-refactor` branch — not yet merged to `main`/pushed to `origin` as of 2026-07-13, see `CJEPA_PROJECT.md`). **Any future experiment that customizes env variation at collection time must explicitly pass the matching `eval.options` override at eval time too — they are not linked.**
- **Match eval-time planning horizon (`plan_config.horizon`) to the model's training-time prediction horizon (`wm.num_preds`), and check literature values before overriding either — don't assume.** A LeWM checkpoint trained with `num_preds=1` (single-step prediction) evaluated at `horizon=10` scored 2.0%; the *actual* literature-correct horizon (per the LeWM paper's own Appendix D, arXiv 2603.19312) is 5, which — importantly — already matched `swm`'s own shipped default (`scripts/plan/config/pusht.yaml`). The mismatch came from *our own* override, following a generic README example, not a bug in the library. **Before touching `horizon`, check what the shipped default already is, and what the relevant paper actually used — don't guess a "fixed" value from a diagnostic sweep.** (CJEPA has its own `wm.num_preds`/planning-horizon knobs in `cjepa.yaml` — same risk, different specific numbers; look up CJEPA's own reference config rather than reusing LeWM's `horizon=5`.)
- **The rendered "goal" overlay (green shape) in PushT eval videos is not necessarily the actual eval target — verify against the code before trusting a visualization.** In dataset-driven eval mode, the green overlay is drawn from a separate, independently-configured `env.goal_pose` variation value that the eval callables (`_set_state`/`_set_goal_state`) never touch — it stays at a fixed default every episode, unrelated to the real per-episode `goal_state` used for the actual success computation. The real target is shown by the **block's position** in the "goal" panel (rendered by temporarily teleporting the physics body to `goal_state` and capturing a frame), not by visual alignment with the green shape. This is upstream's own existing behavior (confirmed via `git log` that neither `env.py` nor the relevant `pusht.yaml` callables have ever been touched by a fork-only commit), not something introduced downstream — worth keeping in mind for any PushT rollout video inspection.
- **Verify that archived/multi-run outputs actually correspond to the run they claim to.** `eval_wm.py`'s video output directory depends on the exact structure of the `policy=` string (a bare run name vs. `<run_name>/weights_epoch_N.pt` resolve to different paths) — a naive glob-based archiving script silently collected identical, stale files across several different runs before this was caught (via checksum comparison). When scripting a multi-checkpoint comparison, checksum or spot-check the archived artifacts rather than assume the copy step worked.

## Repo Structure (swm-runpod)

```
swm-runpod/
├── Dockerfile                    # custom dev image definition
├── setup.sh                      # first-time Pod initialization script
├── CLAUDE.md                     # this file
├── quickstart.ipynb              # PushT random-policy demo
├── quickstart_pusht.ipynb        # PushT extended demo
├── quickstart_ogbscene.ipynb     # OGBScene (robotic arm) random-policy demo
└── .github/workflows/
    └── docker-build.yml          # auto-builds & pushes b8k3/swm-dev:latest on push to main
```

Changes to `Dockerfile` or the workflow file trigger an automatic rebuild via GitHub Actions and push to Docker Hub (`b8k3/swm-dev:latest`). After a rebuild, existing Pods need to be terminated and redeployed to pick up the new image — they don't auto-update.