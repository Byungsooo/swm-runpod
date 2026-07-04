# TODO: torch/CUDA version mismatch breaks GPU on every fresh Pod

## Symptom

`torch.cuda.is_available()` returns `False` even though `nvidia-smi` shows a
healthy GPU. Warning on any CUDA call:

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too
old (found version 12080). Please update your GPU driver...
```

This isn't specific to any one script — it blocks `lightning`'s
`accelerator: gpu` trainer construction entirely, so it breaks all GPU
training (`scripts/train/*.py`), not just CUDA-heavy code paths.

## Root cause

`Dockerfile:42` — `RUN pip install 'stable-worldmodel[all]'` — installs
`torch`/`torchvision` unpinned from PyPI's default index. Recent PyPI torch
wheels (2.12.1 as of this writing) default to CUDA 13.0 (`+cu130`, pulling
`nvidia-*-cu13` deps), which requires a driver newer than what this Pod's
underlying node has (`570.195.03`, CUDA 12.8 max). This is baked into the
image at *build* time — `setup.sh`'s `pip install -e ".[all]"` re-resolves
the same unpinned `torch`/`torchvision` from `pyproject.toml` and doesn't
change anything.

Because the bug lives in the Dockerfile, **every Pod deployed from
`b8k3/swm-dev:latest` hits this**, not just this one.

Related, same root pattern: `torchaudio` (a leftover from the base
`runpod/pytorch:2.4.0-...` image, not an actual project dependency) is *also*
baked into the image and version-mismatched against whatever torch ends up
installed. This was already identified and worked around once, in Phase 3
(see `CJEPA_PROJECT.md`'s 2026-07-01 log — `pip uninstall torchaudio` fixed
`transformers.AutoModel.from_pretrained`) — but since the fix was never
added to the Dockerfile, it silently came back on this Pod.

## What was done this session (live container only, NOT persisted)

```bash
pip uninstall -y torchaudio
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu126 \
  'torch==2.12.1' 'torchvision==0.27.1'
```

This swaps only the CUDA build tag (`+cu130` → `+cu126`) at the same torch
version, so no other package's version constraints changed — verified via a
full `pytest tests/wm/` pass (48/48) and the existing
`scripts/train/smoke_test_videosaur.py` GPU smoke test both passing after
the swap. `nvidia-cudnn-cu13`, `cuda-toolkit==13.0.2`, etc. were replaced
with their `cu12` equivalents automatically by pip's dependency resolution.

Known side effect: `pillow` was pulled to `12.2.0`, which conflicts with
`moviepy`'s `pillow<12.0` pin (pip printed a warning, did not block install).
Not hit by anything in this session; worth a look if `moviepy`-based code
starts failing.

This fix is **local to this container** — it will NOT survive a Pod restart
from the current image, and won't apply to any other Pod deployed from
`b8k3/swm-dev:latest` until the Dockerfile is fixed.

## Proposed permanent fix (not applied — needs a deliberate image rebuild)

In `Dockerfile`, pin the CUDA-12.x build explicitly and drop the unused
`torchaudio`, before the `stable-worldmodel[all]` install pulls in an
unpinned/CUDA-13 torch:

```dockerfile
# Pin torch/torchvision to a CUDA 12.x build compatible with the driver
# available on RunPod's Ada-generation nodes (avoid PyPI's default,
# which resolves newer CUDA-13 wheels the driver can't run).
RUN pip install --index-url https://download.pytorch.org/whl/cu126 \
    'torch==2.12.1' 'torchvision==0.27.1'

RUN pip install 'stable-worldmodel[all]'

# torchaudio comes in as a leftover/transitive dep and isn't used by this
# project; it also ends up version-mismatched against the pinned torch above.
RUN pip uninstall -y torchaudio
```

Alternatively (more future-proof, more work): pin `torch`/`torchvision` in
`stable-worldmodel`'s own `pyproject.toml` to a CUDA-12.x-compatible range so
`pip install -e ".[all]"` in `setup.sh` also resolves correctly without
relying on Dockerfile ordering — but that's a change to the upstream fork,
not just this infra repo, so it's out of scope for a quick Dockerfile patch.

Whichever fix lands, re-run `scripts/train/smoke_test_videosaur.py` (Phase 3's
GPU smoke test) and `pytest tests/wm/` against a freshly rebuilt image before
trusting it, since RunPod's available driver versions can vary by node/region.

## Status

**Applied (2026-07-04).** `Dockerfile` now installs `torch==2.12.1`/
`torchvision==0.27.1` from the `cu126` wheel index before
`stable-worldmodel[all]`, and uninstalls the leftover `torchaudio` — see the
fix described above. Pushed to `main`, which triggers the GitHub Actions
rebuild of `b8k3/swm-dev:latest`.

Still needs the smoke test / `pytest tests/wm/` re-run described above
against a freshly rebuilt image on an actual Pod (this session's sandbox has
no `docker` CLI to verify a build directly, and the node it was authored on
already had a newer, CUDA-13-capable driver so it wouldn't reproduce the
original failure anyway).
