# C-JEPA Implementation Project

Goal: reimplement C-JEPA (Causal-JEPA, ICML 2026, arXiv 2602.11389) inside `stable-worldmodel`, targeting the PushT control benchmark. Eventually contribute as a PR to `galilai-group/stable-worldmodel`.

Reference code: https://github.com/galilai-group/cjepa  
Paper: https://arxiv.org/abs/2602.11389  
Our fork: `Byungsooo/stable-worldmodel`

---

## Cost Estimates

| Phase | Wall time | GPU cost |
|-------|-----------|----------|
| VideoSAUR slot extraction (one-time) | ~30 min | ~$0.20 (cheap GPU) |
| C-JEPA predictor training (per run) | ~45–90 min | ~$0.50–1.50 (RTX 4090) |
| MPC evaluation (50 episodes) | ~15 min | ~$0.20 |
| **Per experiment total** | ~2 hr | **~$1–2** |

Hill-climbing to reproduce paper: 3–5 experiments → **~$5–10 total**.

Paper target (Table 3, PushT): **88.67% success rate** with |M|=1, 6×128 tokens.

---

## Architecture Summary

**VideoSAUR** (frozen): DINOv2 ViT-S/14 → Slot Attention (2 iter) → N=4 slots × 128-d  
**CJEPAPredictor**: 6-layer bidirectional Transformer (16 heads, dim_head=64, MLP=2048)  
**Masking**: object-level — randomly mask |M| ∈ {0,1,2} entire slot trajectories; anchor at t=t0  
**Masked token**: `z̃_τ^i = φ(z_{t0}^i) + e_τ` (linear proj of identity anchor + learnable temporal pos emb)  
**Loss**: `L_mask = L_history + L_future` (MSE on all masked tokens)  
**MPC**: forward-only inference + Hungarian slot matching + CEM optimizer

---

## Implementation Checklist

### Phase 1: Core model ✅ COMPLETE
- [x] `stable_worldmodel/wm/cjepa/__init__.py`
- [x] `stable_worldmodel/wm/cjepa/module.py`
  - [x] `BidirectionalTransformer` (reuses `lewm/module.py:Attention` with `causal=False`)
  - [x] `TemporalPosEmb` (`nn.Embedding(max_T, 128)`, the `e_τ`)
  - [x] `BidirectionalBlock` (wraps Attention + FeedForward)
- [x] `stable_worldmodel/wm/cjepa/cjepa.py`
  - [x] `encode()` — slot encoder + aux embeddings
  - [x] `_build_masked_tokens()` — object-level masking, identity anchor, temporal PE
  - [x] `forward_train()` — masking → predict → MSE loss on masked tokens
  - [x] `rollout()` — inference-only, future-only masking + sliding window
  - [x] `criterion()` / `get_cost()` — Hungarian matching + L2 cost (Costable protocol)
- [x] `DummySlotEncoder` placeholder (64×64 linear, for CPU testing without VideoSAUR)
- [x] Smoke test: `forward_train` loss=1.30, backward OK, `get_cost` shape (2,2) ✓

### Phase 2: Training pipeline ✅ COMPLETE
- [x] `scripts/train/cjepa.py` (mirror of `lewm.py`, `cjepa_forward` calls `forward_train`)
- [x] `scripts/train/config/cjepa.yaml`
  - [x] n_slots=4, slot_dim=128, history_len=3, future_len=1, max_masked=2
  - [x] predictor: depth=6, heads=16, dim_head=64, mlp_dim=2048
  - [x] DummySlotEncoder placeholder (VideoSAUR added in Phase 3)
  - [x] 30 epochs, Adam lr=5e-4, batch=256, bf16

### Phase 3: VideoSAUR integration ✅ COMPLETE
- [x] Find HuggingFace checkpoint for VideoSAUR — `HazelNam/CJEPA` hosts `pusht_videosaur_model.ckpt` (139MB, trained specifically on PushT, not an out-of-domain substitute), plus `pusht_videosaur_slots.pkl` (4.8GB pre-extracted reference slots)
- [x] Implement `VideoSAUREncoder` wrapper — `stable_worldmodel/wm/cjepa/videosaur_encoder.py`, returns (B, T, N, D) slots per clip (not per independent frame — see note below)
  - [x] Vendored minimal MIT-licensed modules from `martius-lab/videosaur` into `wm/cjepa/_videosaur/` (SlotAttention, MLP, temporal recurrence wrappers, RandomInit, slot-dynamics predictor) with attribution + LICENSE
  - [x] `wm/cjepa/download.py`: `download_videosaur_checkpoint()` / `download_videosaur_reference_slots()`
  - [x] Wired into `CJEPAWorldModel._extract_slots` via a `requires_temporal_context` flag (VideoSAUR's Slot Attention is recurrent frame-to-frame, unlike `DummySlotEncoder`)
  - [x] `cjepa.yaml`: `img_size` 224→196 (VideoSAUR's trained resolution), `slot_encoder` now targets `VideoSAUREncoder`
- [x] Smoke-test: `scripts/train/smoke_test_videosaur.py` — loads real checkpoint, runs on a real PushT clip (rendered live from `swm/PushT-v1`, since the full training dataset isn't collected yet — that's Phase 5), verifies shape `(1,4,4,128)`, finite values, and genuine frame-to-frame slot evolution. Also verified full `CJEPAWorldModel.forward_train`+`backward` and `.rollout`/`.get_cost` (MPC path) end-to-end with the real checkpoint.

**Key discovery**: the reference repo's checked-in `pusht_dinov2_hf.yml` config says `backbone.name: TimmExtractor`, but the actual released checkpoint's `state_dict` keys (`embeddings.cls_token`, `encoder.layer.0.attention.attention.key`, `layernorm.weight`) are HuggingFace `transformers.Dinov2Model` naming, not timm's. Confirmed by downloading the checkpoint and inspecting `state_dict.keys()` directly, then cross-checking the fork's actual `encoders.py` source (the "_hf" filename suffix means HF-backbone variant; `TimmExtractor` in the config is vestigial/unused — `FrameEncoder.build()` hardcodes `AutoModel.from_pretrained("facebook/dinov2-small")`). This meant **no new `timm` dependency was needed** — reused this repo's existing `create_backbone("dinov2_small")` (`wm/prejepa/module.py`), matching the pattern already used by `gcbc.py`/`hilp.py`/`gciql.py`/`gcivl.py`. `load_state_dict(strict=False)` loads with **zero missing keys** and exactly 53 unexpected keys, all `decoder.*` (the reconstruction head we don't need for inference) — strong confirmation the reconstructed architecture is exact.

**Environment fix (unrelated bug, affects the whole repo)**: `transformers.AutoModel.from_pretrained(...)` was broken Pod-wide by a stale `torchaudio==2.4.1+cu124` (leftover from the base Docker image) mismatched against `torch==2.12.1+cu130` — newer `transformers` transitively imports it for an ASR loss module, and the guarded `is_torchaudio_available()` check passed (package present) but the compiled `.so` failed to load. `torchaudio` isn't a dependency of this project at all; `pip uninstall torchaudio` fixed it. This was silently breaking `gcbc.py`/`hilp.py`/`gciql.py`/`gcivl.py`/`prejepa` too, not just this Phase 3 work — worth baking the uninstall (or an unpinned/compatible reinstall) into the Docker image so future Pods don't hit it.

**Deferred to Phase 5 (not blocking)**: numerical cross-check of `VideoSAUREncoder`'s output against `pusht_videosaur_slots.pkl` needs the exact same dataset clips used at extraction time, which requires collecting the PushT training dataset first (`pusht_expert_train.lance` isn't present on this Pod yet — that's already a Phase 5 checklist item). Also noted: `data/pusht.yaml` uses `frameskip=5` vs. VideoSAUR's training `frameskip=2` — a minor domain-shift risk, flagged in `cjepa.yaml`, not addressed here.

### Phase 4: Smoke test ✅ COMPLETE
- [x] CPU unit test: shapes, masking logic, loss is not NaN — `tests/wm/test_cjepa.py`
- [x] 1-epoch smoke run (both `DummySlotEncoder` and real `VideoSAUREncoder`, on a tiny disposable toy dataset — see notes below)
- [x] Verify loss decreases — 8-epoch run: `validate/pred_loss` 2.74 → 0.050 → ... → 0.017 (monotonic, converging)

### Phase 5: Full training + eval — 🟡 PARTIAL: pipeline runs end-to-end, task success far below target
- [x] Collect real PushT data — wrote `scripts/data/collect_pusht_expert_train.py` (1000 episodes, `WeakPolicy`), since no existing script produced the exact `pusht_expert_train.lance` name the training config expects
- [x] Run training on RTX 2000 Ada (not the originally-planned RTX 4090 — see below) — 30-epoch run (`cjepa_run1`, final val loss 0.004043, ~3h35m) and a later 5-epoch retrain (`cjepa_run2`, val loss 0.0027, ~33min) after finding a training-time bug (see below)
- [x] Run MPC eval: `python scripts/plan/eval_wm.py` — **runs end-to-end** (CEM solve + env rollout + video output), but success rate has not exceeded the 8% random-policy baseline on this same harness across 5 attempts (4% → 6% → 2% → 0% → 2%)
- [x] Stage-by-stage validation (2026-07-06): frozen VideoSAUR encoder ruled unlikely as the bottleneck (representation probe + stability checks); scaled-up action-sensitivity check confirmed a real but highly state-dependent cost signal (see below)
- [ ] **Next queued: characterize what separates reliable vs. unreliable per-episode cost signal** — see "Next investigation plan" below
- [ ] Compare to Table 3 baselines: OC-JEPA (76%), C-JEPA target (88.67%) — **not yet meaningfully comparable**; current numbers are statistical noise around the random baseline, not a real signal to compare against the paper

**Status as of 2026-07-04**: this is a partial success worth being precise about — the full pipeline (data collection → training → checkpoint → MPC rollout → video output) runs cleanly end-to-end with no crashes, and the debugging process along the way found and fixed 6 real, previously-latent bugs in `stable-worldmodel`'s eval path (written up in `TODO_UPSTREAM_FIXES.md` for upstream contribution — likely valuable on its own regardless of this project's outcome). But the actual planning task success rate has not shown a clear, reproducible improvement over random chance despite five rounds of bug-fixing (wrong policy class → wrong history-frame stride → non-deterministic slot encoder → wrong action/timestep alignment in training → zeroed history-action embeddings at inference). Direct empirical diagnostics (bypassing CEM, comparing `get_cost()`/`forward_train()` on real vs. corrupted actions) showed the later fixes produced a real, measurable, but small improvement in the model's action-sensitivity — not yet reflected in end-to-end MPC success.

**Decision (2026-07-04)**: rather than continue iterating on further suspected bugs blind, pausing here to validate intermediate pipeline stages individually (data, slot encoder outputs, trained representations) against known-good reference implementations (e.g. DINO-WM) and establish calibrated per-stage success criteria, rather than only checking the final MPC number. See `TODO_UPSTREAM_FIXES.md` for the full bug list with file:line references and reasoning.

**Update (2026-07-06) — stage-by-stage validation, per the 2026-07-04 decision**: this session ran on a fresh Pod (previous Pod's `pusht_expert_train.lance`, VideoSAUR checkpoint cache, and `cjepa_run1`/`cjepa_run2` checkpoints were all gone — none had been pushed to S3; the code fixes themselves were safe, already committed at `06c97cd`). Re-collected the dataset, re-downloaded the VideoSAUR checkpoint, and retrained a fresh `cjepa_run2` (5 epochs, same scope as before, final val loss 0.0026 — closely matching the prior session's 0.0027). Considered (and deprioritized) the idea of sanity-checking the MPC harness against a from-scratch-trained baseline (LeWM/DINO-WM/PLDM) run through `eval_wm.py` first — investigation showed all three baselines also require pre-stacked multi-frame `pixels` in their own `rollout()` (`prejepa.py:231`, `lewm.py:69`, same in `pldm.py`), but expose it as `history_size`/`predictor.num_frames`, never `history_len` — so `eval_wm.py`'s auto-selection (`getattr(model, 'history_len', None)`) would never route any of them to a history-aware policy either. This means that path isn't the already-solved, zero-friction reference it first looked like; it was set aside in favor of two direct, per-stage diagnostics instead (both added as reusable scripts under `scripts/diagnose/`):

- **`scripts/diagnose/slot_representation_probe.py`** — probes the frozen VideoSAUR encoder in isolation (no CJEPA predictor involved). Ridge-regression content probe: R²=0.964 vs. `block_pose`, R²=0.907 vs. `pos_agent` (n=980 frames, 50 episodes) — the frozen slots clearly encode the task-relevant geometry well. Hungarian-matched frame-to-frame cosine similarity: 0.993 at stride 1 (raw env step) → 0.989 at stride 2 (VideoSAUR's own training frameskip) → 0.978 at stride 5 (this project's actual data frameskip, the domain-shift risk flagged back in Phase 3 but never checked until now). The degradation from the flagged domain shift is real but small (~0.011 absolute) — **the frozen encoder is very unlikely to be the bottleneck.**
- **`scripts/diagnose/action_sensitivity_check.py`** — scales up last session's 5-episode ad hoc `get_cost()` real-vs-random-action percentile check to n=64 held-out episodes (K=32 random candidates each, all Gaussian since the data pipeline z-scores actions). Result: mean percentile 43.7, median 39.1 (below the 50 random-chance baseline, so there's a real average signal), but std=30.2 and only 54.7% of episodes have the real action beating the median random candidate — 10.9% of episodes the real action is worse than 90% of random candidates. This is a statistically-grounded version of last session's "11th-88th percentile spread" — confirms it's a real, non-noise signal on aggregate, but quantifies that it's currently far too inconsistent per-episode for CEM to reliably exploit.

**Conclusion**: combining both diagnostics points the remaining gap toward the predictor/training regime or the Hungarian-matched cost aggregation itself, not the frozen representation quality and not a residual wiring bug — a more specific narrowing-down than "still just statistical noise" from the 2026-07-04 entry. Also worth revisiting: the OOM/orphaned-forkserver issue from 2026-07-04 recurred on this fresh Pod on the very first training attempt (not a leftover-worker issue this time — a clean process hit the ~29GB cgroup cap with the shipped defaults `num_workers=6`/`persistent_workers=True`/`prefetch_factor=3`); reducing to `num_workers=2`/`prefetch_factor=2`/`persistent_workers=False` fixed it. Worth lowering `cjepa.yaml`'s defaults rather than relying on every session rediscovering this.

**Next investigation plan (queued, not yet started) — why is the per-episode signal so inconsistent?**

The action-sensitivity check's headline number (mean percentile 43.7) hides a std of 30.2 — roughly what you'd get from a uniform distribution over the full 0–100 range. 17.2% of episodes are excellent (real action beats ≥90% of random candidates), 10.9% are actively inverted (real action loses to ≥90% of randoms). This isn't "the model is weak on average" (which more training/data would fix); it's "the model's reliability depends on the state in a way we haven't characterized yet" — and CEM has no way to know in advance which regime a given planning call falls into, since it plans one state at a time and can't average across episodes the way our diagnostic did. Before touching architecture or training again, the next step is to find out *what* separates the reliable states from the unreliable ones.

**Environment setup on a fresh Pod** (checkpoint is on S3, dataset is not — deliberate, since recollection is cheap and storage isn't free):
1. `aws s3 cp --recursive s3://swm-research/checkpoints/cjepa_run2/ /root/.stable_worldmodel/checkpoints/cjepa_run2/` — skips the ~35min retrain entirely.
2. `python -c "from stable_worldmodel.wm.cjepa.download import download_videosaur_checkpoint as d; d()"` — cheap, ~30s.
3. `python scripts/data/collect_pusht_expert_train.py` — ~4 min for 1000 episodes (must be recollected each fresh Pod; not pushed to S3 to save storage).

**Plan**:
1. Extend `scripts/diagnose/action_sensitivity_check.py` (or copy to a new `scripts/diagnose/percentile_covariates.py` — same `get_cost()`-bypasses-CEM construction, just scaled up to ~150-200 episodes since it's cheap) to record, per episode, alongside the percentile score:
   - `block_motion` = `||block_pose[T_h] - block_pose[0]||` (raw units)
   - `agent_motion` = `||pos_agent[T_h] - pos_agent[0]||`
   - `contact_change` = delta in the dataset's `n_contacts` column across the window (**verify its exact semantics first** — inspect a few raw values, it hasn't been used yet in this project)
   - `real_action_magnitude` = `||real_future_action||` (z-scored units, already computed)
   - `history_slot_stability` = mean Hungarian-matched cosine similarity between consecutive history-frame slots *within that episode's own T_h window* (reuse `_hungarian_match_slots` from `wm/cjepa/cjepa.py`, same technique `slot_representation_probe.py` already uses, just scoped per-episode instead of aggregated across many)
2. Compute Spearman correlation between each covariate and the percentile score across all episodes (Spearman since the relationship may be monotonic-but-nonlinear). Also just compare the top-quartile (most reliable) vs. bottom-quartile (least reliable) episodes' covariate distributions directly — with ~150-200 samples this doesn't need to be fancy.
3. **Interpretation guide** (don't commit to a fix before running this — the right fix differs by outcome):
   - Percentile correlates with **low motion** (near-static windows are the unreliable ones) → supports a residual "copy-forward shortcut" hypothesis directly tied to the 2026-07-04 item-0c finding (the model nearly ignored actions before that fix; this would mean a weaker version of that shortcut still exists specifically when little is supposed to move, since copying the last frame forward already nearly minimizes MSE there). Fix direction: reweight training loss toward higher-motion transitions, or explicitly penalize the copy-forward solution.
   - Percentile correlates with **contact_change** → the hardest-to-predict moments are exactly the nonlinear contact transitions in the pushing physics. Fix direction: upweight contact transitions in training, or give the predictor more capacity/attention around them.
   - Percentile correlates with **history_slot_stability** → Hungarian re-matching itself is injecting noise into the cost. Fix direction: a soft-assignment cost instead of hard Hungarian matching, or anchor matching to the t0 identity throughout instead of re-matching pairwise at every step.
   - **No covariate correlates meaningfully** → the inconsistency doesn't trace to an obvious physical/representational factor. This would point toward either genuine stochastic variance in the learned dynamics (needing more training data/epochs/capacity) or estimator variance in a single predictor forward pass (worth trying ensembled/averaged predictions per candidate before concluding it's an architectural dead end).

### Phase 6: PR preparation
- [x] Add `tests/wm/test_cjepa.py` (shape checks, masking, loss) — done in Phase 4
- [x] `stable_worldmodel/wm/__init__.py` already exports `CJEPAWorldModel` — turned out to already be wired up (`from .cjepa import *`) when checked during Phase 4; no action needed
- [ ] Write PR description with results table

---

## Key Files in stable-worldmodel to Reference

| Reference | Path |
|-----------|------|
| Transformer blocks | `stable_worldmodel/wm/lewm/module.py` |
| Training script pattern | `scripts/train/lewm.py` |
| MPC eval (reuse as-is) | `scripts/plan/eval_wm.py` |
| PushT eval config template | `scripts/plan/config/pusht.yaml` |
| Costable protocol | `stable_worldmodel/protocols.py` |
| CEM solver | `stable_worldmodel/solver/cem.py` |

---

## PR Contribution Note

This is a solid contribution to `galilai-group/stable-worldmodel`:
- **Purely additive**: new `wm/cjepa/` module + training script, zero changes to existing code
- **Follows existing patterns**: identical interface to `lewm/`, `pldm/` — no new abstractions
- **Paper-grounded**: reproduces a published ICML 2026 result with matching hyperparameters
- **Target**: open PR after reproducing ≥85% success on PushT (Table 3 shows 88.67%)

---

## Session Log

### 2026-06-28 — Phase 1+2 complete, architecture doc added
- Read C-JEPA paper; assessed codebase; finalized plan; estimated cost ~$1–2/run
- Implemented `wm/cjepa/` module (module.py + cjepa.py): bidirectional transformer, object-level masking, identity anchor, temporal PE, Hungarian matching MPC
- Implemented training script (`scripts/train/cjepa.py`) + config (`cjepa.yaml`)
- All smoke tests pass: forward_train, backward, get_cost ✓
- Added `CJEPA_ARCHITECTURE.html` — architecture diagrams, loss/training explanation, references
- **Next**: Phase 3 — VideoSAUR integration (download checkpoint, implement `VideoSAUREncoder`)

### 2026-07-01 — Phase 3 complete: VideoSAUR integration
- Found and downloaded the official PushT VideoSAUR checkpoint (`HazelNam/CJEPA` on HF) — confirmed live via direct HTTP checks before trusting it, not just an agent's summary (which initially and incorrectly claimed no PushT-specific checkpoint existed)
- Vendored a minimal MIT-licensed subset of `martius-lab/videosaur` into `wm/cjepa/_videosaur/` (SlotAttention, MLP projection, temporal recurrence wrappers, RandomInit, slot-dynamics predictor) with attribution + LICENSE
- Built `VideoSAUREncoder` (`wm/cjepa/videosaur_encoder.py`); discovered mid-implementation that the checkpoint actually uses a HuggingFace DINOv2 backbone (not timm as the reference YAML suggested) by inspecting `state_dict` keys directly — reused this repo's existing `create_backbone` utility instead of adding `timm`. `load_state_dict(strict=False)` loads with zero missing keys
- Wired into `CJEPAWorldModel._extract_slots` via a `requires_temporal_context` flag; updated `cjepa.yaml` (`img_size` 224→196, `slot_encoder` → `VideoSAUREncoder`) and `pyproject.toml` (`+scipy`)
- Fixed an unrelated Pod-wide bug blocking all `transformers.AutoModel.from_pretrained` calls (stale `torchaudio` leftover from the base Docker image, mismatched against the installed torch build) — was silently breaking `gcbc.py`/`hilp.py`/`gciql.py`/`gcivl.py`/`prejepa` too; worth baking the fix into the Dockerfile
- All smoke tests pass: shape/NaN/temporal-evolution checks on real live-rendered PushT frames (`scripts/train/smoke_test_videosaur.py`), plus full `CJEPAWorldModel.forward_train`+`backward` and `.rollout`/`.get_cost` (MPC path) end-to-end with the real checkpoint
- Deferred to Phase 5: numerical cross-check against the authors' pre-extracted `pusht_videosaur_slots.pkl` (needs the same dataset clips, which requires collecting `pusht_expert_train.lance` first — already a Phase 5 item); the `frameskip=5` vs. VideoSAUR's training `frameskip=2` domain-shift risk (flagged, not fixed)
- **Next**: Phase 4 — CPU unit tests (`tests/wm/test_cjepa.py`, mirroring `test_lewm.py`), 1-epoch smoke run, verify loss decreases

### 2026-07-03 — Phase 4 complete: smoke test, found and fixed 3 integration bugs

- Added `tests/wm/test_cjepa.py`: `get_cost` shape-contract tests (mirroring `test_lewm.py`/`test_pldm.py`'s bare-model + monkey-patched-rollout style), plus new real `forward_train`+backward tests and `_build_masked_tokens` invariant checks (future always masked, t0 anchor never masked, history mask count respects `max_masked_slots`) — no prior wm test exercised a real forward+loss pass. All 6 pass; full `pytest tests/wm/` (48 tests) also clean.
- Correction to the Phase 3 log above: `smoke_test_videosaur.py` only exercises `VideoSAUREncoder` in isolation — it never actually calls `forward_train`/`rollout`/`get_cost` on `CJEPAWorldModel`, despite the prior entry's claim. Worth remembering: verify what a smoke test *actually* covers by reading it, not by trusting a summary (including my own from a prior session).
- **Environment blocker, unrelated to CJEPA code**: `torch.cuda.is_available()` was `False` on this fresh Pod — `nvidia-smi` reports driver 570.195.03 (CUDA 12.8 max), but `Dockerfile`'s unpinned `pip install 'stable-worldmodel[all]'` resolved PyPI's latest torch (2.12.1+cu130, CUDA 13.0), which the driver can't run. This blocks *all* GPU training (`trainer.accelerator: gpu` in every `scripts/train/*.yaml`), not just VideoSAUR. Same root pattern as Phase 3's `torchaudio` fix — baked into the image, so it recurs on every fresh Pod since the Dockerfile was never patched. Fixed locally this session (`pip uninstall torchaudio` + reinstall `torch==2.12.1+cu126`/`torchvision==0.27.1+cu126` — same versions, just the CUDA-12 build) and verified via full `pytest tests/wm/` + `scripts/train/smoke_test_videosaur.py` both passing after the swap. **Not persisted** — documented in `TODO_FIX_DRIVER_TORCH_MISMATCH.md` (repo root) with the proposed Dockerfile patch; needs a deliberate image rebuild, didn't want to make that call unilaterally.
- Rather than pull Phase 5's real `pusht_expert_train.lance` collection forward (that dataset doesn't exist on this Pod and collecting it is explicitly a Phase 5 task), added `scripts/data/collect_pusht_smoke.py` — collects a small disposable 50-episode `pusht_smoke.lance` (same schema as the real dataset, ~10s to collect) purely for smoke-test fixture data.
- Ran the 1-epoch smoke train against `pusht_smoke.lance`, first with `DummySlotEncoder` (CPU-cheap), then with the real `VideoSAUREncoder` + real checkpoint on the RTX 2000 Ada. Both runs surfaced (and fixed) real bugs no prior smoke test had caught:
  1. `DummySlotEncoder` didn't accept the `checkpoint_path` kwarg the yaml's `model.slot_encoder` node always sets, and Hydra's `~key` CLI delete-override silently fails on `null`-valued keys (a known OmegaConf quirk — a null value is indistinguishable from "missing" to the delete check) — so the yaml's own documented smoke-test override never actually worked. Fixed by adding an accepted-and-ignored `checkpoint_path=None` param to `DummySlotEncoder`.
  2. `CJEPAWorldModel.encode()` read `info['state']` for the proprio encoder, but every other baseline in this repo (`gcbc`/`gcivl`/`gciql`/`hilp`) — and the training script's own `proprio_encoder.input_dim` sizing — uses the `'proprio'` column. `'state'` and `'proprio'` are different columns with different dims (7 vs. 4). Fixed `cjepa.py:162-163` to read `info['proprio']`.
  3. `scripts/train/cjepa.py` sized `proprio_encoder.input_dim` as `frameskip * dataset.get_dim('proprio')`, copy-pasted from the `action_encoder` line above it — but only `action` gets stacked across the frameskip window by the dataset (`data/dataset.py:70-83`); `proprio` stays at its raw per-frame dim. Fixed to drop the `frameskip *` multiplier for proprio.
  4. Under `trainer.precision: bf16`, `_build_masked_tokens`'s boolean-mask assignment (`tokens[visible_mask] = ...`) requires an exact dtype match (`index_put_` semantics, stricter than plain indexed assignment), but `temporal_emb`'s output dtype didn't reliably match `slots_all`'s. Fixed by casting `t_embs` to `tokens.dtype` right after computing it.
- After those fixes, both the `DummySlotEncoder` and real-`VideoSAUREncoder` 1-epoch runs complete cleanly (`fit/pred_loss` 0.149 and 0.195 respectively; VideoSAUR run confirmed "all tracked parameters received gradients on the first backward pass").
- Verified loss actually decreases (not just "doesn't crash"): an 8-epoch run on the same toy dataset shows `validate/pred_loss` falling monotonically: 2.74 (pre-train) → 0.050 → 0.032 → 0.026 → 0.023 → 0.020 → 0.018 → 0.017 → 0.017 (converging).
- **Next**: Phase 5 — collect the real `pusht_expert_train` PushT dataset, run the full 30-epoch training on an RTX 4090, MPC eval, compare to Table 3 baselines. Before that: consider whether to rebuild the Docker image with the `TODO_FIX_DRIVER_TORCH_MISMATCH.md` fix so future Pods don't hit the GPU blocker again.

### 2026-07-04 — Phase 5: pipeline runs end-to-end, but MPC task success stuck at random-chance level after 5 bug-fix rounds

- Decided to train on the currently-available RTX 2000 Ada rather than switch to an RTX 4090 pod (cost/convenience tradeoff); wrote `scripts/data/collect_pusht_expert_train.py` (1000 episodes via `WeakPolicy`) since no existing collection script produced the exact `pusht_expert_train.lance` dataset name the training config expects.
- First training attempt OOM-killed at 0 steps. Root-caused to the container's cgroup memory cap (~28.9GiB, not host RAM) combined with orphaned `multiprocessing.forkserver` dataloader workers leaked by an earlier crashed run (`persistent_workers=True` doesn't get torn down cleanly on SIGKILL) — cleared with `pkill`, reran with reduced `num_workers`/`prefetch_factor`, worked cleanly from then on.
- Full 30-epoch run (`cjepa_run1`) completed in ~3h35m, final val loss 0.004043 — a healthy, converged number.
- First MPC eval (`eval_wm.py`) crashed immediately: found and fixed a pre-existing bug where the episode-index/step-index column lookup breaks for Lance-format datasets specifically (`column_names` deliberately excludes the reserved index columns; the code's fallback heuristic never actually resolves to the right name for Lance) — 4 occurrences fixed in `eval_wm.py`. Also fixed `get_row_data()` being called for index columns it doesn't expose (switched to `get_col_data()`), and a stale `.h5` default in `scripts/plan/config/pusht.yaml`'s `eval.dataset_name`.
- Eval then ran to completion but scored **4% success (2/50)** — below even a random policy's **8% (4/50)** on the identical harness. Root-caused to `eval_wm.py` using the generic `WorldModelPolicy` (1 history frame) instead of the repo's own (previously unused) `CJEPAHistoryPolicy` (`history_len=3`), silently misaligning the model's fixed-size temporal position embedding table. Fixed, wired in.
- Re-eval: 6% — no real improvement (within noise of the 8% baseline). Found a second bug in the same policy: history frames were being sampled every raw env-step instead of every `frameskip=5` steps, making the 3 "history" frames near-duplicates instead of the well-separated frames seen in training. Fixed by gating the frame-buffer append to fire once per `frameskip` calls.
- Re-eval: 2% — still no improvement. A deeper empirical diagnostic (loading the checkpoint + real data directly, bypassing CEM) found: (a) `get_cost()` scored the real ground-truth trajectory *worse* than the median of 64 random candidates — zero discriminative signal; (b) `forward_train`'s loss was statistically identical whether given the real, zeroed, shuffled, or scaled-random action — reproduced on real training data, meaning this was a property the model *learned*, not an inference-side bug. Traced to a genuine training-time bug: `_build_masked_tokens` conditioned the future/target token on the wrong action block (the dataset's block `T_h` — the action taken *after* the target frame — instead of block `T_h - 1`, the action that actually produced it), for the entire 30-epoch run. Also found (in passing, unrelated) the frozen VideoSAUR slot encoder's `RandomInit` draws unseeded noise every call, making two encodings of the *identical* frame diverge by ~58% of total signal energy even after best-case Hungarian realignment — fixed by seeding it deterministically.
- Fixed the action/timestep bug, retrained 5 epochs (`cjepa_run2`, val loss 0.0027 — faster convergence, though this metric was never the bottleneck). Direct diagnostics confirmed a real, consistent (if modest) improvement: `forward_train` now ranks real actions below all corruptions, and `get_cost()`'s real-vs-random percentile improved from "always ~50th-or-worse" to a 11th-88th spread across 5 test episodes.
- Full eval after this fix: **0%**. Traced to a second, related bug: `_single_step_predict` still zeroed out the *earlier* history positions' action embeddings at inference (only the future position's indexing was fixed), while training always used real ones there. Implemented real action-history tracking in `CJEPAHistoryPolicy` (buffers and frameskip-stacks actual executed actions, normalizes them the same way training does — pre-stacking, matching the dataset pipeline's actual order of operations) and threaded it through `rollout()`, sliding it in lockstep with the slot history across the autoregressive planning loop. Pure inference-side fix — no retrain needed.
- Re-eval (same `cjepa_run2` checkpoint, no retrain): **2%**. Still statistically indistinguishable from the random baseline after 5 full bug-fix-and-reevaluate cycles.
- **6 real bugs found and fixed this session, written up for upstream contribution in `TODO_UPSTREAM_FIXES.md`** (episode-index column resolution, `get_row_data` index-column gap, stale eval dataset default, wrong policy class, wrong history-frame stride, non-deterministic slot encoder, action/timestep misalignment in training, zeroed history-action embeddings at inference) — independently valuable regardless of this project's outcome.
- **Decision**: stop iterating on further suspected bugs blind. Task success has not moved off random-chance level despite 5 real, well-evidenced fixes, which suggests either a remaining issue not yet found, or a more fundamental representation/training problem not reducible to a simple wiring bug. Next session: validate intermediate pipeline stages individually (slot encoder output quality, learned representation structure, dataset statistics) against a known-good reference (e.g. DINO-WM) rather than only checking the final end-to-end MPC number — establish per-stage success criteria before further debugging the full pipeline blind.
- **Next**: stage-by-stage validation (data → slot encoding → learned dynamics → planning) with calibrated success criteria per stage, informed by how DINO-WM/other baselines in this repo validate each of those stages.
