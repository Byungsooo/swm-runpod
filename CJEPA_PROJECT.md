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
- [x] Characterize what separates reliable vs. unreliable per-episode cost signal — `scripts/diagnose/percentile_covariates.py` (n=200 episodes): only `real_action_magnitude` correlates significantly (Spearman r=0.323, p<0.0001); none of `block_motion`/`agent_motion`/`contact_change`/`history_slot_stability` reach significance (p=0.43/0.26/0.52/0.68). See the 2026-07-06 Session Log entry for full numbers and interpretation.
- [x] Tried ensembling (per the interpretation guide) — **ruled out**: `get_cost()` has zero per-call randomness to ensemble away (confirmed empirically), and varying the VideoSAUR encoder's slot-space init seed across an ensemble (`scripts/diagnose/ensembled_action_sensitivity_check.py`) added noise rather than reducing it (std 29.8 → 30.6, no mean improvement). See 2026-07-06 Session Log for detail.
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

**Results (2026-07-06, same session)**: ran `scripts/diagnose/percentile_covariates.py`
(n=200 held-out episodes, K=32 random candidates each, same checkpoint/seed
conventions as `action_sensitivity_check.py`). None of the four originally-hypothesized
covariates reached significance: `block_motion` r=-0.056 (p=0.43), `agent_motion`
r=-0.080 (p=0.26), `contact_change` r=-0.046 (p=0.52), `history_slot_stability`
r=0.029 (p=0.68) — the reliable- and unreliable-quartile means for
`history_slot_stability` were identical to 4 decimal places (0.9250 vs. 0.9250),
further confirming the frozen encoder isn't where the inconsistency lives. The one
covariate that *did* reach significance was **not** one of the four interpretation-guide
branches: `real_action_magnitude` correlated positively with percentile
(r=0.323, p<0.0001 — reliable-quartile mean 3.03 vs. unreliable-quartile mean 3.47),
meaning larger real actions (in z-scored units) tend to have *worse* relative cost
ranking against random candidates. Notably, `real_action_magnitude` and `block_motion`
are only weakly related to each other in this data — a large commanded action doesn't
reliably produce large observed block displacement in a contact-rich pushing task — so
this isn't just `block_motion` in disguise.

By the letter of the interpretation guide above, this result falls under "no covariate
correlates meaningfully" for the four originally-hypothesized physical/representational
factors, and the guide's suggested next step for that branch — try ensembled/averaged
predictions per candidate rather than a single forward pass — also fits a plausible
mechanism for the `real_action_magnitude` finding: larger, rarer actions are less
densely represented in training, so the predictor's single-pass cost estimate is
plausibly noisier for them, and percentile (a rank statistic) is exactly the kind of
measure that noise pushes toward the middle or worse regardless of the true average
cost. Also worth a flag: this n=200 run's own baseline numbers (mean percentile 51.8,
median 53.1, std 30.3) are notably closer to pure random chance than the earlier n=64
check's (mean 43.7, median 39.1) — worth keeping in mind that the "real signal" framing
from the smaller sample may have been partly sample-noise; a larger n is more
trustworthy here (200 vs. 64), but this discrepancy itself hasn't been root-caused.

**Decision**: stopping at diagnosis for this pass, as planned — the next actual step
(try ensembled/averaged predictor forward passes, since it's the one direction both the
"no covariate" branch and the `real_action_magnitude` finding point toward) is a
follow-up implementation task, not yet started. Raw per-episode results saved to
`scripts/diagnose/percentile_covariates_results.csv` for any further analysis without
re-running the GPU pass.

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

## Upstream Bug List (candidates for PR)

Bugs and gaps found in `stable-worldmodel` (not our CJEPA code) while running Phase 5.
Several are already patched locally in `/workspace/stable-worldmodel` and just need to
be pushed upstream as their own small PR(s), separate from the CJEPA contribution
itself. Consolidated here (previously tracked in a separate `TODO_UPSTREAM_FIXES.md`,
now folded into this doc so there's a single source of truth).

### 0. `eval_wm.py`: uses the wrong policy class for history-conditioned world models, silently corrupting MPC eval (most severe finding this project)

**Files**: `scripts/plan/eval_wm.py` (policy instantiation, was ~line 116),
`scripts/plan/cjepa_policy.py` (`CJEPAHistoryPolicy`, pre-existing but never wired
into `eval_wm.py`)

First full 30-epoch CJEPA training run converged cleanly (final validation loss
0.004043 — the masked-slot prediction task learned fine), but the first MPC eval
against it scored **4% success (2/50)** — *worse* than a random policy on the exact
same harness (**8%, 4/50**, confirmed via a same-config `policy=random` run). A model
this much worse than random is a strong signal of a wiring bug, not an undertrained
model, since even mediocre learned dynamics should not systematically underperform
uninformed random actions.

Root cause: `eval_wm.py` instantiated the generic `swm.policy.WorldModelPolicy`,
which passes whatever `pixels` shape it receives straight through to the solver — in
practice a single current frame per env, since `World` only ever supplies one frame
per step (`world/world.py:524`). `CJEPAWorldModel` is trained with `history_len=3`
baked into a *fixed-size* `TemporalPosEmb` lookup table (`wm/cjepa/cjepa.py:110-120`)
whose row semantics (0=anchor, 1-2=history offsets, 3=future) were learned
specifically for a 3-frame history. Feeding it 1 frame instead of 3 doesn't crash —
`_single_step_predict` just derives `T_total` from whatever it's given
(`cjepa.py:308-309`) and ends up indexing the *wrong* row of the embedding table as
the "future" query, silently answering a different (near-static reconstruction)
task instead of genuine forward dynamics. Every `predicted_slots` the CEM cost is
computed from is then wrong in the same biased direction for every candidate action
— exactly the kind of bug that can make a planner confidently steer *worse* than
random, rather than merely noisier.

The fix already existed in the repo, just unused: `scripts/plan/cjepa_policy.py`'s
`CJEPAHistoryPolicy` (its own docstring describes this exact failure mode almost
verbatim) maintains a per-env ring buffer of the last `history_len` frames and
stacks them before delegating to the base class. `scripts/plan/demo_cjepa_rollout.py`
already wraps its policy in `CJEPAHistoryPolicy` for this reason — `eval_wm.py` was
simply never updated to do the same.

**Local fix**: import `CJEPAHistoryPolicy` and duck-type on the model exposing
`history_len` (only `CJEPAWorldModel` does, confirmed via
`grep -rn "self\.history_len\s*=" stable_worldmodel/wm/`, so this doesn't affect
`eval_wm.py`'s use with other world models):
```python
policy_cls = (
    CJEPAHistoryPolicy
    if getattr(model, 'history_len', None) is not None
    else swm.policy.WorldModelPolicy
)
policy = policy_cls(solver=solver, config=config, process=process, transform=transform)
```
**Suggested upstream fix**: same patch, or better, make `WorldModelPolicy` itself
history-aware (check the model for a `history_len` attribute and stack frames
internally) so callers don't need a separate subclass + manual selection logic at
every call site that might evaluate a history-conditioned model.

**Update**: this fix alone only moved success from 4% to 6% (still within noise of
the 8% random baseline) — see item 0b below for the second, more severe part of the
same underlying issue.

---

### 0b. `cjepa_policy.py`'s `CJEPAHistoryPolicy` samples history frames at the wrong stride (every raw step, not every `frameskip` steps)

**File**: `scripts/plan/cjepa_policy.py`

Fixing item 0 (1 frame → 3 frames) only moved success from 4% to 6%, statistically
indistinguishable from the 8% random baseline — so the frame *count* wasn't the
whole story. The frame *spacing* was still wrong: training samples history frames
`frameskip=5` raw env-steps apart (`scripts/train/config/data/pusht.yaml`, enforced
by `LanceDataset._process_batch` striding every column by `self.frameskip`,
`stable_worldmodel/data/formats/lance.py:454`). But `CJEPAHistoryPolicy.get_action`
appended a new frame to its history deque on **every** call
(`cjepa_policy.py`, old lines 40-47), and `get_action` is called once per **raw**
env step (`World._run_iter`, `world/world.py:404-410` — there's no
frameskip/action-repeat wrapper anywhere in the env stack; `action_block` is purely
a macro-action-unpacking construct inside `WorldModelPolicy.get_action`'s internal
buffer, it doesn't change how often `get_action` itself is invoked).

Net effect: at eval time the model's 3 "history" frames were 3 consecutive raw
steps (~0.2s span), instead of the ~1.0s span (3 frames, 5 raw steps apart each) it
was trained on. Since PushT's per-raw-step block motion is small, this made the
eval-time history look like 3 near-duplicate/near-static frames — a degenerate
input distribution the model never saw in training, independent of (and compounding)
item 0's bug.

**Local fix**: added a `frameskip` parameter to `CJEPAHistoryPolicy`, gating the
deque append to only fire once every `frameskip` raw calls (via a per-env step
counter), and instantiate it with `frameskip=cfg.plan_config.action_block` in
`eval_wm.py` — `action_block` and the training data's `frameskip` are already
required to be equal for the action encoder's dimensions to line up (both are `5`
here), so it's already the right value to reuse, no new config field needed.

**Suggested upstream fix**: same patch, plus consider making the required stride
an explicit, named field (e.g. on `PlanConfig` or the model config) rather than
relying on the reader to notice `action_block` and the training frameskip happen to
need to match.

**Also noticed in passing (now fixed — see item 0d)**: `CJEPAWorldModel.
_single_step_predict` (`stable_worldmodel/wm/cjepa/cjepa.py`, ~line 330) hardcoded
history action embeddings to zero at inference (`hist_act = torch.zeros(...)`,
comment: "we don't have them during rollout"), while `forward_train`/
`_build_masked_tokens` always build real, non-zero action embeddings for history
positions from `batch['action']`. Initially deliberately left unfixed to keep
item 0c scoped, but item 0c's fix alone (retrained, re-evaluated) still produced
0% success — see item 0d for the fix and reasoning.

**Update**: even after the frame-stride fix, results were still statistically
indistinguishable from the 8% random baseline (6% -> 2% across two more runs).
Further empirical debugging (see item 0c) traced this to a training-time bug, not
a remaining eval-wiring issue.

---

### 0c. `_build_masked_tokens`: the future/target token was conditioned on the wrong action block — likely root cause of near-random MPC performance

**File**: `stable_worldmodel/wm/cjepa/cjepa.py` (`_build_masked_tokens`, was lines
~241-248)

After fixing items 0 and 0b, success rate was still statistically indistinguishable
from the 8% random-policy baseline (6%, then 2% on a repeat run) — despite training
loss being a healthy 0.004. This pointed away from eval-wiring bugs and back at the
model/training itself. An empirical diagnostic (loading the real checkpoint +
real dataset and calling `forward_train`/`get_cost` directly, bypassing CEM
entirely) confirmed this conclusively:
- `get_cost()` scored the real ground-truth expert trajectory *worse* than the
  median of 64 random action candidates (50th-66th percentile) — i.e. zero
  discriminative signal.
- `forward_train`'s loss was nearly identical (0.0037-0.0039) whether the action
  input was real, zeroed, shuffled across unrelated episodes, or scaled-up random
  noise — reproduced directly on real training data, not just at inference, so
  this is something the model *learned*, not an inference-side plumbing gap.

Root cause: the data pipeline stacks actions into per-position "blocks" where
block `i` is the frameskip-worth of raw actions taken **from** position `i`
**to** position `i+1` (`stable_worldmodel/data/dataset.py`,
`LanceDataset._process_batch`). `_build_masked_tokens` added `act_emb[:, tau]` to
the token at position `tau` for every `tau`, including the future/target position
`tau = T_h` (history_len). But block `T_h` is the action taken **after** the
target frame — it has nothing to do with how the target frame was reached from
the last history frame (that's block `T_h - 1`). So the one token the CEM-optimized
action is actually supposed to influence (the future/masked position) was, for the
entire 30-epoch run, conditioned on an action uncorrelated with the frame it was
predicting. This plausibly explains why the model converged to a low loss by
essentially ignoring the action channel altogether — a "copy the last frame
forward" solution already nearly minimizes masked-slot MSE given PushT's small
per-block motion, and the action signal at the one position where it should have
mattered was pointing at the wrong data anyway.

**Local fix**: shift the future position(s)' action embedding to use block
`T_h - 1` (the causally-correct action) instead of block `tau`, leaving history
positions' (`tau < T_h`) forward-looking convention unchanged (a separate, lower-
priority issue — see item 0b's zeroed-history-action note, deliberately not
addressed in this same pass to keep the fix scoped and low-risk):
```python
if act_emb is not None:
    act_emb_corrected = act_emb.clone()
    if T > T_h:
        act_emb_corrected[:, T_h:] = act_emb[:, T_h - 1 : T_h]
    aux_emb_t = act_emb_corrected + t_embs[:, None, :].unsqueeze(0)
    aux_parts.append(aux_emb_t)
```
**This requires retraining** — the existing `cjepa_run1` checkpoint already
converged under the buggy scheme and empirically learned to ignore actions; no
amount of eval-side patching fixes a model trained this way. A 5-epoch retrain
(vs. the original 30) was chosen for the initial re-check: the original run's own
loss curve shows most of the convergence shape happens by epoch ~10 (0.061 ->
0.011 by epoch 5, 0.004 by epoch 29), and the property being tested (does the
real-vs-corrupted-action loss gap actually open up now) is a relative comparison,
not a "reach the lowest possible loss" one.

**Suggested upstream fix**: same patch. Given how easy this was to get backwards
(the "obvious" `act_emb[:, tau]` reads correctly for history positions, and only
subtly wrong at the boundary), this seems worth an explicit unit test asserting
the future token's action index equals `history_len - 1`, not just a shape check.

**Update**: retrained 5 epochs with this fix (`cjepa_run2`). Direct diagnostics
showed a real, if modest, improvement — `forward_train` loss now consistently
ranks real actions below zeroed/shuffled/noisy ones (unlike before, where all
were statistically identical), and `get_cost()`'s real-vs-random percentile
improved from "always ~50th or worse" to a mix of 11th-88th across 5 test
episodes. But the full MPC eval still scored **0%** — no improvement over (and
arguably worse than) the 8% random baseline. This pointed back at the other
still-open issue: item 0b's zeroed history actions. See item 0d.

---

### 0d. Wire up real history-action tracking, fixing item 0b for real

**Files**: `scripts/plan/cjepa_policy.py` (`CJEPAHistoryPolicy`),
`stable_worldmodel/wm/cjepa/cjepa.py` (`_single_step_predict`, `rollout`)

With item 0c fixed and retrained, the future/target token now gets the
causally-correct action — but `_single_step_predict` still hardcoded EARLY
history positions' (all but the last) action embeddings to zero
(`hist_act = torch.zeros(...)`), while training always uses real ones. Full
eval after the 0c retrain was still 0%, consistent with this remaining gap
also mattering.

One subtlety worth recording: history positions use a **different** action
convention than the (now-fixed) future position. `_build_masked_tokens` was
deliberately left unchanged for history positions (`tau < T_h`) — they still
use block `tau` itself, i.e. "the action about to be taken **from** this
position" (forward-looking), not "the action that produced this position"
(backward-looking, which is what the future position needed). Working through
what that means for the LAST history position (index `T_h - 1`): its
forward-looking action ("from position `T_h-1` to `T_h`") is, by definition,
identical to block `T_h - 1` — the exact same action already used for the
future/target position after the 0c fix. So the last history position doesn't
need any new plumbing at all; it was already implicitly available. Only the
EARLIER history positions (`0` .. `T_h - 2`) need real, actually-past actions
threaded in, since those correspond to actions genuinely already executed by
the time of planning, not the CEM candidate under evaluation.

**Local fix**:
- `CJEPAHistoryPolicy.get_action` now also buffers the real raw actions taken
  between sampled history frames (via `info_dict['action']`, which the env
  wrapper already populates with "the action just executed" —
  `stable_worldmodel/wrapper/default.py`), frameskip-stacks them into blocks
  exactly matching the training data pipeline's own stacking
  (`stable_worldmodel/data/dataset.py`), and normalizes each raw action with
  the same `process['action']` scaler **before** stacking — matching the
  training pipeline's actual order of operations (`LanceDataset._load_slice`
  applies the dataset transform, including the z-score scaler, to raw
  per-step actions, and only afterwards does `Dataset.__getitem__` reshape
  into frameskip blocks). Exposes this as `info_dict['hist_action']`,
  shape `(n_envs, history_len - 1, frameskip * act_dim)`.
- `CJEPAWorldModel._single_step_predict` gained a `hist_act_emb` parameter for
  the early history positions, and reuses `act_emb_step` (unshifted) for the
  last history position instead of zeroing it — both were previously zeroed
  unconditionally.
- `CJEPAWorldModel.rollout()` encodes `info['hist_action']` (when present) via
  the same `action_encoder`, then **slides it forward in lockstep with
  `slots_h`** across the autoregressive planning loop — after the first
  planning step, the window's "early" history actions are this same
  rollout's own earlier candidate actions, not the original real ones,
  exactly mirroring how `slots_h` mixes real and predicted slots as it
  slides.
- No retraining needed for this one — it's purely an inference-time fix;
  training already always used real history actions (only the future
  position had a real train/inference gap, which was item 0c).

**Suggested upstream fix**: same patches. Longer-term, `WorldModelPolicy`
itself could track a generic raw-action history (not just CJEPA-specific)
so world models with an action-conditioned history don't each need a bespoke
policy subclass for this.

---

### 1. `eval_wm.py`/`eval_ff.py`: episode-index column resolution is broken for Lance datasets

**Files**: `scripts/plan/eval_wm.py` (was lines 34, 76-77, 139 — fixed locally),
`scripts/plan/eval_ff.py` (lines 38, 79, 128 — same bug, **not yet fixed**)

Both scripts guess the episode-index column name with:
```python
col_name = 'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
```
This works for the `lerobot` dataset format (`stable_worldmodel/data/formats/lerobot.py`),
whose `column_names` deliberately includes its synthetic `ep_idx`/`step_idx` columns.
It silently breaks for `LanceDataset` (`stable_worldmodel/data/formats/lance.py:181-183,229-230`),
whose `column_names` property deliberately *excludes* the two index columns — they're
reserved/internal, not "data columns". So on any Lance dataset the check always misses
and falls through to the wrong, nonexistent name `'ep_idx'`, crashing with:
```
ValueError: Invalid user input: Schema error: No field named ep_idx.
```
The real column is always `'episode_idx'` — hardcoded in the Lance writer
(`lance.py:870`) and never overridden anywhere in this repo (`grep` confirms zero
`episode_index_column=` overrides exist). This strongly suggests `eval_wm.py` had
never actually been run end-to-end against a real Lance dataset before now.

**Local fix** (`eval_wm.py`, all 3 occurrences): check the raw schema instead of the
filtered `column_names`:
```python
schema_names = getattr(dataset, '_schema_names', dataset.column_names)
col_name = 'episode_idx' if 'episode_idx' in schema_names else 'ep_idx'
```
**Suggested proper upstream fix**: add a public accessor on `LanceDataset` (e.g. an
`episode_index_column` property, or don't exclude index columns from a lower-level
schema-introspection method) so callers don't need to reach into a private
`_schema_names` attribute. Same patch still needed in `eval_ff.py` (untouched, since
we didn't exercise that script this session).

---

### 2. `eval_wm.py`: `get_row_data()` doesn't return index columns on Lance datasets

**File**: `scripts/plan/eval_wm.py` (was lines 158-159 — fixed locally)

```python
eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
eval_start_idx = dataset.get_row_data(random_episode_indices)['step_idx']
```
`LanceDataset.get_row_data()` (`lance.py:540-562`) only returns entries for
`self._keys` (data columns), which — same root cause as bug #1 — excludes the index
columns by design. This raises `KeyError: 'episode_idx'` immediately after bug #1 is
fixed.

**Local fix**: use `get_col_data()` (already used elsewhere in this same script)
indexed by row instead of `get_row_data()`:
```python
eval_episodes = dataset.get_col_data(col_name)[random_episode_indices]
eval_start_idx = dataset.get_col_data('step_idx')[random_episode_indices]
```
**Suggested upstream fix**: same patch, or extend `get_row_data` to optionally include
index columns so call sites don't need to special-case them.

---

### 3. `scripts/plan/config/pusht.yaml`: stale `.h5` dataset default

**File**: `scripts/plan/config/pusht.yaml` (`eval.dataset_name`, was `pusht_expert_train.h5`)

No script in the repo produces a `.h5` PushT dataset — every PushT collection script
(`collect_pusht_fov.py`, `collect_pusht_toy.py`, `collect_pusht_smoke.py`,
`collect_weak_pusht.py`) writes Lance format. The shipped default silently points at a
file that can never exist unless a user happens to know to override it.

**Local fix**: changed to `pusht_expert_train.lance`.
**Suggested upstream fix**: same change (or generalize so the config isn't
hardcoding a specific format/extension that has to match whatever the user actually
collected).

---

### 4. `scripts/train/cjepa.py` (and likely sibling `scripts/train/*.py`): `subdir` vs `output_model_name` split is a checkpoint-overwrite footgun

**Files**: `scripts/train/cjepa.py:139-153`, `scripts/train/config/cjepa.yaml:6-7`,
`stable_worldmodel/wm/utils.py:23` (`save_pretrained`)

`subdir` (`cjepa.yaml:7`, default `${hydra:job.id}` — a random Hydra job id) *looks*
like the knob for naming a training run, and is the more discoverable-sounding CLI
override (`subdir=my_run`). But it only controls where the full Hydra config
snapshot (`config.yaml`) gets written (`cjepa.py:139-149`) — a file nothing else
reads. The actual model weights, and the `config.json` that `load_pretrained()`
reads, are saved under `checkpoints/<output_model_name>/`
(`SaveCkptCallback(run_name=cfg.output_model_name, ...)` → `save_pretrained(...,
run_name=...)`), and `output_model_name` defaults to the fixed literal `'cjepa'`
(`cjepa.yaml:6`) — not tied to `subdir` at all.

Net effect: overriding `subdir=my_run` silently does nothing for checkpoint
location, and every run that doesn't override `output_model_name` overwrites the
same shared `checkpoints/cjepa/` directory. We hit this directly this session — two
1-epoch dry runs and an initial pass all landed in `checkpoints/cjepa/`, and the plan
to reference a `subdir`-named checkpoint for eval had to be corrected to use
`output_model_name` instead once we traced the actual code path.

**Status**: not fixed upstream — worked around locally by using
`output_model_name=cjepa_run1` directly.
**Suggested upstream fix**: unify these into a single config key (or default
`output_model_name` to `subdir`'s value), and/or clarify in `cjepa.yaml`'s comments
what each one actually controls. Worth checking whether `lewm.py`/`prejepa.py`/other
`scripts/train/*.py` share the same pattern (they appear structurally similar per
this doc's "mirror of `lewm.py`" note above).

---

### 5. (Operational, not strictly a code bug) `persistent_workers=True` + crashed runs leak orphaned worker processes

**File**: `scripts/train/config/cjepa.yaml` (`loader.num_workers: 6`,
`persistent_workers: True`, `prefetch_factor: 3`)

On a memory-capped container (cgroup limit, not host RAM — see the RTX 2000 Ada
session log entry above for the full incident), a training run that
gets OOM-killed (SIGKILL) leaves its `multiprocessing.forkserver` dataloader worker
processes running and holding memory indefinitely, since `persistent_workers=True`
means they're never torn down through the normal per-epoch teardown path a SIGKILL
bypasses. This isn't really an app bug (general PyTorch DataLoader/forkserver
characteristic), but it's a real gotcha on cost-conscious/memory-constrained
dev pods: one crashed run can quietly eat ~15GB+ of RAM until someone notices and
manually `pkill`s the leftover workers. Worth a callout in the training script's
docs/README (e.g. "if a run OOMs, check for and kill orphaned
`multiprocessing.forkserver` processes before retrying") rather than a code change.

**Update (2026-07-06)**: this recurred on a completely fresh Pod (no prior crashed
run, no leaked workers possible) — a brand-new `python3 scripts/train/cjepa.py`
process with the shipped `cjepa.yaml` defaults (`num_workers=6`,
`persistent_workers=True`, `prefetch_factor=3`) got SIGKILL'd (exit 137) on its
very first attempt, right as training was about to start (before any step
completed). So this isn't purely a "clean up after a crash" gotcha — the shipped
defaults themselves are apparently too memory-hungry for a ~29GB-cgroup-capped
dev pod even in the best case. Reducing to `num_workers=2`, `prefetch_factor=2`,
`persistent_workers=False` fixed it immediately (trained cleanly end-to-end
afterwards). **Suggested upstream fix**: lower `cjepa.yaml`'s (and likely the
other `scripts/train/config/*.yaml`, which mostly share the same
`num_workers=6`/`persistent_workers=True`/`prefetch_factor=3` pattern) shipped
defaults, or at least document the memory tradeoff next to the config keys,
rather than relying on every session rediscovering this the hard way.

---

### Not filed as upstream bugs (already tracked elsewhere in this doc)

- The Docker image torch/CUDA driver mismatch blocking all GPU training on fresh
  pods, and the unrelated `torchaudio` import breakage — both are `swm-runpod`
  infra issues, not `stable-worldmodel` bugs; see the 2026-07-01 and 2026-07-03/04
  Session Log entries above (fixed permanently in `Dockerfile`, commit `ef7a99a`).
- No script in the repo produces `pusht_expert_train.lance` under that exact name
  (the Phase 5 checklist's suggested `collect_pusht_fov.py` does a different,
  per-variation sweep instead) — this was project-doc/planning debt, not an
  upstream `stable-worldmodel` bug; covered by writing
  `scripts/data/collect_pusht_expert_train.py` locally (see Phase 5 checklist).

---

## Session Log

*(Pre-CJEPA prototype, 2026-06-19, not itemized below: an earlier, simpler
DINOv2+ConvAdapter+1-layer-Transformer pipeline was built in
`train_eval_pusht.ipynb` — frozen DINOv2 ViT-S/14 → ConvAdapter → 2D pos emb →
1-layer Transformer, CEM+cosine-distance MPC cost — before the project pivoted
to reproducing the C-JEPA architecture described above. Superseded; kept only
as historical context for anyone diffing `train_eval_pusht.ipynb`'s origin.)*

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
- **Environment blocker, unrelated to CJEPA code**: `torch.cuda.is_available()` was `False` on this fresh Pod — `nvidia-smi` reports driver 570.195.03 (CUDA 12.8 max), but `Dockerfile`'s unpinned `pip install 'stable-worldmodel[all]'` resolved PyPI's latest torch (2.12.1+cu130, CUDA 13.0), which the driver can't run. This blocks *all* GPU training (`trainer.accelerator: gpu` in every `scripts/train/*.yaml`), not just VideoSAUR. Same root pattern as Phase 3's `torchaudio` fix — baked into the image, so it recurs on every fresh Pod since the Dockerfile was never patched. Fixed locally this session (`pip uninstall torchaudio` + reinstall `torch==2.12.1+cu126`/`torchvision==0.27.1+cu126` — same versions, just the CUDA-12 build) and verified via full `pytest tests/wm/` + `scripts/train/smoke_test_videosaur.py` both passing after the swap. **Update (2026-07-04): made permanent.** `Dockerfile` now pins `torch==2.12.1`/`torchvision==0.27.1` from the `cu126` wheel index before `stable-worldmodel[all]` installs, and uninstalls the leftover `torchaudio` — pushed to `main` as `ef7a99a`, GitHub Actions rebuilt and pushed `b8k3/swm-dev:latest` successfully. Re-verified on a fresh Pod against the rebuilt image: `torch.cuda.is_available() == True` with no `torchaudio` present, full `pytest tests/wm/` (48/48) and `scripts/train/smoke_test_videosaur.py` both passing. Confirmed still in place as of the 2026-07-06 session.
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
- **6 real bugs found and fixed this session, written up for upstream contribution in the "Upstream Bug List" section below** (episode-index column resolution, `get_row_data` index-column gap, stale eval dataset default, wrong policy class, wrong history-frame stride, non-deterministic slot encoder, action/timestep misalignment in training, zeroed history-action embeddings at inference) — independently valuable regardless of this project's outcome.
- **Decision**: stop iterating on further suspected bugs blind. Task success has not moved off random-chance level despite 5 real, well-evidenced fixes, which suggests either a remaining issue not yet found, or a more fundamental representation/training problem not reducible to a simple wiring bug. Next session: validate intermediate pipeline stages individually (slot encoder output quality, learned representation structure, dataset statistics) against a known-good reference (e.g. DINO-WM) rather than only checking the final end-to-end MPC number — establish per-stage success criteria before further debugging the full pipeline blind.
- **Next**: stage-by-stage validation (data → slot encoding → learned dynamics → planning) with calibrated success criteria per stage, informed by how DINO-WM/other baselines in this repo validate each of those stages. Continued in the 2026-07-06 entry below.

### 2026-07-06 — Stage-by-stage validation: encoder cleared, action-sensitivity signal real but too inconsistent for CEM

- Ran on a fresh Pod — the previous Pod's `pusht_expert_train.lance`, VideoSAUR checkpoint cache, and `cjepa_run1`/`cjepa_run2` checkpoints were all gone (none had been pushed to S3), but the code fixes themselves were safe, already committed at `06c97cd`. Re-collected the dataset, re-downloaded the VideoSAUR checkpoint, and retrained a fresh `cjepa_run2` (5 epochs, final val loss 0.0026 — closely matching the prior session's 0.0027).
- Considered, then deprioritized, sanity-checking the MPC harness against a from-scratch-trained baseline (LeWM/DINO-WM/PLDM) through `eval_wm.py` first: all three also require pre-stacked multi-frame `pixels` in their own `rollout()`, but expose it as `history_size`/`predictor.num_frames`, never `history_len` — so `eval_wm.py`'s auto-selection (`getattr(model, 'history_len', None)`) would never route any of them to a history-aware policy either. Not the zero-friction reference it first looked like; set aside in favor of two direct, per-stage diagnostics instead (added as reusable scripts under `scripts/diagnose/`):
  - **`scripts/diagnose/slot_representation_probe.py`** — probes the frozen VideoSAUR encoder in isolation. Ridge-regression content probe: R²=0.964 vs. `block_pose`, R²=0.907 vs. `pos_agent` (n=980 frames, 50 episodes). Hungarian-matched frame-to-frame cosine similarity: 0.993 (stride 1) → 0.989 (stride 2, VideoSAUR's own training frameskip) → 0.978 (stride 5, this project's actual data frameskip — the domain-shift risk flagged in Phase 3 but never checked until now). Degradation from the domain shift is real but small (~0.011 absolute) — **the frozen encoder is very unlikely to be the bottleneck.**
  - **`scripts/diagnose/action_sensitivity_check.py`** — scales the prior session's 5-episode ad hoc `get_cost()` real-vs-random-action percentile check to n=64 held-out episodes (K=32 random candidates each, Gaussian). Mean percentile 43.7, median 39.1 (below the 50 random-chance baseline — a real average signal), but std=30.2 and only 54.7% of episodes have the real action beating the median random candidate; 10.9% of episodes the real action is worse than 90% of random candidates. Confirms a real, non-noise signal on aggregate, but quantifies it as currently far too inconsistent per-episode for CEM to reliably exploit.
- **Conclusion**: both diagnostics together point the remaining gap toward the predictor/training regime or the Hungarian-matched cost aggregation itself, not the frozen representation quality and not a residual wiring bug.
- The OOM/orphaned-forkserver issue recurred on this fresh Pod on the very first training attempt — this time not a leftover-worker issue (a clean process hit the ~29GB cgroup cap with the shipped defaults `num_workers=6`/`persistent_workers=True`/`prefetch_factor=3`). Reducing to `num_workers=2`/`prefetch_factor=2`/`persistent_workers=False` fixed it immediately. Worth lowering `cjepa.yaml`'s shipped defaults rather than relying on every session rediscovering this (see item 5 in the Upstream Bug List below).
- **Next investigation plan (queued, then executed same session)**: the action-sensitivity check's mean (43.7) hides a std of 30.2 — close to uniform over 0–100. 17.2% of episodes are excellent (real action beats ≥90% of random candidates), 10.9% are actively inverted. This looks like state-dependent reliability CEM can't detect in advance (it plans one state at a time), not uniform model weakness. Plan: extend `action_sensitivity_check.py` to ~150-200 episodes, recording per-episode covariates (`block_motion`, `agent_motion`, `contact_change`, `real_action_magnitude`, `history_slot_stability`) alongside the percentile score, then compute Spearman correlations and compare top-vs-bottom-quartile covariate distributions. Interpretation guide: low motion → residual copy-forward shortcut (reweight loss toward higher-motion transitions); `contact_change` → nonlinear contact transitions are the hard cases (upweight in training); `history_slot_stability` → Hungarian re-matching itself is noisy (try soft assignment or anchor-to-t0 matching); no covariate correlates → genuine stochastic/estimator variance (try ensembled predictions before concluding architectural dead end). See the "Next investigation plan" section above (Phase 5) for full detail.
- **Executed the plan same session**: wrote `scripts/diagnose/percentile_covariates.py` (copy of `action_sensitivity_check.py`, extended with the 5 covariates above; built its own dataset instance with `block_pose`/`pos_agent`/`n_contacts` added to `keys_to_load`, deliberately without z-score normalizing those three since `block_motion`/`agent_motion` are raw-unit). Sanity-checked raw `n_contacts` values first (small integers, 0/1/2 — as expected for a per-raw-step contact-point count) before trusting the `contact_change` formula. Ran a 5-episode dry run (no crashes, sane values), then the full n=200 pass:
  - **None of `block_motion`/`agent_motion`/`contact_change`/`history_slot_stability` correlate significantly** (Spearman p=0.43/0.26/0.52/0.68). `history_slot_stability`'s reliable- vs. unreliable-quartile means were identical to 4 decimal places (0.9250 vs. 0.9250) — a strong further confirmation the frozen encoder isn't where the inconsistency comes from.
  - **`real_action_magnitude` is the one significant covariate** (r=0.323, p<0.0001) — larger real actions (z-scored units) tend to rank *worse* against random candidates (reliable-quartile mean 3.03 vs. unreliable-quartile mean 3.47). This wasn't one of the four interpretation-guide branches; by the letter of the guide it falls under "no covariate correlates meaningfully" for the four originally-hypothesized factors, and its likely mechanism (single-forward-pass cost estimates plausibly noisier for larger, rarer actions) points at the same suggested next step as that branch: try ensembled/averaged predictor forward passes.
  - **Flagged, not resolved**: this n=200 run's baseline (mean percentile 51.8, median 53.1) sits much closer to pure random chance than the n=64 check's (mean 43.7, median 39.1) — the earlier "real average signal" framing may have been partly sample-noise from the smaller n. Not root-caused this session.
  - Raw per-episode results saved to `scripts/diagnose/percentile_covariates_results.csv`. Per the agreed scope for this pass, stopped at diagnosis — implementing the ensembling follow-up is a separate future task.
- **Ensembling follow-up, tried same session**: before ensembling anything, checked whether `get_cost()` has any per-call randomness to average away at all — it doesn't. Confirmed empirically (`model._extract_slots(x)` called twice on identical input returns bit-identical tensors): the trained predictor has `dropout=0.0` throughout, and the VideoSAUR slot encoder's `RandomInit` was already fixed in an earlier session to use a hardcoded `manual_seed(0)` every call, specifically to stop unseeded noise from swamping the cost signal. So literally repeating `get_cost()` on the same input and averaging is a no-op by construction — there is no single-forward-pass noise left in this model to ensemble away.
  - The one architecturally real (but currently pinned) source of stochasticity is `RandomInit`'s slot-space anchor — Slot Attention is normally randomly initialized by design. Gave `RandomInit` a settable `seed` attribute (`stable_worldmodel/wm/cjepa/_videosaur/initializer.py`, default 0, so all existing call sites are unaffected) and wrote `scripts/diagnose/ensembled_action_sensitivity_check.py`: for each of the same 200 held-out episodes, run `get_cost()` M=5 times with different seeds (0-4, held constant *within* each pass's history+goal encoding, varied *across* passes — preserving the original fix's self-consistency requirement), average the resulting cost tensor, then recompute the percentile, compared against the seed=0-only single-pass baseline on the identical episodes.
  - Caught and fixed a real bug while building this: `rollout()` mutates its `info` dict in place (`info['slots'] = ...`) and returns the same object, so reusing one `info` dict object across the 5 seed passes silently cached the first pass's encoding and skipped re-encoding on every subsequent seed (all 5 "different" passes came back identical). Fixed by passing a fresh shallow copy of `info` into `get_cost()` on every pass.
  - **Result: ensembling does not help, and isn't just a no-op — it adds noise without any averaging benefit.** Single-pass baseline: mean percentile 49.4, median 50.0, std 29.8. Ensembled (mean cost over 5 seeds): mean 49.5, median 50.0, std 30.6 — essentially unchanged on average, slightly *worse* std. Per-episode, single-pass and ensembled percentiles correlate at only r=0.815 (not 1.0, confirming the seed genuinely changes per-episode results) with 18 episodes flipping from unreliable→reliable and 17 flipping reliable→unreliable — a roughly symmetric, non-directional shuffle, not noise cancellation. This makes sense in hindsight: the predictor was trained against exactly one fixed slot-space anchor (seed=0); a different seed doesn't resample noise around the same estimate, it evaluates the model at a different, never-trained-for anchor point — there's no reason to expect that to average out favorably.
  - **Conclusion**: the "single-forward-pass estimator variance" branch of the interpretation guide doesn't apply to this model as built — there's no such variance to average away, and manufacturing some via the seed makes things slightly worse, not better. Combined with `percentile_covariates.py`'s result (no physical/representational covariate correlates either), this points toward the remaining "genuine stochastic variance in the learned dynamics" reading: the per-episode inconsistency most plausibly reflects a real limitation of what the predictor learned (state-dependent accuracy), not a measurement-noise artifact fixable by averaging at inference time. Next real lever is more likely training-side (more data/epochs/capacity, or an architecture change) than an inference-time trick — not attempted this session.
