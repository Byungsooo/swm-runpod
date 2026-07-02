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

### Phase 4: Smoke test
- [ ] CPU unit test: shapes, masking logic, loss is not NaN
- [ ] 1-epoch smoke run: `python scripts/train/cjepa.py trainer.max_epochs=1 loader.batch_size=8`
- [ ] Verify loss decreases

### Phase 5: Full training + eval
- [ ] Run 30-epoch training on RTX 4090
- [ ] Collect PushT data if not already present (use `collect_pusht_fov.py`)
- [ ] Run MPC eval: `python scripts/plan/eval_wm.py` with CJEPA checkpoint
- [ ] Compare to Table 3 baselines: OC-JEPA (76%), C-JEPA target (88.67%)

### Phase 6: PR preparation
- [ ] Add `tests/wm/test_cjepa.py` (shape checks, masking, loss)
- [ ] Update `stable_worldmodel/wm/__init__.py` to export CJEPAWorldModel
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
