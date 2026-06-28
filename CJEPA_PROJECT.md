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

### Phase 3: VideoSAUR integration
- [ ] Find HuggingFace checkpoint for VideoSAUR (check `HazelNam/CJEPA` or `galilai-group/`)
- [ ] Implement `VideoSAUREncoder` wrapper that returns (B, N, D) slots per frame
- [ ] Smoke-test: load checkpoint, run on single PushT frame, verify slot shapes

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
