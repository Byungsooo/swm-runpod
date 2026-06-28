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

### Phase 1: Core model
- [ ] `stable_worldmodel/wm/cjepa/__init__.py`
- [ ] `stable_worldmodel/wm/cjepa/module.py`
  - [ ] `BidirectionalPredictor` (reuse `lewm/module.py:Block` with `causal=False`)
  - [ ] `IdentityAnchorProjection` (`nn.Linear(128, 128)`, the `φ` in Eq. 3)
  - [ ] `LearnableTemporalPosEmb` (`nn.Embedding(max_T, 128)`, the `e_τ`)
  - [ ] `AuxEmbedder` (copy of `lewm/module.py:Embedder` for actions/proprio)
- [ ] `stable_worldmodel/wm/cjepa/cjepa.py`
  - [ ] `encode()` — VideoSAUR slots + aux embeddings
  - [ ] `mask_history()` — object-level masking logic, identity anchor
  - [ ] `forward_train()` — masking → predict → L_history + L_future
  - [ ] `rollout()` — inference-only, future-only masking + Hungarian matching
  - [ ] `criterion()` / `get_cost()` — Costable protocol for CEM

### Phase 2: Training pipeline
- [ ] `scripts/train/cjepa.py` (copy of `lewm.py`, replace forward fn)
- [ ] `scripts/train/config/cjepa.yaml`
  - [ ] n_slots=4, slot_dim=128, history_len=3, future_len=1, max_masked=2
  - [ ] predictor: depth=6, heads=16, dim_head=64, mlp_dim=2048
  - [ ] VideoSAUR slot_encoder (HuggingFace model id)
  - [ ] 30 epochs, Adam lr=5e-4, batch=256

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

### 2026-06-28 — Planning session
- Read C-JEPA paper in full; understood architecture, masking strategy, training objective
- Assessed stable-worldmodel codebase — confirmed no existing C-JEPA, identified all reusable components
- Finalized implementation plan: ~400–600 lines of novel code, rest is plumbing reuse
- Estimated cost: ~$1–2/run, ~$5–10 to reproduce paper results
- **Next**: start Phase 1 (core model code) in a new coding session
