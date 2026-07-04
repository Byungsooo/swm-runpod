# Upstream Contribution Candidates (stable-worldmodel)

Bugs and gaps found in `stable-worldmodel` (not our CJEPA code) while running Phase 5
of `CJEPA_PROJECT.md`. Filed here so they don't get lost; several are already patched
locally in `/workspace/stable-worldmodel` and just need to be pushed upstream as their
own small PR(s), separate from the CJEPA contribution itself.

---

## 0. `eval_wm.py`: uses the wrong policy class for history-conditioned world models, silently corrupting MPC eval (most severe finding this session)

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

## 0b. `cjepa_policy.py`'s `CJEPAHistoryPolicy` samples history frames at the wrong stride (every raw step, not every `frameskip` steps)

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

## 0c. `_build_masked_tokens`: the future/target token was conditioned on the wrong action block — likely root cause of near-random MPC performance

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

## 0d. Wire up real history-action tracking, fixing item 0b for real

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
need any new plumbing at all; it was already implicitly available.  Only the
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

## 1. `eval_wm.py`/`eval_ff.py`: episode-index column resolution is broken for Lance datasets

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

## 2. `eval_wm.py`: `get_row_data()` doesn't return index columns on Lance datasets

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

## 3. `scripts/plan/config/pusht.yaml`: stale `.h5` dataset default

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

## 4. `scripts/train/cjepa.py` (and likely sibling `scripts/train/*.py`): `subdir` vs `output_model_name` split is a checkpoint-overwrite footgun

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
`CJEPA_PROJECT.md`'s "mirror of `lewm.py`" note).

---

## 5. (Operational, not strictly a code bug) `persistent_workers=True` + crashed runs leak orphaned worker processes

**File**: `scripts/train/config/cjepa.yaml` (`loader.num_workers: 6`,
`persistent_workers: True`, `prefetch_factor: 3`)

On a memory-capped container (cgroup limit, not host RAM — see the RTX 2000 Ada
session log entry in `CJEPA_PROJECT.md` for the full incident), a training run that
gets OOM-killed (SIGKILL) leaves its `multiprocessing.forkserver` dataloader worker
processes running and holding memory indefinitely, since `persistent_workers=True`
means they're never torn down through the normal per-epoch teardown path a SIGKILL
bypasses. This isn't really an app bug (general PyTorch DataLoader/forkserver
characteristic), but it's a real gotcha on cost-conscious/memory-constrained
dev pods: one crashed run can quietly eat ~15GB+ of RAM until someone notices and
manually `pkill`s the leftover workers. Worth a callout in the training script's
docs/README (e.g. "if a run OOMs, check for and kill orphaned
`multiprocessing.forkserver` processes before retrying") rather than a code change.

---

## Not filed here (already tracked elsewhere)

- Docker image torch/CUDA driver mismatch blocking all GPU training on fresh pods,
  and the unrelated `torchaudio` import breakage — both already documented in
  `TODO_FIX_DRIVER_TORCH_MISMATCH.md` (this repo).
- No script in the repo produces `pusht_expert_train.lance` under that exact name
  (the Phase 5 checklist's suggested `collect_pusht_fov.py` does a different,
  per-variation sweep instead) — this is project-doc/planning debt in
  `CJEPA_PROJECT.md`, not an upstream `stable-worldmodel` bug, so no action needed
  there; we wrote `scripts/data/collect_pusht_expert_train.py` locally to cover it.
