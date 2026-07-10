# stable-worldmodel Contribution Ideas

A working list of bugs, gaps, and DX ideas found in `stable-worldmodel` (not our
CJEPA code) while building the CJEPA reproduction and, later, the DINO-WM
(`PreJEPA`) reference run — both in this fork. Most are already patched locally
in `/workspace/stable-worldmodel`. This file exists to turn that into actual
PRs against `galilai-group/stable-worldmodel`, separate from the CJEPA/DINO-WM
research itself.

Items 1-9 below were originally tracked as items `0`/`0b`/`0c`/`0d`/`1`-`5` in
`CJEPA_PROJECT.md`'s "Upstream Bug List" section (2026-06-28 through
2026-07-06 sessions) — full root-cause narratives and reasoning live there;
this file is the condensed, PR-shaped version. Items 10+ are new, from the
2026-07-07/07-10 DINO-WM reproduction sessions and not yet written up
anywhere else.

Status legend: **Fixed locally** (patched in this fork, not yet upstreamed) /
**Not fixed** (worked around or left open) / **Idea** (no bug, a DX/feature
suggestion).

---

## High severity — silently wrong results, no crash

### 1. `WorldModelPolicy`/`eval_wm.py` history handling doesn't generalize across the repo's own baselines
**Status: Fixed locally (CJEPA-specific), gap remains for other models**

`eval_wm.py` originally passed 1 frame/step straight through to whatever
world model it was evaluating. History-conditioned models (fixed-size
temporal position embeddings, action-history buffers, etc.) silently get fed
the wrong shape and produce plausible-looking but wrong predictions — this
made a fully-converged CJEPA checkpoint score *worse than random* (4% vs. 8%)
with no crash or warning. Fixed for CJEPA by wiring in the pre-existing
`CJEPAHistoryPolicy` and gating on `getattr(model, 'history_len', None)`.

But: none of the repo's other shipped world models (`LeWM`, `PreJEPA`,
`PLDM`) expose `history_len` — they use `history_size`/`predictor.num_frames`
instead (confirmed via `grep -rn "self\.history_len\s*=" stable_worldmodel/wm/`
matching only `CJEPAWorldModel`). So `eval_wm.py`'s auto-selection never
routes any of them to a history-aware policy path either; they happen to work
because their own `rollout()`/`get_cost()` re-derive/re-request the frames
they need internally, but the pattern is fragile and CJEPA-specific as
written.

**Suggested fix**: make `WorldModelPolicy` itself history-aware — check for a
common attribute/small protocol (`get_history_length()` or similar) that all
world models implement, rather than a per-caller `getattr(model,
'history_len', None)` check hardcoded to one model's naming.

### 2. `CJEPAWorldModel._build_masked_tokens`: future/target token conditioned on the wrong action block
**Status: Fixed locally, requires retraining**

`_build_masked_tokens` added `act_emb[:, tau]` at every position `tau`,
including the future/target position `tau = T_h`. But block `T_h` is the
action taken *after* the target frame, not the one that produced it (that's
block `T_h - 1`). Net effect: the one token CEM-optimized actions are
supposed to influence was, for an entire 30-epoch run, conditioned on an
action uncorrelated with what it was predicting — the model converged to a
healthy-looking loss (0.004) by learning to ignore the action channel
entirely. Caught via direct diagnostics (`get_cost()` on real vs.
zeroed/shuffled/noise actions all statistically identical), not from the loss
curve.

**Suggested fix**: same patch (shift future position(s)' action embedding to
block `T_h - 1`). Given how easy this is to get backwards, also worth an
explicit unit test asserting the future token's action index equals
`history_len - 1`.

### 3. `CJEPAWorldModel._single_step_predict`: history-frame action embeddings zeroed at inference, real at train time
**Status: Fixed locally**

Early history positions' action embeddings were hardcoded to zero at
inference (`hist_act = torch.zeros(...)`, comment: "we don't have them during
rollout") while training always used real ones — a genuine train/inference
distribution gap, independent of and compounding item 2.

**Suggested fix**: same local patch — thread real executed-action history
through the policy (buffered, frameskip-stacked, normalized the same way the
training pipeline does) and into `rollout()`. Longer-term, `WorldModelPolicy`
could track a generic raw-action history so world models with
action-conditioned history don't each need a bespoke policy subclass.

### 4. VideoSAUR `RandomInit` slot initializer is unseeded — non-deterministic even on identical input
**Status: Fixed locally**

Two encodings of the *identical* frame diverged by ~58% of total signal
energy (post best-case Hungarian realignment) purely from unseeded slot-space
noise. Fixed with a deterministic `manual_seed(0)` per call; later given a
settable `seed` attribute (default 0, so existing call sites are unaffected)
to support an ensembling experiment.

**Suggested fix**: same — seed by default, expose the seed as a constructor
param.

---

## Crashes on first real use against non-toy data

### 5. `eval_wm.py`/`eval_ff.py`: episode-index column resolution broken for Lance datasets
**Status: `eval_wm.py` fixed locally; `eval_ff.py` not yet fixed (same bug, not exercised)**

```python
col_name = 'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
```
Works for `lerobot`-format datasets, whose `column_names` includes its
synthetic index columns. Silently breaks for `LanceDataset`, whose
`column_names` deliberately *excludes* the two index columns — the check
always misses and falls through to a name (`ep_idx`) that never exists on
Lance, crashing with `ValueError: ... No field named ep_idx`. Strongly
suggests `eval_wm.py` had never been run end-to-end against a real Lance
dataset before this project did.

**Suggested fix**: add a public accessor on `LanceDataset` (e.g.
`episode_index_column` property) instead of reaching into
`dataset._schema_names`. Needs the same fix applied to `eval_ff.py`.

### 6. `eval_wm.py`: `get_row_data()` doesn't return index columns on Lance datasets
**Status: Fixed locally**

`LanceDataset.get_row_data()` only returns entries for data columns, same
root cause as #5 — raises `KeyError: 'episode_idx'` immediately after that
fix is applied. Worked around with `get_col_data(col_name)[indices]` instead
(already used elsewhere in the same script).

**Suggested fix**: same patch, or extend `get_row_data` to optionally include
index columns.

### 7. `scripts/plan/config/pusht.yaml`: stale `.h5` dataset default
**Status: Fixed locally**

`eval.dataset_name` defaulted to `pusht_expert_train.h5`, but every PushT
collection script in the repo writes Lance format — the shipped default
points at a file that can never exist.

**Suggested fix**: change default to `pusht_expert_train.lance`, or stop
hardcoding a format-specific extension in the config.

### 8. `FolderDataset` (parent of `VideoDataset`) doesn't accept `keys_to_cache`
**Status: Fixed locally, committed (`7fe49eb`), not yet upstreamed**

`scripts/train/prejepa.py` always passes `keys_to_cache` to `load_dataset(...)`
— every other format (`lance.py`, `hdf5.py`) already accepts it, but
`FolderDataset.__init__` didn't, raising `TypeError` on first use against a
video-format dataset. Fixed by accepting the kwarg (mirroring `lance.py`'s
own "not required" warning, since `FolderDataset` unconditionally caches
non-folder columns from `.npz` at init anyway).

### 9. `FolderDataset` never overrides `get_dim()`
**Status: Fixed locally, same commit as #8**

Base `Dataset.get_dim` raises `NotImplementedError` unconditionally;
`prejepa.py`'s `dataset.get_dim(key)` call crashed. Fixed with the same
one-liner `hdf5.py` already uses.

### 10. `VideoDataset.__init__`'s `folder_keys` default defeats its own subdirectory auto-detection
**Status: Fixed locally, same commit as #8**

```python
folder_keys=video_keys or ['video']
```
Since `prejepa.py` never passes `video_keys`, this always fell back to
looking for a literal `video/` subfolder — defeating `FolderDataset`'s own
auto-detection of per-column subdirectories (our converted dataset stores
frames under `pixels/`/`goal/`, matching `VideoWriter`'s own docstring).
Fixed to `folder_keys=video_keys`, letting `None` fall through to
auto-detect.

### 11. `scripts/train/prejepa.py` calls a `stable_pretraining` callback that no longer exists
**Status: Fixed locally (dead code removed), same commit as #8**

`spt.callbacks.CPUOffloadCallback()` doesn't exist in the pinned
`stable-pretraining>=0.1.7` (checked latest PyPI release too), and isn't used
by any sibling training script (`cjepa.py`/`lewm.py`/`pldm.py`) — evidently
dead code left over from an earlier `stable_pretraining` version. Removed.

### 12. `PreJEPA.get_cost()`/`rollout()`'s planning cache KeyErrors on any offline (non-env) usage
**Status: Found 2026-07-10, worked around locally in a diagnostic script, not fixed in the library**

`get_cost()`/`rollout()` cache the goal/init embedding on the model instance
(`self._goal_cached_info`, `self._init_cached_info`), gated by
`hasattr(self, ...) and torch.equal(cached['id'], info_dict['id'][:, 0])`.
This is fine for real MPC rollouts, where the env wrapper always populates
`info_dict['id']`/`['step_idx']` and repeated `get_cost()` calls are for the
*same* env instance across CEM iterations. It breaks for any direct/offline
`get_cost()` usage that doesn't go through the env (e.g. an
action-sensitivity diagnostic bypassing CEM entirely, mirroring
`scripts/diagnose/action_sensitivity_check.py`'s existing CJEPA-side
methodology): the cache dict only ever contains whatever keys were present
in the caller's `info_dict`, so if `'id'`/`'step_idx'` were never supplied,
the *first* call succeeds (cache miss, `hasattr` is False) but the *second*
call raises `KeyError: 'id'` trying to read them back out of the
now-populated (but keyless) cache. Worked around by explicitly
`delattr`-ing both cache attributes before every independent call
(see `scripts/diagnose/action_sensitivity_check_dinowm.py`).

**Suggested fix**: use `.get('id')`/tolerate absent keys in the cache-hit
check, or make the caching explicitly opt-in rather than an implicit
hasattr-gated side effect on the model object (surprising for anyone calling
`get_cost()` directly, e.g. for diagnostics or tests).

---

## Config/docs footguns

### 13. `subdir` vs `output_model_name` split is a checkpoint-overwrite footgun
**Status: Not fixed upstream, worked around locally (explicit `output_model_name=...` override)**

`subdir` (default a random Hydra job id) *looks* like the run-naming knob and
is the more discoverable CLI override (`subdir=my_run`), but only controls
where a Hydra config snapshot gets written — a file nothing else reads. Actual
checkpoint location is controlled by the separate `output_model_name`, which
defaults to a fixed literal (e.g. `'cjepa'`) shared across every run that
doesn't override it. Hit directly this project: two dry runs and an initial
pass all silently landed in the same shared checkpoint directory.

**Suggested fix**: unify into one config key (or default `output_model_name`
to `subdir`'s value). Worth checking whether `lewm.py`/`prejepa.py`/other
`scripts/train/*.py` share the same pattern — they appear structurally
similar.

### 14. Shipped DataLoader defaults OOM-kill on a memory-capped container, on a clean process, on first attempt
**Status: Not fixed upstream, worked around locally (lower `num_workers`/`prefetch_factor`, disable `persistent_workers`)**

`num_workers=6`, `persistent_workers=True`, `prefetch_factor=3` (shipped in
`cjepa.yaml` and likely sibling configs) SIGKILL'd a completely fresh
training process (no leaked workers, no prior crash) on a ~29GB cgroup-capped
dev container, before a single training step completed. Reproduced
independently on two different fresh pods. Reducing to `num_workers=2`,
`prefetch_factor=2`, `persistent_workers=False` fixed it immediately.

**Suggested fix**: lower the shipped defaults, or at least document the
memory tradeoff next to the config keys.

### 15. Eval wall-clock cost is dramatically underestimated by this project's own cost table, likely also `docs/baselines.md`
**Status: Not a code bug — documentation/expectation-setting gap**

A 50-episode MPC eval (`eval_budget=50`) was assumed to cost "~15 min" in
this project's own cost-estimate table. Measured: **~9-10 hours** on an RTX
2000 Ada, **~65-70 min** even on an RTX 5090. Root cause isn't the GPU —
`solver/cem.yaml`'s `batch_size: 1` means CEM solves episodes strictly
serially regardless of how many envs are vectorized, so cost scales
~linearly with `num_eval × eval_budget`, not amortized across a batch.
`docs/baselines.md` documents the 50-step eval budget target but not the
expected wall-clock cost to run it, which would have caught this
assumption earlier.

**Suggested fix**: a cost/throughput callout near `docs/baselines.md`'s eval
protocol description or the `cem.yaml` solver config's comments, e.g.
"cost scales linearly with `num_eval × eval_budget` since CEM batch_size
defaults to 1 (serial)."

---

## DX / observability ideas (not bugs)

### 16. CEM solver has no progress indicator during `solve()`
**Status: Idea**

For a run that can take over an hour (see #15), the only output during the
entire solve is silence, followed by a single `CEM solve time: X seconds`
line at the very end — no way to tell if it's hung, how far through it is, or
estimate remaining time without doing a manual extrapolation from a smaller
smoke run (which is what this project did, twice, on two different GPUs).
Training scripts get this for free via Lightning's own progress bar; the
solver has no equivalent.

**Suggested fix**: a `tqdm`-style progress bar over the
`for start_idx in range(0, total_envs, self.batch_size)` batch loop and/or
the inner `for step in range(self.n_steps)` CEM iteration loop in
`stable_worldmodel/solver/cem.py` — low-risk, pure addition, no behavior
change. Probably the easiest item on this list to turn into a first PR.

### 17. `solver.batch_size` isn't safe to increase for speed without documented guidance
**Status: Idea (found empirically 2026-07-10)**

Tried raising `solver.batch_size` from the shipped default `1` to `4` on a
32GB RTX 5090 (in response to #15's cost problem) — immediately OOM'd trying
to allocate ~7GB more with only ~1GB free. `num_samples: 300` interacting
with `batch_size` scales memory sharply; there's no documented guidance for
picking a safe value for a given model/GPU, so the one knob that exists for
addressing #15's cost problem is currently unusable without manual
trial-and-error against OOM.

**Suggested fix**: document the actual memory scaling near the config (or in
`CEMSolver`'s docstring), and/or add a memory-aware auto-batch-size helper
(e.g. probe available VRAM and back off on OOM automatically, similar to
patterns used for auto-batch-size training loops elsewhere).

---

## Not `stable-worldmodel` bugs (repo/infra-side, listed for completeness only)

- **swm-runpod's Dockerfile torch/CUDA pin has to track GPU generation**
  (`cu126` for 4090/2000-Ada-class cards, `cu128`+ for 50-series/Blackwell/
  `sm_120`) — infra issue in this fork's dev image, not the library. Updated
  locally 2026-07-10 for the 5090 (Dockerfile diff staged, not yet pushed —
  see `CJEPA_PROJECT.md`'s session log for the 2026-07-03/04 precedent where
  the same class of issue was hit for the 4090 and fixed permanently).
- **No script in the upstream repo produces `pusht_expert_train.lance`** under
  the exact name every PushT training config expects. Filled locally with
  `scripts/data/collect_pusht_expert_train.py` (2026-07-04) — worth
  considering as a canonical upstream PushT expert-data collection script,
  since this project independently needed to write it from scratch and
  reuses it every fresh pod (data isn't persisted to S3 by design, so it's
  re-run often).

---

## Suggested PR grouping

Roughly in order of how easy/isolated they'd be to land upstream:

1. **CEM progress bar** (#16) — pure addition, zero behavior change, good
   first PR to test the contribution waters.
2. **Lance-format `eval_wm.py`/`eval_ff.py` fixes** (#5, #6, #7) — small,
   isolated, easy to review in one PR.
3. **Video-format `FolderDataset`/`VideoDataset` fixes + dead code removal**
   (#8, #9, #10, #11) — already committed together locally (`7fe49eb`),
   ready to upstream as-is.
4. **Config/docs fixes** (#13, #14, #15) — low-risk, low effort, mostly
   default-value and comment changes.
5. **`get_cost()` offline-usage cache bug** (#12) — small fix, but worth a
   test covering the non-env call path so it doesn't regress.
6. **History-aware `WorldModelPolicy` generalization** (#1) — more invasive,
   touches the policy/model interface, probably worth a design discussion
   with maintainers before a PR.
7. **CJEPA correctness fixes** (#2, #3, #4) — these live in code this project
   added (`wm/cjepa/`), not existing upstream code, so they're really part of
   the eventual CJEPA contribution itself rather than a standalone bugfix PR.
