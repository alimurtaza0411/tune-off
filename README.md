# tune-off: Is Mamba Harder to Tune?

**Mamba vs. Transformers under equal, paired tuning budgets, across tasks, data sizes and model sizes.**
COMS 4776 Neural Networks and Deep Learning, Columbia University, Fall 2026 · Mentor: Todd Morrill
Team: Alimurtaza Mustafa Merchant · Rajvardhan Patil · Pranav Prabhat Sinha

> **Status: v0.1.0 (first lightweight version).** The engine runs end to end on CPU (`python -m mvt smoke`).
> The three real tasks (`sst2`, `conll2003`, `wikitext2`) and the analysis metrics are still to be written by
> their owners, against the contracts below.

---

## 1. Overview

Mamba, a selective state-space model, is the main alternative to the Transformer. At large scale it matches strong
Transformer baselines on language modelling, but head-to-head comparisons usually train each architecture with a
single recipe. Tuning effort alone can change which architecture looks better, and a 2025 study found that Mamba
trains well only inside a narrow learning-rate window, though only on synthetic tasks.

We train small Mamba and LLaMA-style Transformer models from scratch on three real NLP tasks, at two training-data
sizes and two model sizes. Both models get **identical, paired tuning budgets**. We measure who wins as tuning effort
grows, how forgiving each architecture is to its hyperparameters, and which hyperparameters explain the difference.

**How it works, in three lines**
1. Every cell (task × data fraction × model size) trains both architectures on the **same 32 Sobol hyperparameter points with the same seeds**, then 24 TPE trials, then 5+5 test replicates.
2. Each trial writes one JSON result; `mvt export` turns a cell into one CSV.
3. The analysis computes expected-best curves, robustness, fANOVA importance and tunability from those CSVs.

## 2. Research questions and hypotheses

| ID | Question |
|---|---|
| **RQ1** | Under equal, paired tuning budgets, which architecture performs better on each task, and does the answer change with the tuning budget *n*? |
| **RQ2** | Is Mamba less forgiving to hyperparameter choice than a LLaMA-style Transformer, and how much of its sensitivity is explained by the learning rate? |
| **RQ3** | How do data scarcity (25% vs. 100%) and model size (≈2M vs. ≈8M non-embedding parameters) change the answers? |
| **RQ4** | Do the conclusions hold under both random (Sobol) and Bayesian (TPE) search, and do good configurations transfer across tasks and sizes? |

**Pre-registered hypotheses** (support rules fixed before any experiment runs):
- **C1 (sensitivity).** For each task, fraction and scale: Robust_Mamba < Robust_Transformer **and** Imp_Mamba(lr) > Imp_Transformer(lr). *Supported* if the paired-bootstrap 95% CI of each difference excludes 0 in the predicted direction.
- **C2 (data scarcity).** Gap(n) = Best_T(n) − Best_M(n) is larger with 100% data than with 25%. *Supported* if the 95% CI of Gap_100%(n) − Gap_25%(n) lies above 0 at every n ∈ {1, 2, 4, 8, 16, 32}.
- **E1 (model size, exploratory).** How the robustness gap and LR importance change from 2M to 8M. No directional prediction.

A refuted hypothesis is a valid result and is reported as such.

**Scope.** We claim results for ≈2M/≈8M-parameter models trained from scratch on three tasks, Mamba-1 vs. a
LLaMA-style Transformer, under equal budgets. We do **not** claim anything about billion-parameter models, Mamba's
speed or long-sequence efficiency, Mamba-2, hybrids, pretrained models, or which architecture is "better" overall.

**Grid.** 2 architectures × 3 tasks (`sst2`, `conll2003`, `wikitext2`) × 2 fractions (0.25, 1.0) × 2 scales
(`2m`, `8m`) = **12 cells**, each run for both architectures. Per (cell, architecture): 32 Sobol + 24 TPE + 5 best +
5 default = **66 runs**; **1,584 runs** total. All `2m` cells run first as a rehearsal.

## 3. Quickstart

```bash
# CPU / laptop
pip install -e ".[dev,analysis]"
python -m mvt info                 # environment report
python -m mvt smoke                # end-to-end CPU check, < 3 minutes
pytest -m "not gpu and not data"   # the CI test suite

# Colab / GPU: see notebooks/colab_runner.ipynb (installs the [cuda,tasks] extras)

# Run one cell (resumable: just run it again after a disconnect)
export MVT_OWNER=<your-name>
python -m mvt run --task sst2 --fraction 0.25 --scale 2m
python -m mvt status
python -m mvt export --task sst2   # -> results/sst2-025-2m.csv ...
python -m mvt validate             # checks results/*.csv
```

## 4. Architecture overview

```
configs/*.yaml ──> core.config ──> ResolvedConfig (+ config_hash)
                                        │
        registry ──> Task.build(fraction) ──> TaskData (examples)
                                        │
             data.batching (shared) ──> batches  ──>  models.build_model ──> SequenceModel
                                        │
                     training.trainer ──> TrainOutput ──> search.trial ──> TrialResult JSON
                                        │
     search.runner (Sobol / TPE / replicates, resumable) ──> mvt export ──> results/{cell}.csv
                                        │
                                  analysis.io ──> metrics and figures
```

| Module | Responsibility |
|---|---|
| `mvt/__main__.py` | The CLI |
| `core/config.py` | Pydantic models for every YAML, `load_resolved_config`, `config_hash` |
| `core/registry.py` | `register_task`, `register_model`, auto-discovery |
| `core/types.py` | `TaskSpec`, examples, `TaskData`, `HParams`, `TrialRequest` |
| `core/seeding.py` | `derive_seed`, `seed_everything` |
| `core/env.py` | git state, versions, device, Mamba kernel detection |
| `core/results.py` | `TrialResult` schema v1, atomic write, export, validation |
| `data/` | `Vocab`, batching (bucketed, right-padded), subsampling, cache |
| `tasks/base.py`, `_template.py`, `debug.py` | Task contract, a commented template, synthetic tasks |
| `models/` | Transformer, Mamba (`cuda` + pure-PyTorch `ref`, BiMamba), shared head |
| `training/` | AdamW groups, warmup-cosine schedule, the training loop |
| `search/` | Search space, Sobol table, TPE, single trial, cell runner, pilot |
| `analysis/io.py` | Load CSVs with the inclusion and floor rules |

## 5. Design decisions with rationale

### 5.1 Models (both architectures)
- Trained from scratch. **Embedding → dropout → N pre-norm residual layers → final RMSNorm → task head.**
- Residual layer: `x = x + Dropout(Mixer(RMSNorm(x)))`. One tuned dropout rate, applied after the embedding and on every residual branch. RMSNorm ε = 1e-5.
- Heads: `seq_cls` masked mean pooling + linear; `tok_cls` per-token linear; `lm` tied to the embedding.
- **Transformer (LLaMA-style):** RoPE (base 10,000), no biases, SwiGLU MLP, head dimension 64, SDPA attention (causal for `lm`, key-padding mask otherwise). Init N(0, 0.02), residual output projections scaled by 1/√(2L). *Rationale:* the strongest, most stable modern baseline; a fragile baseline would exaggerate differences.
- **Mamba-1:** `mamba_ssm.Mamba` v1 mixer (d_state 16, d_conv 4, expand 2, dt_rank ⌈d/16⌉), no separate MLP, no positional encoding. Mixer internals keep the `mamba_ssm` init; embedding and linear layers N(0, 0.02); `out_proj` scaled by 1/√L.
  - `impl: cuda` uses the fused kernels; `impl: ref` is a pure-PyTorch mixer with identical parameter names, shapes and init (CPU, tests, fallback; valid scores but not valid timings). `auto` picks `cuda` when possible. The implementation is recorded in every result.
  - **BiMamba** for `seq_cls`/`tok_cls`: `Mixer(x) + Reverse(Mixer(Reverse(x)))` with **shared weights** (as a Transformer encoder uses the same weights both ways; keeps parameter counts equal).
  - **Padding:** `mamba_ssm` has no padding mask. Batches are right-padded, pads are zeroed before each mixer, and `Reverse` is length-aware (real tokens reversed, pads stay at the end), so pads never influence real positions. A contract test enforces this.
- **Parameter counting:** "non-embedding" = all trainable parameters except the token embedding (counted once when tied) and the task head.

### 5.2 Scales (width only; depth fixed: Transformer 10 layers, Mamba 18)

| Scale | d_model | Transformer layers / heads / mlp_hidden | Transformer non-emb. | Mamba layers | Mamba non-emb. | Diff. |
|---|---|---|---|---|---|---|
| `2m` | 128 | 10 / 2 / 384 | 2,132,608 | 18 | 2,099,072 | 1.6% |
| `8m` | 256 | 10 / 4 / 704 | 8,033,536 | 18 | 7,884,544 | 1.9% |
| `4m` (ext.) | 192 | 10 / 3 / 512 | 4,427,712 | 18 | 4,531,008 | 2.3% |
| `30m` (ext.) | 512 | 10 / 8 / 1344 | 31,140,352 | 18 | 30,514,688 | 2.0% |

Per layer: Transformer = 4d² + 3·d·h + 2d. Mamba (D = 2d, R = ⌈d/16⌉, N = 16) = 4d² + 5D + D(R + 2N) + (RD + D) + DN + D + Dd + d. Both stacks add a final norm (d). The counts are checked exactly by tests.

### 5.3 Training recipe (fixed for every run)

| Setting | Value |
|---|---|
| Optimiser | AdamW, β = (0.9, 0.95), eps 1e-8, fused on CUDA |
| Weight decay | On ndim ≥ 2 parameters except `_no_weight_decay` (Mamba `A_log`, `D`); never biases or norm gains |
| Schedule | Linear warmup over `warmup_fraction × total_steps`, cosine decay to 10% of peak |
| Precision | fp16 autocast + GradScaler on CUDA, fp32 on CPU. If a cell shows precision NaNs, **both** architectures move to fp32 |
| Clipping | Global norm at the tuned value after unscaling; 0.0 = off |
| Length | A fixed number of **epochs per (task, fraction)**, calibrated once by the pilot |
| Steps | `total_steps = epochs × ceil(n_train_units / batch_size)`, run exactly |
| Evaluation | 20 evenly spaced validation points; the run's score is the **best** of them |
| Test | Only in `best`/`default` replicates, from the best-validation checkpoint |
| Not used | `torch.compile`, gradient accumulation, early stopping |

**Why epochs calibrated to convergence.** Too short rewards whichever architecture learns faster early (a different
question); too long wastes most of the cosine schedule after the peak. The pilot picks epochs so that good
configurations of **both** architectures peak between 30% and 80% of training. Within a (task, fraction), every
configuration, both architectures and both scales use the same epochs. Adequacy is reported (share of trials still
improving at the last evaluation), not assumed. Initial guesses: SST-2 8 / 16 epochs, CoNLL 25 / 50, WikiText-2 8 / 16
(100% / 25%).

### 5.4 Hyperparameter search
Six training hyperparameters, **identical space for both architectures**:

| # | Hyperparameter | Range / choices | Default λ₀ |
|---|---|---|---|
| 0 | `learning_rate` | log [1e-5, 1e-2] | 1e-3 |
| 1 | `warmup_fraction` | [0, 0.1] | 0.05 |
| 2 | `weight_decay` | log [1e-5, 0.3] | 0.1 |
| 3 | `dropout` | [0, 0.5] | 0.1 |
| 4 | `batch_size` | choice index 0–3: SST-2 {32, 64, 128, 256}; CoNLL {16, 32, 64, 128}; WikiText-2 {8, 16, 32, 64} × 256 tokens | index 1 |
| 5 | `grad_clip` | {0.25, 0.5, 1.0, 0.0} | 1.0 |

- **Paired Sobol design.** One committed table `configs/sobol_32.csv` (scrambled Sobol, seed 4776, u-values only). Every (cell, architecture) uses the same 32 points with the same seeds, so each pair of trials differs only in architecture. 32 = 2⁵ points put exactly 8 points on each categorical choice; the first 16 are balanced too.
- **Phases:** `sobol` 32 (all primary metrics) → `tpe` 24 (Optuna TPE seeded with all Sobol results, rebuilt from saved results for each suggestion; deterministic and resumable; no Optuna database) → `best` 5 (λ* = best floored validation score over sobol ∪ tpe, ties to the earlier phase then lower index) → `default` 5 (λ₀). Architectures are interleaved inside each phase.

### 5.5 Data rules
- Tasks produce examples; the core batches them, so every task is batched identically.
- Word-level tokens; `<pad>` = 0, `<unk>` = 1. **Vocabulary from the full training pool**, independent of fraction.
- Fractions are a fixed random subset (task `subset_seed`), identical for every trial, architecture and scale. **Validation and test never depend on the fraction.**
- Truncation to `max_len` is counted (must be zero on val/test). Processed data is cached on local disk by fingerprint.
- `sst2`: official train → 90/10 train pool / validation; test = official validation (872). Accuracy, floor 0.5, ε 0.01.
- `conll2003`: case preserved, 9 IOB2 tags, entity-level micro-F1 (`seqeval`), floor 0, ε 0.01.
- `wikitext2`: `wikitext-2-v1`, lines + `<eos>`, contiguous 257-token chunks. Score = −mean NLL; floor = −ln V; ε = ln 1.05.

### 5.6 Scoring
Every task reports a **score where higher is better** plus the raw metric. In analysis every score is
`max(score, floor)`. Status `ok`; `diverged` (non-finite training loss: scored at the floor and **always** included);
`failed` (infrastructure error: retried once, excluded, must be re-run). GradScaler-skipped steps are not divergence.

### 5.7 Seeds and provenance
`derive_seed(*parts) = int(sha256("|".join(parts))[:8], 16)`. Trial seeds **exclude the architecture** (the pairing):
Sobol `derive_seed(cell, "sobol", i)`, TPE `derive_seed(cell, "tpe", k)`, replicates `derive_seed(cell, "replicate", r)`.
Every result records the git commit and dirty flag, code version, config hash, Mamba implementation, versions, GPU,
owner and timestamps. Real phases refuse to run from a dirty tree or without `MVT_OWNER`.

### 5.8 Pilot and code freeze
`mvt pilot --task T`: an 8m learning-rate sweep (7 log-spaced values, λ₀ otherwise) for both fractions and
architectures, then a 2m check at each architecture's best LR. Pre-registered rules, applied once:
- **LR range:** if a best sweep value is at an edge, widen that end by one decade for all tasks.
- **Length:** if any good run peaks in the last 20%, double the epochs (at most twice); else if all peak in the first 30%, halve (once); prefer longer on conflict.
- **Sanity:** λ₀ or the best sweep run must clearly beat the floor for both architectures.

Then tag **v1.0.0**; every real result must come from v1.0.0 or an approved bug-fix version.

### 5.9 Compute and extensions
Roughly 50–90 T4 GPU-hours for the grid (WikiText-2 ≈ 40%). If behind, cut in this order: TPE at 2m; replicates
5 → 3; TPE 24 → 16 at 8m; Sobol 32 → 16 at 2m. **Never** cut a task, a fraction, the 8m Sobol phase or the calibrated
length. Scales `4m` and `30m` exist in `configs/scales.yaml` as optional extensions.

## 6. Contracts

**Task** (`tasks/base.py`): `build(fraction) -> TaskData` (deterministic; val/test independent of fraction; vocab
from the full pool; subset with `subset_seed`) and `compute_metric(predictions, references) -> (score, raw_metric)`.
Tasks register with `@register_task("name")` in their own file. Never edit a central list.

**Batch** (`data/batching.py`): `input_ids` [B, L] right-padded with 0; `attention_mask` [B, L] bool, real tokens a
prefix; `labels` [B] (`seq_cls`), [B, L] with −100 at pads (`tok_cls`), [B, L] next-token targets (`lm`); `n_tokens`.
Batches are length-bucketed inside pools of 50 batches, shuffled per epoch with the trial seed.

**Model** (`models/`): `build_model(arch, scale, model_cfg, spec, dropout) -> SequenceModel`;
`forward(input_ids, attention_mask)` returns [B, C], [B, L, C] or [B, L, V]; `param_counts()`. Pads never influence
real positions; `lm` models are causal. A new architecture is a new registered file.

**Trial** (`search/trial.py`): `run_trial(TrialRequest, ResolvedConfig, owner) -> TrialResult`. Never raises on
divergence; raises only on infrastructure errors (the runner records them as `failed` after one retry).

**Results** (`core/results.py`, schema v1): one JSON per trial at
`{MVT_RESULTS_ROOT}/{cell_id}/{arch}/{phase}/{index:03d}.json`, `cell_id = {task}-{percent:03d}-{scale}`
(e.g. `sst2-025-8m`). Written atomically, never overwritten without force. Fields: identity, `hparams`, `model`,
`status`, `val_score`/`val_metric`, `test_score`/`test_metric`, `floor`, `epsilon`, curves, steps, timings, memory,
parameter counts, precision, device, provenance. `mvt export` flattens to `results/{cell_id}.csv` (`hp_*`, `model_*`
columns). `mvt validate` checks schema, one config hash and code-version family per cell, complete counts
(32/24/5/5), no duplicates, identical Sobol hyperparameters and seeds across architectures, and zero failed trials.

**Config** (`core/config.py`): every experimental choice lives in YAML. `config_hash` = first 12 hex chars of SHA-256
over the canonical resolved config, excluding runtime fields (`results_root`, `data_cache_dir`, `num_workers`,
`device`). Environment: `MVT_OWNER` (required for real phases), `MVT_RESULTS_ROOT` (default `./runs`),
`MVT_DATA_CACHE` (default `./.cache/mvt`), `MVT_DEVICE` (default `auto`).

**CLI** (`python -m mvt`): `info`, `smoke`, `sobol --write|--check`, `pilot --task T`,
`run --task T --fraction F --scale S [--arch] [--phase all|sobol|tpe|replicates] [--limit N] [--retry-failed]`,
`status`, `export --task T|--cell C|--all`, `validate`.

**Changing a contract:** a GitHub issue labelled `contract-change`, agreement from all three members, implementation
by the skeleton owner (additive where possible), a version bump with a note in the release description.

## 7. Adding a task
1. Copy `src/mvt/tasks/_template.py` to `src/mvt/tasks/<name>.py` and uncomment `@register_task("<name>")`.
2. Fill `configs/tasks/<name>.yaml` (already drafted for the three real tasks) and **pin `dataset.revision`**.
3. Implement `build` following the rules in the template. Override `compute_metric` only if needed.
4. Add `tests/data/test_<name>.py` (marker `data`) asserting the expected split sizes and vocabulary size.
5. `pytest tests/contract -m "data or not data"` must pass: the task contract tests run for every registered task.
6. Run `python -m mvt pilot --task <name>` and report the outcome.

## 8. Adding a model (extension)
Add `src/mvt/models/<name>.py` with a `SequenceModel` subclass decorated `@register_model("<name>")`, a
`configs/models/<name>.yaml`, a block in `configs/scales.yaml`, and make `tests/contract/test_models.py` pass.

## 9. Running experiments
- Set `MVT_OWNER`, `MVT_RESULTS_ROOT` (on Colab, a Google Drive folder) and `MVT_DATA_CACHE` (on Colab, local disk).
- Use `notebooks/colab_runner.ipynb`. It is a thin launcher. Re-running it after a disconnect resumes the cell.
- A cell is **done** when `mvt status` shows 66/66 per architecture, `mvt validate` passes and the CSV is merged.
- Commit **only your own cells' CSVs**, via a pull request. Never commit raw run folders, data or checkpoints.

## 10. Analysis contract
- Inputs: `results/*.csv` only. Phases `sobol`, `tpe`, `best`, `default`. Statuses `ok` and `diverged` (failed rows are dropped and counted). Score `s = floor` if diverged, else `max(val_score, floor)`. `mvt.analysis.io.load_results` applies these rules.
- **Expected best:** Best(n) = Σᵢ s₍ᵢ₎[(i/N)ⁿ − ((i−1)/N)ⁿ], scores ascending, N = 32 (Dodge et al., 2019).
- **Gap** Gap(n) = Best_T(n) − Best_M(n). **Flip budget** n* = smallest n where Gap's 95% CI lies entirely on the opposite side of 0 from some smaller n.
- **GPU-time axis:** t(n) = n × mean Sobol `wall_time_s` (only for `mamba_impl = cuda`).
- **Robustness:** share of Sobol trials with s ≥ max(s) − ε (also at ε/2, 2ε, and on the 16-point LR sub-range u_lr ∈ [0.25, 0.75]).
- **Importance:** fANOVA main effects (Optuna `FanovaImportanceEvaluator`, fixed seed) on the 32 Sobol trials.
- **Tunability:** mean test over `best` − mean over `default`, paired by replicate seed. **Final performance:** test mean ± sd of λ*.
- **Search-strategy check, training-length adequacy, transfer** (Spearman ρ of the 32 Sobol scores across tasks, scales and fractions).
- **Uncertainty:** paired bootstrap over Sobol indices, B = 2,000 (200 for fANOVA), percentile 95% CI, seed 0.
- **Reference test vectors:** scores [70, 80, 85, 90] → Best(1) = 81.25, Best(2) = 85.3125, Best(4) = 88.06640625; Robust with ε = 6 → 0.5.

## 11. Troubleshooting
- **`mamba_ssm` import fails ("undefined symbol")**: the wheel does not match your PyTorch/CUDA. Install the PyTorch version the wheel was built for, or build from source with `pip install . --no-build-isolation`.
- **No CUDA Mamba:** `impl: auto` falls back to the `ref` mixer. Scores stay valid; timings do not.
- **fp16 NaNs:** if a cell shows precision-caused NaNs, switch `precision: fp32` for **both** architectures in that cell and record it.
- **Colab disconnects:** re-run the notebook; existing results are skipped. Keep the data cache on `/content`, results on Drive.

## 12. References
Gu & Dao (2023), *Mamba*, arXiv:2312.00752 · Waleffe et al. (2024), arXiv:2406.07887 · Okpekpe & Orvieto (2025),
arXiv:2508.19029 · Seo et al. (2026), arXiv:2610.01172 · Jelassi et al. (2024), arXiv:2402.01032 · Arora et al. (2023),
arXiv:2312.04927 · Melis, Dyer & Blunsom (2018), arXiv:1707.05589 · Dodge et al. (2019), arXiv:1909.03004 ·
Touvron et al. (2023), arXiv:2302.13971 · Su et al. (2021), arXiv:2104.09864 · Shazeer (2020), arXiv:2002.05202 ·
Zhang & Sennrich (2019), arXiv:1910.07467 · Zhu et al. (2024), arXiv:2401.09417 · Schiff et al. (2024),
arXiv:2403.03234 · Hutter, Hoos & Leyton-Brown (2014) · Bergstra et al. (2011) · Bergstra & Bengio (2012) ·
Akiba et al. (2019), arXiv:1907.10902 · Probst, Boulesteix & Bischl (2019), arXiv:1802.09596 · Socher et al. (2013) ·
Tjong Kim Sang & De Meulder (2003) · Merity et al. (2017), arXiv:1609.07843 · Nakayama (2018), seqeval.
