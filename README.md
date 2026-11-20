# tune-off: Is Mamba Harder to Tune?

A controlled comparison of **Mamba** and a **LLaMA-style Transformer** in which both models get exactly the same
hyperparameter-tuning budget.

COMS 4776 Neural Networks and Deep Learning, Columbia University, Fall 2026 · Mentor: Todd Morrill
Team: Alimurtaza Mustafa Merchant · Rajvardhan Patil · Pranav Prabhat Sinha

> **Status (v0.1.0):** the full pipeline runs end to end on CPU with synthetic debug tasks.
> Still to do: the three real tasks (`sst2`, `conll2003`, `wikitext2`) and the analysis metrics.

**Contents:** [Why](#why-this-project) · [The experiment](#the-experiment) · [Getting started](#getting-started) ·
[Running experiments](#running-experiments) · [Code map](#how-the-code-is-organised) ·
[Design decisions](#design-decisions) · [Rules for contributors](#rules-every-contributor-must-keep) ·
[Adding a task](#adding-a-task) · [Analysis](#analysis) · [Troubleshooting](#troubleshooting) · [Glossary](#glossary)

---

## Why this project

Mamba is the main alternative to the Transformer, and at large scale it matches strong Transformers on language
modelling. But most head-to-head comparisons train each model with **one fixed recipe**. That hides a confounder:
one model may simply have been easier to tune. Re-tuning has reversed published rankings before, and a 2025 study
found that Mamba trains well only inside a narrow learning-rate window. That study used synthetic tasks only.

We test this on **real NLP tasks**. Both architectures get identical, paired tuning budgets, and we ask:

| | Question |
|---|---|
| **RQ1** | Which architecture wins on each task, and does the winner change as the tuning budget grows? |
| **RQ2** | Is Mamba less forgiving of hyperparameter choices, and is the learning rate the reason? |
| **RQ3** | How do less training data (25% vs. 100%) and a bigger model (≈2M vs. ≈8M parameters) change the answers? |
| **RQ4** | Do the conclusions survive a different search method (Sobol vs. TPE), and do good settings transfer between tasks? |

**Hypotheses, fixed before any run** (a refuted hypothesis is still a result):
- **C1, sensitivity:** Mamba is less forgiving (lower robustness), and the learning rate explains more of its score variance. Supported where the paired-bootstrap 95% confidence interval of each difference excludes zero in that direction.
- **C2, data scarcity:** the Transformer's lead is larger with 100% data than with 25%. Supported where the confidence interval of that difference is above zero at every budget n ∈ {1, 2, 4, 8, 16, 32}.
- **E1, model size (exploratory):** how robustness and learning-rate importance change from 2M to 8M parameters. No predicted direction.

**From per-setting verdicts to a headline claim** (also fixed in advance). C1 is tested in 12 settings (3 tasks ×
2 data sizes × 2 model sizes) and C2 in 6 (3 tasks × 2 model sizes). Every per-setting verdict is reported in one table.

| Headline verdict | Rule |
|---|---|
| **Supported overall** | Supported in a majority of settings (C1: at least 7 of 12; C2: at least 4 of 6), and significantly contradicted in none |
| **Refuted overall** | The opposite direction is significant in a majority of settings |
| **Mixed** | Anything else, described task by task |
| **Supported for one task** | Supported in every setting of that task (C1: 4 of 4; C2: 2 of 2) |

For C1, "contradicted" means either part, robustness or learning-rate importance, is significant in the opposite
direction. Each setting uses its own 95% interval with no multiple-comparison correction. The majority rule keeps one
or two chance results from deciding the headline. E1 is exploratory: its direction and interval are reported per
task and data size, with no headline claim.

**What we do not claim:** anything about billion-parameter models, Mamba's speed or long-sequence efficiency
(our inputs are short), Mamba-2, hybrids, pretrained models, or that either architecture is "better" overall.

## The experiment

### The grid

| Axis | Values |
|---|---|
| Architecture | `transformer` (LLaMA-style) · `mamba` (Mamba-1) |
| Task | `sst2` sentiment (accuracy) · `conll2003` named entities (entity F1) · `wikitext2` language modelling (perplexity) |
| Training data | 25% · 100% |
| Model size | `2m` (≈2M non-embedding parameters) · `8m` (≈8M) |

A **cell** is one task × data size × model size, which gives **12 cells**. Every cell is run for both architectures.

### What happens inside one cell

| Phase | Runs per architecture | What it does | Used for |
|---|---|---|---|
| `sobol` | 32 | Trains 32 fixed hyperparameter settings | **All main measurements** |
| `tpe` | 24 | Optuna's TPE studies the Sobol results and proposes better settings | Finding the best setting; checking RQ4 |
| `best` | 5 | Retrains the best setting found with 5 seeds, scored on test | Final test performance |
| `default` | 5 | Retrains the shared default setting with the same 5 seeds | How much tuning helps |

That is 66 runs per architecture per cell, and **1,584 runs** in total. All `2m` cells run first as a cheap rehearsal.

**Pairing is the core idea.** Sobol trial *i* uses identical hyperparameters **and** an identical random seed for
both architectures. The architecture is then the only difference within each pair. The 32 Sobol settings come from
one committed table, [`configs/sobol_32.csv`](configs/sobol_32.csv), reused in every cell. This makes results
comparable across tasks, data sizes and model sizes.

## Getting started

Requires Python ≥ 3.10.

```bash
git clone <repo-url> tune-off && cd tune-off
pip install -e ".[dev,analysis]"     # CPU is enough for development

python -m mvt info                    # environment report (GPU, versions, Mamba kernel)
python -m mvt smoke                   # full pipeline on synthetic tasks, about 1 minute on CPU
pytest                                # the CPU test suite CI runs (GPU and dataset tests are opt-in)
```

The `[tasks]` extra installs `datasets` and `seqeval` for the real tasks.

**On a GPU (Colab):** use [`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb). It installs the package and
then compiles the Mamba CUDA kernels against Colab's PyTorch:

```bash
pip install -e ".[tasks]"
pip install packaging ninja wheel
pip install --no-build-isolation -r requirements/colab.txt   # pinned mamba-ssm and causal-conv1d
```

Version pins: Optuna is pinned exactly, because TPE suggestions change between versions, and SciPy is pinned to a
range. The Mamba pins in `requirements/colab.txt` are confirmed by the first Colab session, whose full `pip freeze`
is then committed as `requirements/colab-freeze.txt`. Every result also records the PyTorch, `mamba_ssm` and Optuna
versions it ran with.

## Running experiments

**1. Set three environment variables:**

| Variable | Meaning | On Colab |
|---|---|---|
| `MVT_OWNER` | Your name, recorded in every result (required for real runs) | your name |
| `MVT_RESULTS_ROOT` | Where result files go (default `./runs`) | a Google Drive folder |
| `MVT_DATA_CACHE` | Where processed datasets are cached (default `./.cache/mvt`) | `/content/...` (local disk, not Drive) |

**2. Run a cell, check progress, export:**

```bash
python -m mvt run --task sst2 --fraction 0.25 --scale 2m   # all phases, both architectures
python -m mvt status                                       # progress per cell / architecture / phase
python -m mvt export --task sst2                           # writes results/sst2-025-2m.csv, ...
python -m mvt validate                                     # checks results/*.csv
```

Useful options for `run`: `--phase sobol|tpe|replicates`, `--arch transformer|mamba`, `--limit N`
(stop after N trials) and `--retry-failed`.

**Interruptions are safe.** Each finished trial is saved immediately as its own file. If a run stops for any reason,
such as Ctrl-C or a Colab disconnect, run the same command again. Finished trials are skipped, and only the
interrupted trial restarts, with the same seed.

**Real runs need a clean checkout.** The `sobol`, `tpe`, `best` and `default` phases refuse to start if `src/`,
`configs/` or `pyproject.toml` contain modified **or untracked** files, or if `MVT_OWNER` is unset. This way, every
result can be traced to an exact commit. Pilot runs are exempt, and their results never block a cell.

**Setup problems stop the run before any trial starts.** The task data is loaded and the Mamba kernels are checked
first. A missing dataset or a broken install is reported once and is never recorded as 66 failed trials.

**A cell is done** when `mvt status` shows 66/66 for both architectures, `mvt validate` passes, and the exported CSV
is merged through a pull request. Commit only your own cells' CSVs, never the raw run folders.

## How the code is organised

```
configs/*.yaml ─▶ resolved config ─▶ Task.build() ─▶ examples ─▶ shared batching ─▶ model ─▶ trainer
                                                                                              │
results/{cell}.csv ◀── mvt export ◀── one JSON file per trial ◀── runner (Sobol → TPE → replicates)
        │
        ▼
   analysis (metrics and figures)
```

| Folder | What lives there |
|---|---|
| `configs/` | **Every experimental setting.** Grid, recipe, search space, model sizes, one file per task, the Sobol table |
| `src/mvt/core/` | Config loading, seeds, shared data types, the result-file format, export and validation |
| `src/mvt/data/` | Vocabulary, batching and padding, data subsets, the dataset cache |
| `src/mvt/tasks/` | One file per task, plus `base.py`, a commented `_template.py` and synthetic `debug.py` tasks |
| `src/mvt/models/` | The Transformer, Mamba (fast CUDA and pure-PyTorch versions) and the shared model shell |
| `src/mvt/training/` | Optimiser, learning-rate schedule and the training loop |
| `src/mvt/search/` | Search space, Sobol table, TPE, running one trial, running a cell, the pilot |
| `src/mvt/analysis/` | Loading exported results; metrics and figures go here |
| `tests/` | `unit`, `contract`, `smoke` (CPU, in CI), `gpu` (Colab), `data` (real datasets) |
| `notebooks/` | The Colab launcher (no experiment logic) |

Entry point: `python -m mvt <command>`. Commands are `info`, `smoke`, `sobol --check`, `pilot`, `run`, `status`,
`export` and `validate`.

## Design decisions

Each choice below exists to keep the comparison fair. All values live in `configs/` and are fixed before experiments.

### Models

Both models share the same shell: token embedding → dropout → residual layers → final RMSNorm → task head. Each
layer is `x = x + Dropout(Mixer(RMSNorm(x)))`. Both use the same dropout placement and the same heads: mean pooling
for classification, a per-token layer for tagging, and an output layer tied to the embedding for language modelling.

- **Transformer:** LLaMA-style, with rotary position embeddings, a SwiGLU feed-forward block and no biases. This is the "Transformer++" baseline the Mamba paper itself uses. *Why:* it's the strongest, most stable modern baseline. A fragile baseline would exaggerate any difference.
- **Mamba:** Mamba-1 blocks from `mamba_ssm` (state size 16, convolution width 4, expansion 2), initialised exactly as `mamba_ssm` initialises its own models. Only the classification and tagging head shares the Transformer's initialisation. On GPU we use the fast CUDA kernels. A GPU without working kernels is an error, never a silent switch to the slow version. A pure-PyTorch copy with identical weights runs on CPU for tests; its scores are valid but its timings are not.
- **Classification and tagging read text in both directions.** The Transformer attends both ways. Mamba runs the same block forwards and backwards, with shared weights, and adds the two results. Language modelling is left-to-right for both.
- **Padding never leaks into results.** Mamba has no padding mask, so pads are zeroed and the backward pass reverses only the real tokens. A test checks that adding padding never changes outputs.

**Model sizes are matched.** Depth is fixed (Transformer 10 layers, Mamba 18) and only the width changes between
sizes. Counts exclude the embedding and the task head, because WikiText-2's 33k-word vocabulary would otherwise
dominate.

| Size | Width | Transformer parameters | Mamba parameters | Difference |
|---|---|---|---|---|
| `2m` | 128 | 2,132,608 | 2,099,072 | 1.6% |
| `8m` | 256 | 8,033,536 | 7,884,544 | 1.9% |

### Training recipe (identical for every run)

| Setting | Value |
|---|---|
| Optimiser | AdamW, β = (0.9, 0.95); no weight decay on biases, norms or Mamba's `A_log` / `D` |
| Schedule | Linear warmup, then cosine decay to 10% of the peak learning rate |
| Precision | fp16 mixed precision on GPU. If a cell hits fp16 NaNs, **both** architectures switch to fp32 |
| Length | A fixed number of epochs per task and data size, the same for every configuration, both architectures and both sizes |
| Evaluation | Validation at 20 evenly spaced points; a run's score is its **best** validation score |
| Test set | Used only in the `best` and `default` phases, from the best validation checkpoint |
| Not used | Early stopping, gradient accumulation, `torch.compile` |

**Training length is calibrated, not guessed.** If runs are too short, the faster starter wins, which is a different
question. If they are too long, most of the schedule is spent after the model has peaked. The pilot sets the epochs
so that good runs of both architectures peak between 30% and 80% of training. The analysis then reports how many runs
were still improving at the end.

### What gets tuned

Six training hyperparameters, with **the same search space for both architectures**:

| Hyperparameter | Range | Default |
|---|---|---|
| Learning rate | 1e-5 to 1e-2 (log scale) | 1e-3 |
| Warmup fraction | 0 to 0.1 | 0.05 |
| Weight decay | 1e-5 to 0.3 (log scale) | 0.1 |
| Dropout | 0 to 0.5 | 0.1 |
| Batch size | 4 choices per task: SST-2 {32, 64, 128, 256}, CoNLL {16, 32, 64, 128}, WikiText-2 {8, 16, 32, 64} × 256 tokens | 2nd choice |
| Gradient clipping | {0.25, 0.5, 1.0, off} | 1.0 |

Model shapes are fixed, not tuned. The default setting is one shared modern recipe, used as the "untuned" reference.
The Sobol table stores positions between 0 and 1 rather than actual values, so one row means the same relative
setting on every task. Its 32 points cover each range evenly, and each batch-size and clipping choice gets exactly
8 points.

### Data

- Tasks only produce examples. **All batching and padding is shared code**, so every task is treated identically.
- Word-level tokens. The vocabulary is built from the full training set, even when training on 25%.
- The 25% subset is a fixed random sample, the same for every run. Validation and test sets never change.

| Task | Splits | Metric (higher is better) | Random-guess floor | Robustness margin ε |
|---|---|---|---|---|
| `sst2` | 90/10 split of the official train set; test = official validation (872) | accuracy | 0.5 | 0.01 |
| `conll2003` | official train / validation / test | entity-level F1 (`seqeval`) | 0 | 0.01 |
| `wikitext2` | official splits, 256-token chunks | −log perplexity | −ln(vocab size) | ln 1.05 (≈ 5% perplexity) |

### Scores and failed runs

- Every score is "higher is better" and is never counted below random guessing (the floor).
- **Diverged** (the loss became NaN or infinite): a genuine finding about that setting. It is scored at the floor and always kept.
- **Failed** (out of memory, crash): an infrastructure problem. It is retried once, excluded from analysis and must be re-run.

### Pilot and code freeze

Before the real runs, each task owner runs `python -m mvt pilot --task <task>`. It sweeps 7 learning rates at `8m`
and checks the best one at `2m`. Three pre-registered rules then apply once:

1. If the best learning rate sits at the edge of the range, widen the range by 10× on that side, for all tasks.
2. If good runs are still improving in the last 20% of training, double the epochs. If they all peak in the first 30%, halve them.
3. If neither architecture clearly beats random guessing, there is a bug. Investigate before going further.

Then the code is tagged **v1.0.0** and frozen. All real results must come from that version or an approved bug fix.

### If compute runs short

Cut in this order: TPE at `2m` → replicates 5 → 3 → TPE 24 → 16 at `8m` → Sobol 32 → 16 at `2m`.
**Never** cut a task, a data size, the `8m` Sobol phase or the calibrated training length.

## Rules every contributor must keep

These are the interfaces everyone codes against. To change one, open a GitHub issue labelled `contract-change` and get
agreement from all three members. The skeleton owner then implements it, adding to it rather than breaking it.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.

1. **All experimental settings live in `configs/`**, never hard-coded in Python.
2. **Tasks and models register themselves** in their own file (`@register_task("name")`). There is no central list to edit.
3. **Tasks are deterministic.** Validation and test never depend on the data fraction, and the vocabulary comes from the full training set.
4. **Never edit result files by hand.** Results are written by the code and checked by `mvt validate`.

<details>
<summary><b>Exact interfaces: task, batch, model, results files</b></summary>

**Task** (`src/mvt/tasks/base.py`): `build(fraction) -> TaskData` returns a `TaskSpec` plus train, validation and test
example lists. `compute_metric(predictions, references) -> (score, raw_metric)`.

**Batch** (built only by `src/mvt/data/batching.py`): `input_ids` [B, L] right-padded with 0; `attention_mask` [B, L]
(True = real token); `labels` [B] for classification, [B, L] with −100 at pads for tagging, [B, L] next-token targets
for language modelling.

**Model** (`src/mvt/models/`): `forward(input_ids, attention_mask)` returns [B, classes], [B, L, tags] or
[B, L, vocab]. Padding must not affect real positions, and language models must be causal.

**Result files:** one JSON per trial at `{MVT_RESULTS_ROOT}/{cell_id}/{arch}/{phase}/{index:03d}.json`. The
`cell_id` looks like `sst2-025-8m` (task, data percent, size). Each file records the hyperparameters, scores,
validation curve, timings, memory, parameter counts and provenance: git commit, config hash, GPU and owner.
`mvt export` flattens one cell into `results/{cell_id}.csv`. `mvt validate` checks that both architectures are
present, complete counts (32/24/5/5), no duplicates, identical Sobol settings and seeds across architectures, one config version per cell, and no failed
trials.

</details>

## Adding a task

1. Copy `src/mvt/tasks/_template.py` to `src/mvt/tasks/<name>.py` and uncomment `@register_task("<name>")`.
2. Complete `configs/tasks/<name>.yaml` (already drafted for the three tasks) and **pin the dataset revision**.
3. Implement `build()` by following the comments in the template.
4. Add `tests/data/test_<name>.py` asserting the expected split sizes and vocabulary size.
5. Make the contract tests pass: `pytest tests/contract -m "data or not data"`. They run automatically for every registered task.
6. Run the pilot: `python -m mvt pilot --task <name>`.

## Analysis

The analysis reads only `results/*.csv`. `mvt.analysis.io.load_results()` already applies the inclusion rules: real
phases only, failed runs dropped and counted, and every score floored. Main measurements use the 32 Sobol trials.
Uncertainty comes from a paired bootstrap over the 32 Sobol indices, resampling the same indices for both
architectures.

| Measure | Question it answers | Definition |
|---|---|---|
| Expected best, Best(n) | How good is the best of *n* random tries? (RQ1) | Dodge et al. (2019) formula over the 32 Sobol scores |
| Gap(n) and flip budget | Who leads at budget *n*, and does the lead flip? (RQ1) | Best_Transformer(n) − Best_Mamba(n), with confidence intervals |
| Robustness | How forgiving is the model? (RQ2) | Share of Sobol trials within ε of that model's best |
| Importance | Which hyperparameters matter? (RQ2) | fANOVA on the 32 Sobol trials |
| Tunability | How much does tuning help? | Mean test score of `best` − mean of `default` |
| Transfer, search check | Do settings and conclusions carry over? (RQ4) | Rank correlation of Sobol scores; Sobol-only vs. Sobol + TPE |

<details>
<summary><b>Exact formulas and reference test values</b></summary>

- **Expected best:** sort the N = 32 scores ascending as s₍₁₎ ≤ … ≤ s₍N₎. Then Best(n) = Σᵢ s₍ᵢ₎ [(i/N)ⁿ − ((i−1)/N)ⁿ].
- **Flip budget:** the smallest *n* at which Gap(n)'s 95% interval lies entirely on the opposite side of zero from some smaller *n*.
- **Robustness:** also report it at ε/2 and 2ε, and on the 16 Sobol points with a mid-range learning rate.
- **GPU-time axis:** plot Best against *n* × the mean trial time. This is valid only for runs on the CUDA Mamba kernels.
- **Bootstrap:** 2,000 resamples (200 for fANOVA), percentile 95% intervals, seed 0.
- **Test values:** scores [70, 80, 85, 90] give Best(1) = 81.25, Best(2) = 85.3125, Best(4) = 88.06640625. Robustness with ε = 6 is 0.5.

</details>

## Troubleshooting

| Problem | Fix |
|---|---|
| Mamba kernels fail to build, or `mamba_ssm` import fails with "undefined symbol" | They must be compiled against the installed PyTorch: `pip install --no-build-isolation -r requirements/colab.txt`. If the pinned versions don't build on Colab's PyTorch, update the pins in a pull request. |
| `mvt info` or `mvt run` reports "GPU found but mamba_ssm does not import" | The kernels aren't installed (see above). Don't work around this for real runs. Only for debugging, `impl: ref` in `configs/models/mamba.yaml` forces the slow version. |
| NaN losses in one cell that look like fp16 overflow | Add that cell's id to `fp32_cells` in its task YAML, for example `fp32_cells: [sst2-025-8m]`. Both architectures of that cell switch to fp32 and no other cell changes. Move that cell's old results aside, then re-run it. |
| Colab disconnected mid-run | Re-run the notebook. Finished trials are skipped automatically. |
| `run` refuses to start | Commit your changes (real runs need a clean tree) and set `MVT_OWNER`. |

## Glossary

| Term | Meaning |
|---|---|
| Cell | One task × data size × model size, for example `sst2-025-8m`. |
| Sobol points | 32 hyperparameter settings spread evenly over the search space, fixed in a committed table. |
| TPE | Tree-structured Parzen Estimator, Optuna's Bayesian search that proposes settings near the good ones found so far. |
| λ* / λ₀ | The best setting found / the shared default setting. |
| Floor | The score of random guessing. No run counts below it. |
| ε (epsilon) | The margin that counts as "about as good as the best" when measuring robustness. |
| Non-embedding parameters | Model parameters excluding the word-embedding table and the task head. |

## References

- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752
- Waleffe et al. (2024). *An Empirical Study of Mamba-based Language Models.* arXiv:2406.07887
- Okpekpe & Orvieto (2025). *Revisiting Associative Recall in Modern Recurrent Models.* arXiv:2508.19029
- Melis, Dyer & Blunsom (2018). *On the State of the Art of Evaluation in Neural Language Models.* arXiv:1707.05589
- Dodge et al. (2019). *Show Your Work: Improved Reporting of Experimental Results.* arXiv:1909.03004
- Hutter, Hoos & Leyton-Brown (2014). *An Efficient Approach for Assessing Hyperparameter Importance.* ICML
- Probst, Boulesteix & Bischl (2019). *Tunability: Importance of Hyperparameters of Machine Learning Algorithms.* arXiv:1802.09596
- Akiba et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* arXiv:1907.10902
- Touvron et al. (2023). *LLaMA.* arXiv:2302.13971 · Zhu et al. (2024). *Vision Mamba.* arXiv:2401.09417
- Datasets: Socher et al. (2013), SST · Tjong Kim Sang & De Meulder (2003), CoNLL-2003 · Merity et al. (2017), WikiText. arXiv:1609.07843
