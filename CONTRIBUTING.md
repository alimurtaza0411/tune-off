# Contributing

1. **Ownership.** Edit only files you own (see `.github/CODEOWNERS`). To change someone else's file, open an issue or a pull request assigned to its owner.
2. **No central lists.** Tasks and models register themselves in their own files.
3. **Workflow.**
   - `main` is protected.
   - Use short-lived feature branches (`<name>/<topic>`) and small pull requests.
   - Each pull request needs **one approving review** (core paths: the skeleton owner) and **green CI**.
   - Rebase on `main` before merging.
4. **CI must pass:** `ruff check`, `ruff format --check`, and `pytest -m "not gpu and not data"`.
   - Task owners also run `pytest -m data` locally or on Colab before merging their task.
   - The skeleton owner runs `pytest -m gpu` on Colab before each tagged release.
5. **Contracts change only via the contract-change protocol** (README §6). Prefer additive changes.
6. **Code freeze.** Real experiments run only from a tagged release with a clean tree. After `v1.0.0`, changes to shared code are bug fixes reviewed by two people.
7. **Results.**
   - Raw JSON in `MVT_RESULTS_ROOT` is the source of truth while running.
   - Each owner exports and commits **only their own cells' CSVs**, through a pull request, after `mvt validate` passes.
   - Never edit result files by hand.
   - Never commit raw run folders, datasets, caches or checkpoints.
8. **Style.**
   - Python ≥ 3.10, type hints on public functions, Google-style docstrings.
   - Use `logging`, not `print`, in library code.
   - No experiment logic in notebooks.
   - No hard-coded experimental constants (they live in `configs/`).
   - Pin new dependencies in `pyproject.toml` through a pull request.
9. **Definition of done.**
   - **For a task:** the contract tests and `-m data` tests pass, with expected split sizes asserted; the dataset revision is pinned; the pilot is complete; the config is reviewed.
   - **For a cell:** `mvt status` shows 66/66 per architecture; `mvt validate` passes; the CSV is merged.

## Pull-request checklist
- [ ] What changed and why is described.
- [ ] Contract touched? If yes, link the `contract-change` issue.
- [ ] Tests added or updated.
- [ ] `ruff check`, `ruff format --check` and `pytest -m "not gpu and not data"` pass.
- [ ] For tasks: `pytest -m data` passes and the dataset revision is pinned.
- [ ] For results: `python -m mvt validate` passes.
