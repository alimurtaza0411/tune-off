## What changed and why


## Checklist
- [ ] Contract touched? If yes, link the `contract-change` issue: #
- [ ] Tests added or updated
- [ ] `ruff check`, `ruff format --check` and `pytest -m "not gpu and not data"` pass
- [ ] Task PR: `pytest -m data` passes and the dataset revision is pinned
- [ ] Results PR: `python -m mvt validate` passes
