"""Deterministic, resumable TPE (README: "What happens inside one cell").

For TPE trial k, an in-memory Optuna study is rebuilt from the SAVED results (32 Sobol + TPE
0..k-1), and a freshly seeded ``TPESampler`` suggests one configuration. There is no Optuna
database: the result JSON files are the only source of truth.
"""

from __future__ import annotations

from collections.abc import Sequence

import optuna

from mvt.core.types import HParams
from mvt.search.space import SearchSpace

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest(space: SearchSpace, history: Sequence[tuple[HParams, float]], seed: int) -> HParams:
    """Suggest the next configuration given (hparams, floored validation score) pairs."""
    distributions = space.distributions()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    for hp, score in history:
        study.add_trial(
            optuna.trial.create_trial(
                params=space.to_optuna_params(hp), distributions=distributions, value=score
            )
        )
    trial = study.ask(distributions)
    return space.from_optuna_params(trial.params)
