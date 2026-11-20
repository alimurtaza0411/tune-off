"""TEMPLATE for a real task. Copy to ``tasks/<name>.py`` and fill in. (Not registered.)

Checklist (see README: "Adding a task"):
  1. ``configs/tasks/<name>.yaml`` exists with every field filled and ``dataset.revision`` pinned.
  2. Decorate the class with ``@register_task("<name>")`` (and uncomment it below).
  3. ``build`` is deterministic, and:
       * val and test NEVER depend on ``fraction``;
       * the vocabulary is built from the FULL training pool (never from the subset);
       * the subset is drawn by ``self.finalize`` with ``config.subset_seed``;
       * truncation to ``config.max_len`` is counted (must be zero on val and test).
  4. Override ``compute_metric`` only if the head's default metric is wrong for the task.
  5. Import ``datasets`` and ``seqeval`` INSIDE functions, never at the top of the file: every
     task module is imported when tasks are discovered, so a top-level import of an optional
     package would break all tasks wherever it is missing.
  6. Add ``tests/data/test_<name>.py`` (marker ``data``) asserting the split sizes.
  7. ``pytest tests/contract -m "data or not data"`` must pass; it runs for every registered task.
"""

from __future__ import annotations

from mvt.core.types import ClsExample, TaskData
from mvt.data.vocab import Vocab
from mvt.tasks.base import BaseTask

# from mvt.core.registry import register_task


# @register_task("my_task")
class MyTask(BaseTask):
    def build(self, fraction: float) -> TaskData:
        c = self.config
        # 1) Load the raw splits. Pin the revision so the data never silently changes:
        #    from datasets import load_dataset
        #    ds = load_dataset(c.dataset.id, c.dataset.config, revision=c.dataset.revision)
        raw_train = [("a good movie".split(), 1), ("a bad movie".split(), 0)]
        raw_val, raw_test = raw_train, raw_train

        # 2) Vocabulary from the FULL training pool.
        vocab = Vocab.build((toks for toks, _ in raw_train), min_freq=c.min_freq)

        # 3) Encode + truncate, counting truncations.
        n_truncated = {"train": 0, "val": 0, "test": 0}

        def encode(split: str, rows: list) -> list[ClsExample]:
            out = []
            for toks, label in rows:
                ids, cut = self.truncate(vocab.encode(toks), c.max_len)
                n_truncated[split] += cut
                out.append(ClsExample(input_ids=ids, label=label))
            return out

        train_pool = encode("train", raw_train)
        val, test = encode("val", raw_val), encode("test", raw_test)

        # 4) Spec + subsample + stats.
        spec = self.make_spec(vocab_size=len(vocab))
        stats = {f"n_truncated_{k}": v for k, v in n_truncated.items()}
        return self.finalize(spec, train_pool, val, test, fraction, stats)
