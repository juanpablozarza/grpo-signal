"""Read rollouts from a prime-rl run directory's file-monitor traces."""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from grpo_signal.types import Rollout


def read_prime_run(
    run_dir: Path,
    *,
    kind: Literal["train", "eval"] = "train",
    steps: slice | None = None,
) -> Iterator[Rollout]:
    """Yield rollouts from ``<run_dir>/monitors/file/`` traces.

    Uses the episode stream (plain and ``.jsonl.zst`` chunks) so rows can be
    grouped by task. ``kind`` selects train vs eval traces. ``steps`` limits
    dispatch/arrival step, e.g. ``slice(0, 50)``.
    """
    raise NotImplementedError("read_prime_run")
