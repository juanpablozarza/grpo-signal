"""Read rollouts from a prime-rl run directory's file-monitor traces."""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from grpo_signal.types import GroupRecord


def read_prime_run(
    run_dir: Path,
    *,
    kind: Literal["train", "eval"] = "train",
    steps: slice | None = None,
) -> Iterator[GroupRecord]:
    """Yield GRPO groups from ``<run_dir>/monitors/file/`` traces.

    Reads the episode stream (plain and ``.jsonl.zst`` chunks via
    ``iter_chunked_jsonl``) so rows can be grouped by task. The sibling
    ``stream.index.jsonl`` has no task key and no per-component rewards, so it
    is not enough. Joins ``traces/annotations/<producer>/`` records by
    ``trace_id`` in write order, newest wins.

    Each yielded group has ``env``, ``task_key`` (``task.key`` or fallback
    ``task.hash``), ``group_id``, and one ``TraceRecord`` per trace with
    correctness ``score``, final ``reward``, ``step``, and ``effective``.
    Dead groups are reported on both metrics. ``kind`` selects train vs eval
    traces (``trace.info.kind``). ``steps`` limits dispatch/arrival step, e.g.
    ``slice(0, 50)``.
    """
    raise NotImplementedError("read_prime_run")
