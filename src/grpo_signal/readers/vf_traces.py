"""Read rollouts from a verifiers vf-run traces.jsonl file."""

from collections.abc import Iterator
from pathlib import Path

from grpo_signal.types import Rollout


def read_vf_traces(path: Path) -> Iterator[Rollout]:
    """Yield rollouts from an uncompressed vf-run ``traces.jsonl``.

    Each line is the same Episode JSON as the prime-rl file-monitor stream.
    """
    raise NotImplementedError("read_vf_traces")
