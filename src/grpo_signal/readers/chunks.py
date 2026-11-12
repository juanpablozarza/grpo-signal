"""Read prime-rl chunked JSONL streams (plain and zstd-sealed)."""

from collections.abc import Iterator
from pathlib import Path


def iter_chunked_jsonl(directory: Path) -> Iterator[bytes]:
    """Yield raw JSONL lines from a chunked stream directory.

    Chunks are ``00000.jsonl``, ``00001.jsonl``, and so on. A finished chunk is
    sealed as ``NNNNN.jsonl.zst`` (zstd) and the plain file is removed. The live
    chunk stays uncompressed. A line never spans chunks.
    """
    raise NotImplementedError("iter_chunked_jsonl")
