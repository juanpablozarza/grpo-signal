"""Read prime-rl chunked JSONL streams (plain and zstd-sealed)."""

from collections.abc import Iterator
from pathlib import Path


def iter_lines(path: Path) -> Iterator[bytes]:
    """Yield complete JSONL records from a plain ``.jsonl`` or sealed ``.jsonl.zst``.

    Sealed chunks are zstd level 3 with seekable 4 MiB frames (a skippable seek
    table may trail the data frames). A truncated final line, typical of a crash
    mid-write on the live chunk, has no trailing newline and is dropped. Empty
    lines are skipped. A line never spans files.
    """
    raise NotImplementedError("iter_lines")


def iter_chunked_jsonl(directory: Path) -> Iterator[bytes]:
    """Yield raw JSONL lines from a chunked stream directory, in chunk order.

    Chunks are ``00000.jsonl``, ``00001.jsonl``, and so on. A finished chunk is
    sealed as ``NNNNN.jsonl.zst`` (zstd) and the plain file is removed. The live
    chunk stays uncompressed. Prefer the plain file when both exist. Each chunk
    is read with ``iter_lines``, so a truncated last line on the live chunk is
    dropped.
    """
    raise NotImplementedError("iter_chunked_jsonl")
