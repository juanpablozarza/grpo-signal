"""Read prime-rl chunked JSONL streams (plain and zstd-sealed).

prime-rl's file monitor stores an append-only episode stream at
``<run_dir>/monitors/file/traces/stream/``. Live chunks are
``{n:05d}.jsonl``; a finished chunk is sealed beside itself as
``{n:05d}.jsonl.zst`` (zstd, seekable concatenated frames) and the
plain file is then removed. A line never spans chunks.

Naming is defined by prime-rl ``chunk_path`` / ``seal``:

https://github.com/PrimeIntellect-ai/prime-rl/blob/bd326dca348a1af68b6d543c237837354bd783b7/src/prime_rl/monitors/file/traces/chunks.py#L1-L50
"""

from __future__ import annotations

import io
import json
import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Generic, TypeVar, cast

import zstandard

# Live: NNNNN.jsonl; sealed: NNNNN.jsonl.zst. Width is 5 in prime-rl until 99999.
_CHUNK_NAME = re.compile(r"^(?P<index>\d+)\.jsonl(?P<sealed>\.zst)?$")

T = TypeVar("T")


@dataclass(kw_only=True, slots=True)
class ChunkStats:
    """Counters for one pass over a chunked stream directory.

    ``bytes`` is the sum of on-disk sizes of chunk files actually opened.
    ``lines`` counts complete (newline-terminated) lines, including ones later
    skipped as malformed.
    """

    files_read: int = 0
    lines: int = 0
    skipped_truncated: int = 0
    skipped_malformed: int = 0
    bytes: int = 0


class ChunkLineError(ValueError):
    """A JSONL line that cannot be used, with its file and 1-based line number."""

    def __init__(self, path: Path, line_number: int, message: str) -> None:
        self.path = path
        self.line_number = line_number
        super().__init__(f"{path}:{line_number}: {message}")


class ChunkIterator(Generic[T]):
    """Iterator that also carries :class:`ChunkStats` for the same scan.

    ``stats`` is updated as items are consumed and is complete after the
    iterator is exhausted.
    """

    def __init__(self, stats: ChunkStats, items: Iterator[T]) -> None:
        self.stats = stats
        self._items = items

    def __iter__(self) -> ChunkIterator[T]:
        return self

    def __next__(self) -> T:
        return next(self._items)


def iter_chunked_jsonl(directory: Path) -> Iterator[bytes]:
    """Yield raw JSONL lines from a chunked stream directory.

    Chunks are ``00000.jsonl``, ``00001.jsonl``, and so on. A finished chunk is
    sealed as ``NNNNN.jsonl.zst`` (zstd) and the plain file is removed. The live
    chunk stays uncompressed. A line never spans chunks.
    """
    raise NotImplementedError("iter_chunked_jsonl")


def discover_chunks(directory: Path) -> list[Path]:
    """Return chunk files in numeric index order.

    When both ``NNNNN.jsonl`` and ``NNNNN.jsonl.zst`` exist for the same index,
    the sealed ``.zst`` is kept and a warning is issued.
    """
    by_index: dict[int, dict[str, Path]] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        matched = _CHUNK_NAME.fullmatch(path.name)
        if matched is None:
            continue
        index = int(matched.group("index"))
        kind = "zst" if matched.group("sealed") else "jsonl"
        by_index.setdefault(index, {})[kind] = path

    chosen: list[Path] = []
    for index in sorted(by_index):
        kinds = by_index[index]
        zst = kinds.get("zst")
        plain = kinds.get("jsonl")
        if zst is not None and plain is not None:
            warnings.warn(
                f"chunk {index} has both {plain.name} and {zst.name}; "
                f"preferring sealed {zst.name}",
                UserWarning,
                stacklevel=2,
            )
            chosen.append(zst)
        elif zst is not None:
            chosen.append(zst)
        elif plain is not None:
            chosen.append(plain)
    return chosen


def iter_lines(
    directory: Path,
    *,
    lenient: bool = False,
) -> ChunkIterator[str]:
    """Yield complete JSONL lines (no trailing newline) from ``directory``.

    A truncated final line in the last live ``.jsonl`` chunk is skipped and
    counted. Any other truncated line raises :class:`ChunkLineError` unless
    ``lenient`` is true.
    """
    stats = ChunkStats()
    return ChunkIterator(stats, _iter_line_text(directory, stats, lenient))


def iter_records(
    directory: Path,
    *,
    lenient: bool = False,
) -> ChunkIterator[dict[str, object]]:
    """Yield parsed JSON objects from ``directory``.

    Same truncation rules as :func:`iter_lines`. A non-object or invalid JSON
    line raises :class:`ChunkLineError` unless ``lenient`` is true, in which
    case it is skipped and counted on ``stats.skipped_malformed``.
    """
    stats = ChunkStats()
    return ChunkIterator(stats, _iter_parsed_records(directory, stats, lenient))


def _iter_line_text(
    directory: Path,
    stats: ChunkStats,
    lenient: bool,
) -> Iterator[str]:
    for _path, _line_number, text in _iter_complete_lines(directory, stats, lenient):
        yield text


def _iter_parsed_records(
    directory: Path,
    stats: ChunkStats,
    lenient: bool,
) -> Iterator[dict[str, object]]:
    for path, line_number, text in _iter_complete_lines(directory, stats, lenient):
        try:
            parsed: object = json.loads(text)
        except json.JSONDecodeError as exc:
            _malformed(
                path,
                line_number,
                f"malformed JSON: {exc.msg}",
                lenient=lenient,
                stats=stats,
            )
            continue
        if not isinstance(parsed, dict):
            _malformed(
                path,
                line_number,
                f"expected a JSON object, got {type(parsed).__name__}",
                lenient=lenient,
                stats=stats,
            )
            continue
        yield cast(dict[str, object], parsed)


def _iter_complete_lines(
    directory: Path,
    stats: ChunkStats,
    lenient: bool,
) -> Iterator[tuple[Path, int, str]]:
    paths = discover_chunks(directory)
    last = paths[-1] if paths else None
    for path in paths:
        live_tail = path == last and not path.name.endswith(".zst")
        yield from _iter_file_lines(path, stats, lenient, live_tail)


def _iter_file_lines(
    path: Path,
    stats: ChunkStats,
    lenient: bool,
    live_tail: bool,
) -> Iterator[tuple[Path, int, str]]:
    stats.files_read += 1
    stats.bytes += path.stat().st_size
    with path.open("rb") as fh:
        with _text_wrapper(path, fh) as text:
            line_number = 0
            for line in text:
                line_number += 1
                if line.endswith("\n"):
                    stats.lines += 1
                    yield path, line_number, line[:-1]
                    continue
                if live_tail:
                    stats.skipped_truncated += 1
                    continue
                _truncated(path, line_number, lenient=lenient, stats=stats)


def _text_wrapper(path: Path, fh: BinaryIO) -> io.TextIOWrapper:
    if path.name.endswith(".zst"):
        fh = cast(
            BinaryIO,
            zstandard.ZstdDecompressor().stream_reader(fh, read_across_frames=True),
        )
    return io.TextIOWrapper(fh, encoding="utf-8")


def _truncated(
    path: Path,
    line_number: int,
    *,
    lenient: bool,
    stats: ChunkStats,
) -> None:
    stats.skipped_truncated += 1
    if lenient:
        return
    raise ChunkLineError(path, line_number, "truncated line (missing newline)")


def _malformed(
    path: Path,
    line_number: int,
    message: str,
    *,
    lenient: bool,
    stats: ChunkStats,
) -> None:
    stats.skipped_malformed += 1
    if lenient:
        return
    raise ChunkLineError(path, line_number, message)
