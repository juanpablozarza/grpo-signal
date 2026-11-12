from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import zstandard

from grpo_signal.readers.chunks import (
    ChunkLineError,
    discover_chunks,
    iter_lines,
    iter_records,
)


def _write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
    *,
    truncated: str | None = None,
) -> None:
    body = "".join(json.dumps(record) + "\n" for record in records)
    if truncated is not None:
        body += truncated
    path.write_text(body, encoding="utf-8")


def _write_zst_frames(path: Path, frames: list[bytes]) -> None:
    compressor = zstandard.ZstdCompressor()
    with path.open("wb") as fh:
        for frame in frames:
            fh.write(compressor.compress(frame))


def _frame(*records: dict[str, Any]) -> bytes:
    return b"".join(json.dumps(record).encode("utf-8") + b"\n" for record in records)


def test_plain_jsonl(tmp_path: Path) -> None:
    stream = tmp_path / "stream"
    stream.mkdir()
    _write_jsonl(
        stream / "00000.jsonl",
        [{"id": "a"}, {"id": "b"}],
    )

    records = iter_records(stream)
    assert list(records) == [{"id": "a"}, {"id": "b"}]
    assert records.stats.files_read == 1
    assert records.stats.lines == 2
    assert records.stats.skipped_truncated == 0
    assert records.stats.skipped_malformed == 0
    assert records.stats.bytes == (stream / "00000.jsonl").stat().st_size

    lines = iter_lines(stream)
    assert list(lines) == ['{"id": "a"}', '{"id": "b"}']
    assert lines.stats.lines == 2


def test_multi_frame_zstd(tmp_path: Path) -> None:
    stream = tmp_path / "stream"
    stream.mkdir()
    _write_zst_frames(
        stream / "00000.jsonl.zst",
        [
            _frame({"frame": 1, "id": "a"}),
            _frame({"frame": 2, "id": "b"}, {"frame": 2, "id": "c"}),
        ],
    )

    records = iter_records(stream)
    assert list(records) == [
        {"frame": 1, "id": "a"},
        {"frame": 2, "id": "b"},
        {"frame": 2, "id": "c"},
    ]
    assert records.stats.files_read == 1
    assert records.stats.lines == 3
    assert records.stats.bytes == (stream / "00000.jsonl.zst").stat().st_size


def test_truncated_last_line_in_live_jsonl(tmp_path: Path) -> None:
    stream = tmp_path / "stream"
    stream.mkdir()
    _write_jsonl(
        stream / "00000.jsonl",
        [{"id": "ok"}],
        truncated='{"id": "partial"',
    )

    records = iter_records(stream)
    assert list(records) == [{"id": "ok"}]
    assert records.stats.lines == 1
    assert records.stats.skipped_truncated == 1
    assert records.stats.skipped_malformed == 0


def test_truncated_line_in_sealed_chunk_raises(tmp_path: Path) -> None:
    stream = tmp_path / "stream"
    stream.mkdir()
    _write_zst_frames(
        stream / "00000.jsonl.zst",
        [_frame({"id": "ok"}) + b'{"id": "partial"'],
    )

    records = iter_records(stream)
    with pytest.raises(ChunkLineError, match=r"00000\.jsonl\.zst:2: truncated line"):
        list(records)
    assert records.stats.skipped_truncated == 1


def test_truncated_line_lenient(tmp_path: Path) -> None:
    stream = tmp_path / "stream"
    stream.mkdir()
    _write_zst_frames(
        stream / "00000.jsonl.zst",
        [_frame({"id": "ok"}) + b'{"id": "partial"'],
    )
    _write_jsonl(stream / "00001.jsonl", [{"id": "later"}])

    records = iter_records(stream, lenient=True)
    assert list(records) == [{"id": "ok"}, {"id": "later"}]
    assert records.stats.skipped_truncated == 1
    assert records.stats.lines == 2


def test_truncated_line_in_non_last_jsonl_raises(tmp_path: Path) -> None:
    stream = tmp_path / "stream"
    stream.mkdir()
    _write_jsonl(
        stream / "00000.jsonl",
        [{"id": "ok"}],
        truncated='{"id": "partial"',
    )
    _write_jsonl(stream / "00001.jsonl", [{"id": "later"}])

    records = iter_records(stream)
    with pytest.raises(ChunkLineError, match=r"00000\.jsonl:2: truncated line"):
        list(records)


def test_ordering_by_numeric_index(tmp_path: Path) -> None:
    stream = tmp_path / "stream"
    stream.mkdir()
    (stream / "readme.txt").write_text("ignore me\n", encoding="utf-8")
    (stream / "00002.jsonl.zst.tmp").write_bytes(b"not a chunk")
    _write_jsonl(stream / "00002.jsonl", [{"i": 2}])
    _write_zst_frames(stream / "00000.jsonl.zst", [_frame({"i": 0})])
    _write_zst_frames(stream / "00001.jsonl.zst", [_frame({"i": 1})])
    _write_jsonl(stream / "00010.jsonl", [{"i": 10}])

    records = iter_records(stream)
    assert [row["i"] for row in records] == [0, 1, 2, 10]
    assert records.stats.files_read == 4
    assert discover_chunks(stream) == [
        stream / "00000.jsonl.zst",
        stream / "00001.jsonl.zst",
        stream / "00002.jsonl",
        stream / "00010.jsonl",
    ]


def test_both_plain_and_zst_prefers_sealed(tmp_path: Path) -> None:
    stream = tmp_path / "stream"
    stream.mkdir()
    _write_jsonl(stream / "00000.jsonl", [{"from": "plain"}])
    _write_zst_frames(stream / "00000.jsonl.zst", [_frame({"from": "zst"})])
    _write_jsonl(stream / "00001.jsonl", [{"from": "live"}])

    with pytest.warns(UserWarning, match=r"preferring sealed 00000\.jsonl\.zst"):
        records = iter_records(stream)
        assert list(records) == [{"from": "zst"}, {"from": "live"}]
    assert records.stats.files_read == 2


def test_malformed_middle_line_raises(tmp_path: Path) -> None:
    stream = tmp_path / "stream"
    stream.mkdir()
    (stream / "00000.jsonl").write_text(
        '{"id": "a"}\nnot-json\n{"id": "c"}\n',
        encoding="utf-8",
    )

    records = iter_records(stream)
    with pytest.raises(
        ChunkLineError, match=r"00000\.jsonl:2: malformed JSON"
    ) as caught:
        list(records)
    assert caught.value.path == stream / "00000.jsonl"
    assert caught.value.line_number == 2
    assert records.stats.lines == 2
    assert records.stats.skipped_malformed == 1


def test_malformed_middle_line_lenient(tmp_path: Path) -> None:
    stream = tmp_path / "stream"
    stream.mkdir()
    (stream / "00000.jsonl").write_text(
        '{"id": "a"}\nnot-json\n[1]\n{"id": "c"}\n',
        encoding="utf-8",
    )

    records = iter_records(stream, lenient=True)
    assert list(records) == [{"id": "a"}, {"id": "c"}]
    assert records.stats.lines == 4
    assert records.stats.skipped_malformed == 2
    assert records.stats.skipped_truncated == 0


def test_empty_dir(tmp_path: Path) -> None:
    stream = tmp_path / "stream"
    stream.mkdir()

    records = iter_records(stream)
    assert list(records) == []
    assert records.stats.files_read == 0
    assert records.stats.lines == 0
    assert records.stats.skipped_truncated == 0
    assert records.stats.bytes == 0
    assert discover_chunks(stream) == []
    assert list(iter_lines(stream)) == []
