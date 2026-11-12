from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import zstandard

from grpo_signal.readers import iter_chunked_jsonl, read_prime_run
from grpo_signal.readers.chunks import iter_lines
from grpo_signal.types import CORRECTNESS_HEAD, DEAD_EPS, GroupRecord
from tests.fixtures.gen_prime_run import (
    SEED,
    default_paths,
    generate,
    write_seekable_zst,
)

FIXTURE_DIR, EXPECTED_PATH = default_paths()
STREAM = FIXTURE_DIR / "monitors" / "file" / "traces" / "stream"
ANN = FIXTURE_DIR / "monitors" / "file" / "traces" / "annotations" / "orchestrator"


def _load_expected() -> dict:
    return json.loads(EXPECTED_PATH.read_text())


def _complete_lines(blob: bytes) -> list[bytes]:
    parts = blob.split(b"\n")
    if parts and parts[-1] == b"":
        parts = parts[:-1]
    elif parts:
        parts = parts[:-1]
    return [p for p in parts if p]


def _read_chunk(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix == ".zst":
        return zstandard.ZstdDecompressor().decompress(data)
    return data


def _stream_records() -> list[dict]:
    chunks = sorted(STREAM.glob("*.jsonl*"), key=lambda p: p.name)
    records: list[dict] = []
    for path in chunks:
        for line in _complete_lines(_read_chunk(path)):
            records.append(json.loads(line))
    return records


def _annotations() -> dict[str, dict]:
    folded: dict[str, dict] = {}
    for path in sorted(ANN.glob("*.jsonl*"), key=lambda p: p.name):
        for line in _complete_lines(_read_chunk(path)):
            rec = json.loads(line)
            folded[rec["trace_id"]] = rec
    return folded


def _correctness_score(trace: dict) -> float:
    rewards = trace.get("rewards") or {}
    head = rewards.get(CORRECTNESS_HEAD)
    if isinstance(head, dict) and isinstance(head.get("score"), (int, float)):
        return float(head["score"])
    for rec in rewards.values():
        if isinstance(rec, dict) and isinstance(rec.get("score"), (int, float)):
            return float(rec["score"])
    return float(trace["reward"])


def _task_key(episode: dict) -> str:
    task = episode.get("task") or {}
    key = task.get("key")
    if isinstance(key, str) and key:
        return key
    return str(task.get("hash"))


def _is_dead(values: list[float]) -> bool:
    if len(values) < 2:
        return True
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return var**0.5 < DEAD_EPS


def _groups_from_fixture() -> list[GroupRecord]:
    from grpo_signal.types import TraceRecord

    anns = _annotations()
    by_group: dict[str, GroupRecord] = {}
    order: list[str] = []
    for episode in _stream_records():
        gid = (episode.get("group") or {}).get("id")
        env = (episode.get("env") or {}).get("name") or (episode.get("env") or {}).get(
            "id"
        )
        if gid not in by_group:
            by_group[gid] = GroupRecord(
                env=env,
                task_key=_task_key(episode),
                group_id=gid,
                traces=[],
            )
            order.append(gid)
        for trace in episode.get("traces") or []:
            ann = anns.get(trace["id"]) or {}
            info = ann.get("info") or {}
            effective = bool(info.get("effective"))
            ship = info.get("ship") or {}
            arrival = (trace.get("info") or {}).get("arrival") or {}
            step = ship.get("step") if effective else arrival.get("step")
            by_group[gid].traces.append(
                TraceRecord(
                    trace_id=trace["id"],
                    score=_correctness_score(trace),
                    reward=float(trace["reward"]),
                    step=step,
                    effective=effective,
                )
            )
    return [by_group[gid] for gid in order]


def _counts(groups: list[GroupRecord]) -> tuple[list[dict], list[dict]]:
    tasks: dict[tuple[str, str], dict] = {}
    families: dict[str, dict] = {}
    for group in groups:
        key = (group.env, group.task_key)
        task = tasks.setdefault(
            key,
            {
                "env": group.env,
                "task_key": group.task_key,
                "n": 0,
                "k": 0,
                "dead_on_score": 0,
                "dead_on_reward": 0,
            },
        )
        task["n"] += len(group.traces)
        task["k"] += sum(1 for t in group.traces if t.score == 1.0)
        task["dead_on_score"] += int(_is_dead([t.score for t in group.traces]))
        task["dead_on_reward"] += int(_is_dead([t.reward for t in group.traces]))
        fam = families.setdefault(
            group.env,
            {
                "family": group.env,
                "groups": 0,
                "dead_on_score": 0,
                "dead_on_reward": 0,
                "effective": 0,
            },
        )
        fam["groups"] += 1
        fam["dead_on_score"] += int(_is_dead([t.score for t in group.traces]))
        fam["dead_on_reward"] += int(_is_dead([t.reward for t in group.traces]))
        fam["effective"] += int(any(t.effective for t in group.traces))
    task_rows = list(tasks.values())
    family_rows = [families[name] for name in ("gsm8k", "countdown", "code")]
    return task_rows, family_rows


def test_committed_fixture_matches_expected_counts() -> None:
    """Ground-truth file matches a walk of the committed run directory."""
    expected = _load_expected()
    groups = _groups_from_fixture()
    assert len(groups) == expected["meta"]["n_groups"]
    assert sum(len(g.traces) for g in groups) == expected["meta"]["n_episodes"]
    tasks, families = _counts(groups)
    exp_tasks = [
        {
            k: row[k]
            for k in ("env", "task_key", "n", "k", "dead_on_score", "dead_on_reward")
        }
        for row in expected["tasks"]
    ]
    assert tasks == exp_tasks
    assert families == expected["families"]


def test_fixture_layout_and_index_omissions() -> None:
    expected = _load_expected()
    assert (STREAM / "00000.jsonl.zst").is_file()
    assert (STREAM / "00001.jsonl").is_file()
    assert not (STREAM / "00000.jsonl").exists()
    live = (STREAM / "00001.jsonl").read_bytes()
    assert not live.endswith(b"\n")
    with pytest.raises(json.JSONDecodeError):
        json.loads(live.rsplit(b"\n", 1)[-1])

    sealed = zstandard.ZstdDecompressor().decompress(
        (STREAM / "00000.jsonl.zst").read_bytes()
    )
    assert sealed.endswith(b"\n")
    assert json.loads(sealed.splitlines()[0])["task"]["key"]

    index_path = FIXTURE_DIR / "monitors" / "file" / "traces" / "stream.index.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line]
    assert len(rows) == expected["meta"]["n_episodes"]
    for row in rows:
        assert "task" not in row
        assert "rewards" not in row
        assert "metrics" not in row
        assert "key" not in row
        assert isinstance(row.get("dispatch"), float)
        assert isinstance(row.get("arrival"), float)

    metrics = [
        json.loads(line)
        for line in (FIXTURE_DIR / "monitors" / "file" / "metrics.jsonl")
        .read_text()
        .splitlines()
        if line
    ]
    assert metrics[0]["producer"] == "orchestrator"
    assert metrics[0]["step"] == 0
    assert "time" in metrics[0]

    anns = _annotations()
    assert anns
    sample = next(iter(anns.values()))
    assert sample["version"] == 1
    assert sample["info"]["effective"] is True
    assert "step" in sample["info"]["ship"]
    assert "time" in sample["info"]["ship"]


def test_length_penalty_groups_dead_on_score_alive_on_reward() -> None:
    expected = _load_expected()
    keys = {
        row["task_key"] for row in expected["tasks"] if row["kind"] == "length_penalty"
    }
    groups = [g for g in _groups_from_fixture() if g.task_key in keys]
    assert groups
    for group in groups:
        scores = [t.score for t in group.traces]
        rewards = [t.reward for t in group.traces]
        assert all(s == 1.0 for s in scores)
        assert _is_dead(scores)
        assert not _is_dead(rewards)
        assert all(t.effective for t in group.traces)
        assert all(t.step is not None for t in group.traces)


def test_pruned_groups_have_no_effective_annotation() -> None:
    groups = _groups_from_fixture()
    pruned = [g for g in groups if not any(t.effective for t in g.traces)]
    trained = [g for g in groups if any(t.effective for t in g.traces)]
    assert pruned and trained
    assert all(t.effective is False for g in pruned for t in g.traces)
    assert all(t.effective is True for g in trained for t in g.traces)


def test_one_task_falls_back_to_hash() -> None:
    records = _stream_records()
    missing = [r for r in records if "key" not in (r.get("task") or {})]
    assert missing
    assert all((r["task"].get("hash")) for r in missing)
    groups = _groups_from_fixture()
    assert any(g.task_key == missing[0]["task"]["hash"] for g in groups)


def test_checksums_match_committed_files() -> None:
    expected = _load_expected()
    checksums = expected["checksums"]
    files = {
        p.relative_to(FIXTURE_DIR).as_posix(): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in sorted(FIXTURE_DIR.rglob("*"))
        if p.is_file()
    }
    assert files == checksums


def test_generator_is_deterministic(tmp_path: Path) -> None:
    dest = tmp_path / "prime_run_small"
    expected_path = tmp_path / "expected_small.json"
    generate(dest, expected_path=expected_path, seed=SEED)
    committed = _load_expected()
    regenerated = json.loads(expected_path.read_text())
    assert regenerated["checksums"] == committed["checksums"]
    assert regenerated["tasks"] == committed["tasks"]
    assert regenerated["families"] == committed["families"]
    for rel, digest in committed["checksums"].items():
        assert hashlib.sha256((dest / rel).read_bytes()).hexdigest() == digest


def test_seekable_zst_roundtrip(tmp_path: Path) -> None:
    src = b'{"a":1}\n{"b":2}\n'
    dest = tmp_path / "00000.jsonl.zst"
    write_seekable_zst(src, dest)
    assert zstandard.ZstdDecompressor().decompress(dest.read_bytes()) == src
    footer_magic = (0x8F92EAB1).to_bytes(4, "little")
    assert dest.read_bytes().endswith(footer_magic)


def _skip_unimplemented(exc: NotImplementedError) -> None:
    pytest.skip(f"reader not implemented: {exc}")


def test_iter_chunked_jsonl_matches_stream() -> None:
    try:
        lines = list(iter_chunked_jsonl(STREAM))
    except NotImplementedError as exc:
        _skip_unimplemented(exc)
        return
    expected_n = _load_expected()["meta"]["n_episodes"]
    assert len(lines) == expected_n
    for line in lines:
        json.loads(line)


def test_iter_lines_on_plain_and_zst() -> None:
    try:
        zst_lines = list(iter_lines(STREAM / "00000.jsonl.zst"))
        live_lines = list(iter_lines(STREAM / "00001.jsonl"))
    except NotImplementedError as exc:
        _skip_unimplemented(exc)
        return
    assert zst_lines
    assert live_lines
    json.loads(zst_lines[0])
    json.loads(live_lines[-1])
    live = (STREAM / "00001.jsonl").read_bytes()
    assert not live.endswith(b"\n")
    assert len(live_lines) == len(_complete_lines(live))


def test_read_prime_run_matches_expected() -> None:
    try:
        groups = list(read_prime_run(FIXTURE_DIR, kind="train"))
    except NotImplementedError as exc:
        _skip_unimplemented(exc)
        return
    expected = _load_expected()
    tasks, families = _counts(groups)
    exp_tasks = [
        {
            k: row[k]
            for k in ("env", "task_key", "n", "k", "dead_on_score", "dead_on_reward")
        }
        for row in expected["tasks"]
    ]
    assert tasks == exp_tasks
    assert families == expected["families"]
