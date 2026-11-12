"""Write a small synthetic prime-rl file-monitor run directory.

The layout, field names, and zstd framing follow the writers cited in
``docs/FORMAT.md``. This module does not import prime-rl or verifiers.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import struct
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import zstandard

PRIME_RL_SHA = "5af925d4b1568559607a7b42b3679d9edf3d39de"
VERIFIERS_SHA = "7407a3ea4f4f5e720bd5acfde007a41a33bb65a1"
SEED = 0
GROUP_SIZE = 8
N_STEPS = 20
ZSTD_LEVEL = 3
FRAME_BYTES = 4 << 20
BASE_TIME = 1_700_000_000.0
DEAD_EPS = 1e-6
LENGTH_OUTPUT_WEIGHT = 0.25
N_SEALED_GROUPS = 24
CORRECTNESS_HEAD = "correct"
LENGTH_HEAD = "length"

SKIPPABLE_MAGIC = 0x184D2A5E
SEEKABLE_MAGIC = 0x8F92EAB1

Kind = Literal[
    "p0",
    "p1",
    "mixed",
    "continuous_const",
    "continuous_var",
    "length_penalty",
]


@dataclass(frozen=True)
class TaskSpec:
    env: str
    idx: int
    name: str
    kind: Kind
    n_groups: int
    n_pruned: int
    n_success: int = 0
    const: float = 0.0
    omit_key: bool = False


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("gsm8k", 0, "gsm8k-000", "p0", n_groups=2, n_pruned=2),
    TaskSpec("gsm8k", 1, "gsm8k-001", "p0", n_groups=2, n_pruned=2),
    TaskSpec("gsm8k", 2, "gsm8k-002", "p0", n_groups=2, n_pruned=2),
    TaskSpec("gsm8k", 3, "gsm8k-003", "p1", n_groups=2, n_pruned=2),
    TaskSpec("gsm8k", 4, "gsm8k-004", "p1", n_groups=2, n_pruned=2),
    TaskSpec("gsm8k", 5, "gsm8k-005", "p1", n_groups=2, n_pruned=2),
    TaskSpec("gsm8k", 6, "gsm8k-006", "mixed", n_groups=2, n_pruned=0, n_success=4),
    TaskSpec("gsm8k", 7, "gsm8k-007", "mixed", n_groups=2, n_pruned=0, n_success=4),
    TaskSpec("gsm8k", 8, "gsm8k-008", "mixed", n_groups=1, n_pruned=0, n_success=1),
    TaskSpec("gsm8k", 9, "gsm8k-009", "p0", n_groups=1, n_pruned=1, omit_key=True),
    TaskSpec(
        "countdown",
        0,
        "countdown-000",
        "continuous_const",
        n_groups=1,
        n_pruned=1,
        const=0.5,
    ),
    TaskSpec(
        "countdown",
        1,
        "countdown-001",
        "continuous_const",
        n_groups=1,
        n_pruned=1,
        const=0.5,
    ),
    TaskSpec(
        "countdown",
        2,
        "countdown-002",
        "continuous_const",
        n_groups=1,
        n_pruned=1,
        const=0.5,
    ),
    TaskSpec(
        "countdown",
        3,
        "countdown-003",
        "continuous_const",
        n_groups=1,
        n_pruned=1,
        const=0.5,
    ),
    TaskSpec(
        "countdown",
        4,
        "countdown-004",
        "continuous_const",
        n_groups=1,
        n_pruned=1,
        const=0.5,
    ),
    TaskSpec(
        "countdown",
        5,
        "countdown-005",
        "continuous_const",
        n_groups=1,
        n_pruned=1,
        const=0.5,
    ),
    TaskSpec(
        "countdown",
        6,
        "countdown-006",
        "continuous_const",
        n_groups=1,
        n_pruned=1,
        const=0.0,
    ),
    TaskSpec(
        "countdown",
        7,
        "countdown-007",
        "continuous_const",
        n_groups=1,
        n_pruned=1,
        const=0.0,
    ),
    TaskSpec("countdown", 8, "countdown-008", "continuous_var", n_groups=1, n_pruned=0),
    TaskSpec("countdown", 9, "countdown-009", "continuous_var", n_groups=1, n_pruned=0),
    TaskSpec("code", 0, "code-000", "length_penalty", n_groups=1, n_pruned=0),
    TaskSpec("code", 1, "code-001", "length_penalty", n_groups=1, n_pruned=0),
    TaskSpec("code", 2, "code-002", "length_penalty", n_groups=1, n_pruned=0),
    TaskSpec("code", 3, "code-003", "length_penalty", n_groups=1, n_pruned=0),
    TaskSpec("code", 4, "code-004", "p1", n_groups=1, n_pruned=1),
    TaskSpec("code", 5, "code-005", "p1", n_groups=1, n_pruned=1),
    TaskSpec("code", 6, "code-006", "p1", n_groups=1, n_pruned=1),
    TaskSpec("code", 7, "code-007", "mixed", n_groups=1, n_pruned=0, n_success=3),
    TaskSpec("code", 8, "code-008", "mixed", n_groups=1, n_pruned=0, n_success=3),
    TaskSpec("code", 9, "code-009", "mixed", n_groups=1, n_pruned=0, n_success=3),
)


@dataclass
class PlantedTrace:
    trace_id: str
    score: float
    reward: float
    n_out: int
    effective: bool
    advantage: float | None


@dataclass
class PlantedGroup:
    spec: TaskSpec
    group_id: str
    step: int
    traces: list[PlantedTrace]
    episode_ids: list[str] = field(default_factory=list)


def dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()


def content_hash(data: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def pop_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return var**0.5


def is_dead(values: list[float]) -> bool:
    return pop_std(values) < DEAD_EPS


def write_seekable_zst(src: bytes, dest: Path) -> None:
    """zstd level 3, independent frames of at most 4 MiB, seekable skippable footer.

    Mirrors prime-rl ``chunks.seal`` (pyzstd SeekableZstdFile, LEVEL=3,
    FRAME_BYTES=4 MiB). python-zstandard has no seekable API, so the framing is
    written here. A regular decompressor skips the footer.
    """
    cctx = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
    out = bytearray()
    frames: list[tuple[int, int]] = []
    offset = 0
    while True:
        chunk = src[offset : offset + FRAME_BYTES]
        compressed = cctx.compress(chunk)
        out.extend(compressed)
        frames.append((len(compressed), len(chunk)))
        offset += len(chunk)
        if offset >= len(src):
            break
    entries = b"".join(struct.pack("<II", csize, dsize) for csize, dsize in frames)
    footer = struct.pack("<IBI", len(frames), 0, SEEKABLE_MAGIC)
    payload = entries + footer
    dest.write_bytes(
        bytes(out) + struct.pack("<II", SKIPPABLE_MAGIC, len(payload)) + payload
    )


def scores_for(spec: TaskSpec) -> list[float]:
    if spec.kind == "p0":
        return [0.0] * GROUP_SIZE
    if spec.kind in ("p1", "length_penalty"):
        return [1.0] * GROUP_SIZE
    if spec.kind == "mixed":
        return [1.0] * spec.n_success + [0.0] * (GROUP_SIZE - spec.n_success)
    if spec.kind == "continuous_const":
        return [spec.const] * GROUP_SIZE
    if spec.kind == "continuous_var":
        return [0.1 * (i + 1) for i in range(GROUP_SIZE)]
    raise ValueError(spec.kind)


def output_lens_for(spec: TaskSpec) -> list[int]:
    if spec.kind == "length_penalty":
        return [10 + 2 * i for i in range(GROUP_SIZE)]
    return [8] * GROUP_SIZE


def rewards_for(spec: TaskSpec, scores: list[float], n_outs: list[int]) -> list[float]:
    if spec.kind != "length_penalty":
        return list(scores)
    max_out = max(n_outs)
    mean = sum(scores) / len(scores)
    return [
        score - mean * LENGTH_OUTPUT_WEIGHT * (n_out / max_out)
        for score, n_out in zip(scores, n_outs, strict=True)
    ]


def plant(rng: random.Random) -> list[PlantedGroup]:
    groups: list[PlantedGroup] = []
    g_index = 0
    for spec in TASKS:
        scores = scores_for(spec)
        n_outs = output_lens_for(spec)
        rewards = rewards_for(spec, scores, n_outs)
        mean_r = sum(rewards) / len(rewards)
        for g in range(spec.n_groups):
            pruned = g >= spec.n_groups - spec.n_pruned
            step = g_index % N_STEPS
            traces = [
                PlantedTrace(
                    trace_id=rng.randbytes(16).hex(),
                    score=scores[i],
                    reward=rewards[i],
                    n_out=n_outs[i],
                    effective=not pruned,
                    advantage=None if pruned else rewards[i] - mean_r,
                )
                for i in range(GROUP_SIZE)
            ]
            groups.append(
                PlantedGroup(
                    spec=spec,
                    group_id=str(uuid.UUID(bytes=rng.randbytes(16))),
                    step=step,
                    traces=traces,
                    episode_ids=[rng.randbytes(16).hex() for _ in range(GROUP_SIZE)],
                )
            )
            g_index += 1
    return groups


def task_payload(spec: TaskSpec) -> dict[str, Any]:
    data = {"idx": spec.idx, "name": spec.name, "prompt": f"q-{spec.env}-{spec.idx}"}
    payload: dict[str, Any] = {
        "type": "Task",
        "data": data,
        "hash": content_hash(data),
    }
    if not spec.omit_key:
        payload["key"] = f"{spec.env}/{spec.idx:03d}"
    return payload


def resolved_task_key(spec: TaskSpec) -> str:
    payload = task_payload(spec)
    key = payload.get("key")
    if isinstance(key, str):
        return key
    return str(payload["hash"])


def episode_record(
    group: PlantedGroup,
    member: int,
    trace: PlantedTrace,
    episode_id: str,
) -> dict[str, Any]:
    spec = group.spec
    t0 = BASE_TIME + group.step + member * 0.001
    n_out = trace.n_out
    token_ids = list(range(1, n_out + 2))
    task = task_payload(spec)
    rewards: dict[str, dict[str, float]] = {
        CORRECTNESS_HEAD: {"score": trace.score, "weight": 1.0},
    }
    if spec.kind == "length_penalty":
        rewards[LENGTH_HEAD] = {"score": trace.reward - trace.score, "weight": 1.0}
    run = {
        "type": "train",
        "id": "prime-run-small",
        "name": "prime-run-small",
        "work": {
            "type": "train",
            "step": group.step,
            "policy": {"start": group.step, "end": group.step, "drift": 0},
        },
    }
    info = {
        "kind": "train",
        "dispatch": {"step": group.step, "time": t0},
        "arrival": {"step": group.step, "time": t0 + 0.5},
    }
    usage = {"prompt_tokens": 1, "completion_tokens": n_out}
    node_user = {
        "parent": None,
        "message": {"role": "user", "content": task["data"]["prompt"]},
        "sampled": False,
        "timestamp": t0,
        "token_ids": [token_ids[0]],
        "mask": [False],
        "is_content": [True],
        "logprobs": [],
        "semantic_parents": [],
    }
    node_asst = {
        "parent": 0,
        "message": {"role": "assistant", "content": "a"},
        "sampled": True,
        "timestamp": t0 + 0.2,
        "token_ids": token_ids[1:],
        "mask": [True] * n_out,
        "is_content": [True] * n_out,
        "logprobs": [-0.1] * n_out,
        "semantic_parents": [],
    }
    tr: dict[str, Any] = {
        "version": 1,
        "id": trace.trace_id,
        "attempt": 1,
        "verifiers": {"version": "0.0.0+unknown", "commit": VERIFIERS_SHA},
        "task": task,
        "run": run,
        "agent": {
            "config": {"runtime": {"type": "prime"}, "network_notice": True},
            "name": "solver",
            "trainable": True,
        },
        "tools": [],
        "nodes": [node_user, node_asst],
        "calls": [
            {
                "node": 1,
                "usage": usage,
                "finish_reason": "stop",
                "time": {"start": t0, "end": t0 + 0.2, "duration": 0.2},
            }
        ],
        "mm_token_type_id_map": {},
        "request_rewrites": [],
        "response_rewrites": [],
        "rewards": rewards,
        "metrics": {},
        "info": info,
        "extra_usage": [],
        "is_completed": True,
        "ok": True,
        "stop_condition": "stop",
        "is_timeout": False,
        "errors": [],
        "timing": {
            "start": t0,
            "boot": {"start": t0, "end": t0, "duration": 0.0},
            "setup": {"start": t0, "end": t0, "duration": 0.0},
            "agent": {
                "start": t0,
                "end": t0 + 0.2,
                "duration": 0.2,
                "model": {"duration": 0.2},
                "harness": {"duration": 0.0},
            },
            "finalize": {"start": t0 + 0.2, "end": t0 + 0.2, "duration": 0.0},
            "scoring": {"start": t0 + 0.2, "end": t0 + 0.5, "duration": 0.3},
        },
        "reward": trace.reward,
        "is_format_violation": False,
        "has_error": False,
        "num_input_tokens": 1,
        "num_output_tokens": n_out,
        "num_total_tokens": 1 + n_out,
        "num_turns": 1,
        "num_branches": 1,
        "is_truncated": False,
        "usage": usage,
    }
    return {
        "id": episode_id,
        "attempt": 1,
        "env": {"id": spec.env, "name": spec.env},
        "task": task,
        "group": {"id": group.group_id},
        "run": run,
        "ok": True,
        "is_timeout": False,
        "errors": [],
        "traces": [tr],
        "usage": usage,
        "num_input_tokens": 1,
        "num_output_tokens": n_out,
        "num_total_tokens": 1 + n_out,
        "num_turns": 1,
    }


def index_row(
    line: int, rec: dict[str, Any], chunk: int, offset: int
) -> dict[str, Any]:
    """The row ``index_row`` writes. Nested reward/metric/timing maps are omitted."""
    traces = rec.get("traces") or []
    rewards = [
        t["reward"]
        for t in traces
        if t.get("rewards") and isinstance(t.get("reward"), (int, float))
    ]
    first_info = (traces[0].get("info") if traces else {}) or {}
    dispatch = (first_info.get("dispatch") or {}).get("time")
    arrival = (first_info.get("arrival") or {}).get("time")
    duration = None
    if isinstance(dispatch, (int, float)) and isinstance(arrival, (int, float)):
        duration = max(0.0, arrival - dispatch)
    return {
        "cost": (rec.get("usage") or {}).get("cost"),
        "line": line,
        "offset": offset,
        "id": rec.get("id"),
        "kind": first_info.get("kind") or "eval",
        "trace_ids": [t["id"] for t in traces if t.get("id")],
        "env": (rec.get("env") or {}).get("name") or (rec.get("env") or {}).get("id"),
        "group": (rec.get("group") or {}).get("id"),
        "ok": rec.get("ok"),
        "num_errors": len(rec.get("errors") or []),
        "reward": sum(rewards) / len(rewards) if rewards else None,
        "advantage": None,
        "input_tokens": rec.get("num_input_tokens"),
        "output_tokens": rec.get("num_output_tokens"),
        "turns": rec.get("num_turns"),
        "branches": sum(t.get("num_branches") or 0 for t in traces),
        "stop_condition": traces[-1].get("stop_condition") if traces else None,
        "truncated": any(t.get("is_truncated") for t in traces),
        "timeout": bool(rec.get("is_timeout"))
        or any(t.get("is_timeout") for t in traces),
        "dispatch": dispatch,
        "arrival": arrival,
        "duration": duration,
        "chunk": chunk,
    }


def annotation_record(trace: PlantedTrace, group: PlantedGroup) -> dict[str, Any]:
    t0 = BASE_TIME + group.step + 1.0
    info: dict[str, Any] = {
        "effective": True,
        "ship": {"step": group.step, "time": t0},
    }
    if trace.advantage is not None:
        info["advantage"] = trace.advantage
    return {
        "version": 1,
        "trace_id": trace.trace_id,
        "info": info,
        "branches": [],
    }


def expected_from(groups: list[PlantedGroup]) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    families: dict[str, dict[str, int]] = {}
    for spec in TASKS:
        planted = [g for g in groups if g.spec is spec]
        traces = [t for g in planted for t in g.traces]
        scores = [t.score for t in traces]
        k = sum(1 for s in scores if s == 1.0)
        dead_score = sum(1 for g in planted if is_dead([t.score for t in g.traces]))
        dead_reward = sum(1 for g in planted if is_dead([t.reward for t in g.traces]))
        effective = sum(1 for g in planted if any(t.effective for t in g.traces))
        tasks.append(
            {
                "env": spec.env,
                "task_key": resolved_task_key(spec),
                "kind": spec.kind,
                "n": len(traces),
                "k": k,
                "dead_on_score": dead_score,
                "dead_on_reward": dead_reward,
            }
        )
        fam = families.setdefault(
            spec.env,
            {
                "family": spec.env,
                "groups": 0,
                "dead_on_score": 0,
                "dead_on_reward": 0,
                "effective": 0,
            },
        )
        fam["groups"] += len(planted)
        fam["dead_on_score"] += dead_score
        fam["dead_on_reward"] += dead_reward
        fam["effective"] += effective
    return {
        "meta": {
            "seed": SEED,
            "group_size": GROUP_SIZE,
            "n_steps": N_STEPS,
            "prime_rl_sha": PRIME_RL_SHA,
            "verifiers_sha": VERIFIERS_SHA,
            "n_episodes": sum(len(g.traces) for g in groups),
            "n_groups": len(groups),
            "sealed_chunk": "00000.jsonl.zst",
            "live_chunk": "00001.jsonl",
            "truncated_last_line": True,
            "n_sealed_groups": N_SEALED_GROUPS,
            "correctness_head": CORRECTNESS_HEAD,
            "dead_eps": DEAD_EPS,
            "length_output_weight": LENGTH_OUTPUT_WEIGHT,
        },
        "tasks": tasks,
        "families": [families[name] for name in ("gsm8k", "countdown", "code")],
    }


def _write_stream(
    records: list[dict[str, Any]],
    stream_dir: Path,
    index_path: Path,
    *,
    seal: bool,
) -> None:
    stream_dir.mkdir(parents=True, exist_ok=True)
    parts: list[bytes] = []
    rows: list[bytes] = []
    offset = 0
    chunk = 0 if seal else 1
    start_line = 1 if seal else 1 + N_SEALED_GROUPS * GROUP_SIZE
    for i, rec in enumerate(records):
        line = dumps(rec)
        rows.append(dumps(index_row(start_line + i, rec, chunk, offset)) + b"\n")
        parts.append(line + b"\n")
        offset += len(line) + 1
    blob = b"".join(parts)
    if seal:
        write_seekable_zst(blob, stream_dir / "00000.jsonl.zst")
    else:
        live = stream_dir / "00001.jsonl"
        live.write_bytes(blob + b'{"id":"truncated-incomplete-line"')
    index_path.write_bytes(b"".join(rows))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def generate(
    dest: Path, expected_path: Path | None = None, seed: int = SEED
) -> dict[str, Any]:
    rng = random.Random(seed)
    groups = plant(rng)
    episodes: list[dict[str, Any]] = []
    for group in groups:
        for i, trace in enumerate(group.traces):
            episodes.append(episode_record(group, i, trace, group.episode_ids[i]))

    if dest.exists():
        shutil.rmtree(dest)
    monitor = dest / "monitors" / "file"
    stream = monitor / "traces" / "stream"
    stream.parent.mkdir(parents=True, exist_ok=True)

    sealed_eps = episodes[: N_SEALED_GROUPS * GROUP_SIZE]
    live_eps = episodes[N_SEALED_GROUPS * GROUP_SIZE :]
    index_path = monitor / "traces" / "stream.index.jsonl"
    _write_stream(sealed_eps, stream, index_path, seal=True)
    live_index = b"".join(
        dumps(
            index_row(
                1 + N_SEALED_GROUPS * GROUP_SIZE + i,
                rec,
                1,
                sum(len(dumps(e)) + 1 for e in live_eps[:i]),
            )
        )
        + b"\n"
        for i, rec in enumerate(live_eps)
    )
    live_blob = b"".join(dumps(rec) + b"\n" for rec in live_eps)
    stream.mkdir(parents=True, exist_ok=True)
    (stream / "00001.jsonl").write_bytes(
        live_blob + b'{"id":"truncated-incomplete-line"'
    )
    index_path.write_bytes(index_path.read_bytes() + live_index)

    ann_dir = monitor / "traces" / "annotations" / "orchestrator"
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_index = monitor / "traces" / "annotations" / "orchestrator.index.jsonl"
    ann_parts: list[bytes] = []
    ann_rows: list[bytes] = []
    offset = 0
    for group in groups:
        for trace in group.traces:
            if not trace.effective:
                continue
            rec = annotation_record(trace, group)
            line = dumps(rec) + b"\n"
            ann_rows.append(
                dumps(
                    {
                        "trace_id": rec["trace_id"],
                        "chunk": 0,
                        "offset": offset,
                        "info": rec["info"],
                    }
                )
                + b"\n"
            )
            ann_parts.append(line)
            offset += len(line)
    (ann_dir / "00000.jsonl").write_bytes(b"".join(ann_parts))
    ann_index.write_bytes(b"".join(ann_rows))

    metrics_lines: list[bytes] = []
    by_step: dict[int, list[PlantedGroup]] = {}
    for group in groups:
        by_step.setdefault(group.step, []).append(group)
    for step in range(N_STEPS):
        step_groups = by_step.get(step, [])
        pruned = sum(1 for g in step_groups if not any(t.effective for t in g.traces))
        shipped = [g for g in step_groups if any(t.effective for t in g.traces)]
        row: dict[str, Any] = {
            "step": step,
            "time": BASE_TIME + step,
            "producer": "orchestrator",
            "rollout/queued/pruned": float(pruned * GROUP_SIZE),
            "rollout/no_signal": float(pruned * GROUP_SIZE),
            "rollout/attempts": float(len(step_groups) * GROUP_SIZE),
        }
        n_ship = sum(len(g.traces) for g in shipped) or 1
        for env in ("gsm8k", "countdown", "code"):
            n_env = sum(len(g.traces) for g in shipped if g.spec.env == env)
            row[f"batch/{env}"] = n_env / n_ship if shipped else 0.0
        metrics_lines.append(dumps(row) + b"\n")
    (monitor / "metrics.jsonl").write_bytes(b"".join(metrics_lines))

    expected = expected_from(groups)
    checksums: dict[str, str] = {}
    for path in sorted(dest.rglob("*")):
        if path.is_file():
            checksums[path.relative_to(dest).as_posix()] = sha256_file(path)
    expected["checksums"] = checksums
    if expected_path is not None:
        expected_path.write_bytes(dumps(expected) + b"\n")
    return expected


def default_paths() -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    return here / "prime_run_small", here / "expected_small.json"


def main() -> None:
    dest, expected_path = default_paths()
    generate(dest, expected_path=expected_path)
    print(f"wrote {dest}")
    print(f"wrote {expected_path}")


if __name__ == "__main__":
    main()
