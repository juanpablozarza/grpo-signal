"""Domain types for traces, stats, config plans, and recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict


class EnvRef(TypedDict, total=False):
    """Environment name/id as stored on an episode."""

    name: str
    id: str


class TaskPayload(TypedDict, total=False):
    """verifiers TaskData fields used to group rollouts."""

    type: str
    data: dict[str, object]
    key: str
    hash: str


class GroupRef(TypedDict, total=False):
    """Dispatcher group that GRPO compares."""

    id: str


class RewardComponent(TypedDict):
    """Named reward head: score times weight sums to the trace reward."""

    score: float
    weight: float


class TraceRow(TypedDict, total=False):
    """One on-disk trace inside a vf.Episode."""

    rewards: dict[str, RewardComponent]
    metrics: dict[str, object]
    info: dict[str, object]
    reward: float
    ok: bool
    dropped: bool


class EpisodeRow(TypedDict, total=False):
    """One on-disk vf.Episode (prime-rl stream line or vf-run traces.jsonl)."""

    id: str
    env: str | EnvRef
    task: TaskPayload
    group: GroupRef
    run: dict[str, object]
    ok: bool
    traces: list[TraceRow]


DEAD_EPS = 1e-6
"""Std below this counts as a dead group. prime-rl prunes on advantage == 0.0."""

CORRECTNESS_HEAD = "correct"
"""Named reward head used as the pass/fail score when present.

prime-rl tests plant ``rewards={"correct": Reward(score=...)}``. Final reward is
always ``Trace.reward`` (sum of score × weight), which differs from score when
another head is present or a weight is not 1.
"""


@dataclass(kw_only=True)
class TraceRecord:
    """One trace after joining the episode stream with ship-time annotations."""

    trace_id: str
    score: float
    """Correctness score. ``rewards["correct"].score`` when that head exists,
    otherwise the first named head's score, otherwise ``trace.reward``."""
    reward: float
    """Final reward: the on-disk ``trace.reward`` computed field
    (sum of each head's score × weight)."""
    step: int | None
    """``info.ship.step`` when effective, else ``info.arrival.step`` /
    ``info.dispatch.step``."""
    effective: bool
    """True when an annotation record has ``info.effective`` true. A train
    trace with no such annotation never trained."""


@dataclass(kw_only=True)
class GroupRecord:
    """The G traces that share a dispatcher group id (one GRPO comparison).

    Deadness is reported on both metrics. A group is dead on a metric when the
    population std of that metric across ``traces`` is below ``DEAD_EPS``.
    """

    env: str
    """``episode.env.name``, falling back to ``episode.env.id``."""
    task_key: str
    """``episode.task.key``, falling back to ``episode.task.hash``."""
    group_id: str
    """``episode.group.id`` (dispatcher stamps ``str(group_uuid)``)."""
    traces: list[TraceRecord]


@dataclass(kw_only=True)
class Rollout:
    """One scored attempt, after the reader has flattened an episode/trace."""

    env: str
    task_key: str
    task_idx: int | None
    family: str
    group_id: str
    reward: float
    rewards: dict[str, float]
    shipped: bool | None
    step: int | None


@dataclass(kw_only=True)
class FamilySpec:
    """How to map a rollout to a family name.

    Default ``kind='env'`` is one family per environment name. ``field`` reads a
    TaskData key (``--family-field``). ``csv`` loads a ``task_key,family`` table
    (``--family-map``). ``name`` and ``regex`` are extra selectors.
    """

    kind: Literal["env", "field", "name", "regex", "csv"] = "env"
    field: str | None = None
    pattern: str | None = None
    csv_path: Path | None = None


@dataclass(kw_only=True)
class TaskStats:
    """Per-task pass rate, reward spread, and predicted dead-group rates."""

    env: str
    task_key: str
    family: str
    n: int
    k: int
    p_hat: float
    p_low: float
    p_high: float
    reward_mean: float
    reward_std: float
    dead_rate: dict[int, float]
    constant_non_binary: bool


@dataclass(kw_only=True)
class FamilyStats:
    """Per-family mix, signal, and dead rates at G = 4/8/16."""

    family: str
    n_tasks: int
    n_rollouts: int
    dataset_share: float
    gradient_share_mean_only: float
    gradient_share_std_div: float
    p_hat: float
    p_low: float
    p_high: float
    reward_std: float
    dead_rate: dict[int, float]
    wasted_rollout_fraction: float
    constant_non_binary: bool


@dataclass(kw_only=True)
class Report:
    """Stable report schema written as report.json."""

    source: str
    group_sizes: list[int]
    tasks: list[TaskStats]
    families: list[FamilyStats]


@dataclass(kw_only=True)
class SelectKeys:
    """Task-key include/exclude lists for a train source."""

    include: list[str] = field(default_factory=list[str])
    exclude: list[str] = field(default_factory=list[str])


@dataclass(kw_only=True)
class TrainSource:
    """One ``[[orchestrator.train.source]]`` entry."""

    name: str
    ratio: float
    group_size: int | None = None
    select: SelectKeys | None = None


@dataclass(kw_only=True)
class SamplerPool:
    """One difficulty-pool bucket: ``pools.<name>.threshold`` / ``weight``."""

    name: str
    threshold: float
    weight: float


@dataclass(kw_only=True)
class AdvantageRangeGate:
    """``curriculum.gates.<name>`` with ``type='advantage_range'``."""

    name: str
    reject_min: float = 0.0
    reject_max: float = 0.0
    type: Literal["advantage_range"] = "advantage_range"


@dataclass(kw_only=True)
class PrimeConfig:
    """prime-rl settings that decide which groups reach the trainer batch."""

    algorithm: Literal["grpo", "max_rl"] = "grpo"
    group_size: int = 1
    batch_size: int = 128
    constant_trainer_batch_size: bool = True
    sources: list[TrainSource] = field(default_factory=list[TrainSource])
    sampler_type: Literal["standard", "difficulty_pool"] | None = None
    pools: list[SamplerPool] = field(default_factory=list[SamplerPool])
    gates: list[AdvantageRangeGate] = field(default_factory=list[AdvantageRangeGate])


@dataclass(kw_only=True)
class Mix:
    """Target share of the trainer batch per family (values should sum to 1)."""

    shares: dict[str, float]


@dataclass(kw_only=True)
class MixResult:
    """Simulated trainer-batch mix."""

    family_share: dict[str, float]
    steps_per_pass: float
    rollouts_per_step: float


@dataclass(kw_only=True)
class ConfigDiff:
    """Recommended prime-rl config changes (existing keys only) as TOML."""

    toml: str
    predicted_metrics: dict[str, float] = field(default_factory=dict[str, float])
    notes: list[str] = field(default_factory=list[str])
