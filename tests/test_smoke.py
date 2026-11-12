from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import grpo_signal
import grpo_signal.cli
import grpo_signal.estimate
import grpo_signal.recommend
import grpo_signal.report
import grpo_signal.simulate
import grpo_signal.types
from grpo_signal.estimate import (
    dead_rate_binary,
    dead_rate_subsets,
    signal_weight,
    wilson,
)
from grpo_signal.readers import (
    assign_family,
    iter_chunked_jsonl,
    read_prime_run,
    read_vf_traces,
)
from grpo_signal.recommend import recommend
from grpo_signal.report import render_terminal, write_html, write_json
from grpo_signal.simulate import simulate_mix
from grpo_signal.types import (
    FamilySpec,
    FamilyStats,
    Mix,
    PrimeConfig,
    Report,
    Rollout,
    TaskStats,
)


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    exe = Path(sys.executable).parent / "grpo-signal"
    return subprocess.run(
        [str(exe), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _rollout() -> Rollout:
    return Rollout(
        env="env",
        task_key="t0",
        task_idx=0,
        family="env",
        group_id="g0",
        reward=1.0,
        rewards={"score": 1.0},
        shipped=True,
        step=0,
    )


def _task() -> TaskStats:
    return TaskStats(
        env="env",
        task_key="t0",
        family="env",
        n=16,
        k=8,
        p_hat=0.5,
        p_low=0.28,
        p_high=0.72,
        reward_mean=0.5,
        reward_std=0.5,
        dead_rate={8: 0.0},
        constant_non_binary=False,
    )


def _family() -> FamilyStats:
    return FamilyStats(
        family="env",
        n_tasks=1,
        n_rollouts=16,
        dataset_share=1.0,
        gradient_share_mean_only=1.0,
        gradient_share_std_div=1.0,
        p_hat=0.5,
        p_low=0.28,
        p_high=0.72,
        reward_std=0.5,
        dead_rate={4: 0.0, 8: 0.0, 16: 0.0},
        wasted_rollout_fraction=0.0,
        constant_non_binary=False,
    )


def _report() -> Report:
    return Report(
        source="test",
        group_sizes=[8],
        tasks=[_task()],
        families=[_family()],
    )


def test_package_imports() -> None:
    assert grpo_signal.__version__ == "0.1.0"
    assert grpo_signal.cli.__name__ == "grpo_signal.cli"
    assert grpo_signal.estimate.__name__ == "grpo_signal.estimate"
    assert grpo_signal.recommend.__name__ == "grpo_signal.recommend"
    assert grpo_signal.report.__name__ == "grpo_signal.report"
    assert grpo_signal.simulate.__name__ == "grpo_signal.simulate"
    assert grpo_signal.types.__name__ == "grpo_signal.types"


def test_cli_help() -> None:
    result = _cli("--help")
    assert result.returncode == 0
    assert "audit" in result.stdout
    assert "simulate" in result.stdout
    assert "card" in result.stdout


def test_cli_subcommands_not_implemented() -> None:
    audit = _cli("audit")
    assert audit.returncode == 1
    assert "not implemented yet" in audit.stdout

    simulate = _cli("simulate", "report.json")
    assert simulate.returncode == 1
    assert "not implemented yet" in simulate.stdout

    card = _cli("card", "report.json")
    assert card.returncode == 1
    assert "not implemented yet" in card.stdout


def test_stubs_raise_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        read_prime_run(Path("."))
    with pytest.raises(NotImplementedError):
        read_vf_traces(Path("traces.jsonl"))
    with pytest.raises(NotImplementedError):
        iter_chunked_jsonl(Path("."))
    assert assign_family(_rollout(), FamilySpec()) == "env"
    with pytest.raises(NotImplementedError):
        dead_rate_binary(8, 16, 8)
    with pytest.raises(NotImplementedError):
        dead_rate_subsets([0.0, 1.0], 2)
    with pytest.raises(NotImplementedError):
        wilson(8, 16)
    with pytest.raises(NotImplementedError):
        signal_weight(0.5, 8, False)
    with pytest.raises(NotImplementedError):
        simulate_mix([_family()], PrimeConfig(), steps=1)
    with pytest.raises(NotImplementedError):
        recommend(_report(), Mix(shares={"env": 1.0}))
    with pytest.raises(NotImplementedError):
        write_json(_report(), Path("report.json"))
    with pytest.raises(NotImplementedError):
        write_html(_report(), Path("report.html"))
    with pytest.raises(NotImplementedError):
        render_terminal(_report())
