"""Copy of prime-rl sampler and batch-refill mix."""

from grpo_signal.types import FamilyStats, MixResult, PrimeConfig


def simulate_mix(
    families: list[FamilyStats],
    cfg: PrimeConfig,
    steps: int,
    seed: int = 42,
) -> MixResult:
    """Predict each family's share of the trainer batch under ``cfg``.

    Replays env ``ratio`` draws, StandardSampler / DifficultyPoolSampler, and
    constant-batch refill. Analytic check: share is proportional to
    ``ratio * G * live``.
    """
    raise NotImplementedError("simulate_mix")
