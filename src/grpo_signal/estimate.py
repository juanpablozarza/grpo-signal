"""Per-task and per-family signal estimators."""

from collections.abc import Sequence


def dead_rate_binary(k: int, n: int, G: int) -> float:
    """Expected dead-group rate for a binary task with ``k`` successes in ``n``.

    Posterior on ``p`` is Beta(``k+1``, ``n-k+1``). The expectation
    ``E[p^G + (1-p)^G] = (B(a+G, b) + B(a, b+G)) / B(a, b)`` is computed in
    log space.
    """
    raise NotImplementedError("dead_rate_binary")


def dead_rate_subsets(
    rewards: Sequence[float],
    G: int,
    eps: float = 1e-6,
    max_subsets: int = 20_000,
    seed: int = 0,
) -> float:
    """Share of ``G``-subsets whose reward std is below ``eps``.

    Subsets are drawn without replacement (needs ``N >= G``). Enumerate all
    when ``C(N, G) <= max_subsets``, otherwise Monte Carlo with ``seed``.
    """
    raise NotImplementedError("dead_rate_subsets")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval on a binomial proportion, for display."""
    raise NotImplementedError("wilson")


def signal_weight(p: float, G: int, normalize_std: bool) -> float:
    """Expected group weight at success rate ``p``.

    Mean-only (prime-rl): ``E[sum |A|] = 2(G-1)p(1-p)``. With std division,
    each live group weighs about the same.
    """
    raise NotImplementedError("signal_weight")
