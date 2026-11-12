"""Config diff and predicted metrics."""

from grpo_signal.types import ConfigDiff, Mix, Report


def recommend(report: Report, target: Mix | None = None) -> ConfigDiff:
    """Suggest prime-rl config changes from a report.

    Uses existing keys only (group size, source ratios, select include/exclude,
    difficulty-pool sampler, advantage-range gates, batch size, constant batch).
    If ``target`` is set, put each family in its own source with ratio
    proportional to ``target / (G * live)``.
    """
    raise NotImplementedError("recommend")
