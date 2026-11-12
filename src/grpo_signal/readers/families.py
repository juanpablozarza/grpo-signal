"""Map rollouts onto task families."""

from grpo_signal.types import FamilySpec, Rollout


def assign_family(r: Rollout, spec: FamilySpec) -> str:
    """Return the family name for a rollout under ``spec``.

    Default is the environment name. Other specs use a TaskData field, the task
    name, a regex, or a CSV of ``task_key,family``.
    """
    raise NotImplementedError("assign_family")
