# grpo-signal

Looks at rollout traces from a GRPO training run (prime-rl or verifiers) and
estimates how much of each task family will actually produce a training
signal. When a family is too easy or too hard, every rollout in a group can
get the same reward and get thrown away. This tool estimates how often that
happens and suggests prime-rl config changes.

## Status

Scaffold. Package layout, types, CLI shape, and checks are in place.
Reading traces, estimation, simulation, and reports are not implemented yet.

## Setup

Python 3.11 or newer. [uv](https://docs.astral.sh/uv/) is required:

```bash
uv sync
```

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

## CLI

```bash
uv run grpo-signal --help
```
