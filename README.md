# grpo-signal

Looks at rollout traces from a GRPO training run (prime-rl or verifiers) and
estimates how much of each task family will actually produce a training
signal. When a family is too easy or too hard, every rollout in a group can
get the same reward and get thrown away. This tool estimates how often that
happens and suggests prime-rl config changes.

## Status

Package layout, types, CLI shape, and a synthetic prime-rl run fixture are in
place. Reading traces, estimation, simulation, and reports are not implemented
yet. Reader tests skip until those functions are filled in.

The on-disk format the readers will consume is described in `docs/FORMAT.md`.
A small fake run lives at `tests/fixtures/prime_run_small/`.

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

Rebuild the fake run (same output every time):

```bash
uv run python tests/fixtures/gen_prime_run.py
```

## CLI

```bash
uv run grpo-signal --help
```
