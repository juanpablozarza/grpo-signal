# Synthetic trace fixtures

`prime_run_small/` is a tiny fake training-run directory. It uses the same
on-disk layout prime-rl writes under `monitors/file/`. Field names follow
`docs/FORMAT.md`.

Regenerate (deterministic, seed 0):

```bash
uv run python tests/fixtures/gen_prime_run.py
```

That also rewrites `expected_small.json`, which holds per-task and per-family
counts computed from the generator's own parameters. Reader tests compare
against that file, not against an implementation of the estimator.

## prime_run_small

```
tests/fixtures/prime_run_small/
  monitors/file/
    metrics.jsonl
    traces/stream/00000.jsonl.zst    # sealed chunk (zstd level 3, seekable 4 MiB frames)
    traces/stream/00001.jsonl        # live chunk; last line is truncated
    traces/stream.index.jsonl
    traces/annotations/orchestrator/00000.jsonl
    traces/annotations/orchestrator.index.jsonl
```

What is planted:

- 3 environments (`gsm8k`, `countdown`, `code`), 30 tasks, group size 8, 20 steps
- tasks that always fail (`p=0`), always pass (`p=1`), mixed pass/fail, and
  continuous rewards that are not 0/1
- all-correct groups whose correctness score is constant but whose final reward
  varies (length-penalty case)
- groups that appear in the stream with no `effective` annotation (never
  trained), and groups with `effective: true` plus `ship.step`
- one task with no `task.key`, so the reader must fall back to `task.hash`

`stream.index.jsonl` has no task key and no per-component rewards. Group by
task using the stream.

## vf-run file

Not in this fixture set. A vf-run `traces.jsonl` is the same Episode JSON, one
line per episode, uncompressed. Add it when the vf-run reader is filled in.

## Family map

Not in this fixture set. Default family is the environment name.
