# Synthetic trace fixtures

Not generated yet. Milestone 2 should add planted traces here so the estimator
can recover a known pass rate within its interval, and so the readers can be
checked against both file layouts.

## prime-rl run directory

Layout matches the file-monitor output under
`<output_dir>/<run.dir>/monitors/file/`:

```
tests/fixtures/prime_run/
  monitors/file/
    metrics.jsonl
    traces/stream/00000.jsonl          # live / unsealed chunk
    traces/stream/00001.jsonl.zst      # sealed chunk (zstd, seekable)
    traces/stream.index.jsonl
    traces/annotations/orchestrator/00000.jsonl
    traces/annotations/orchestrator.index.jsonl
```

- Stream chunks: `00000.jsonl`, `00001.jsonl`, … A finished chunk is sealed as
  `NNNNN.jsonl.zst` and the plain file is removed. The live chunk stays plain.
  A JSON line never spans chunks.
- Each stream line is one Episode JSON (`id`, `env`, `task`, `group`, `run`,
  `ok`, `traces[]`). Use the stream to group by task key; the index is only a
  fast path for env/group/reward.
- Annotation records: `{"version": 1, "trace_id", "info", "branches": [...]}`.
  A train trace with no `effective` annotation never trained.

Plant at least two families with known binary pass rates (include `p=0.9`,
which is dead about 43% of the time at `G=8`) and one family with a constant
non-0/1 reward.

## vf-run file

```
tests/fixtures/vf_run/traces.jsonl
```

Same Episode JSON, one episode per line, uncompressed.

## Family map

```
tests/fixtures/families.csv
```

Header `task_key,family`. Used by `--family-map`.
