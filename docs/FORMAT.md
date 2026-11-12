# On-disk format: prime-rl file monitor and verifiers Episode

Reference for the files under a run's `monitors/file/` directory, and for the
Episode JSON that each stream line holds. Field names and layout are taken from
the sources below. Nothing here is inferred from docs alone.

Pinned trees:

- prime-rl [`5af925d4b1568559607a7b42b3679d9edf3d39de`](https://github.com/PrimeIntellect-ai/prime-rl/tree/5af925d4b1568559607a7b42b3679d9edf3d39de)
- verifiers [`7407a3ea4f4f5e720bd5acfde007a41a33bb65a1`](https://github.com/PrimeIntellect-ai/verifiers/tree/7407a3ea4f4f5e720bd5acfde007a41a33bb65a1)

`docs/PLAN.md` cited prime-rl `bd326dca` and the same verifiers SHA. The writers
below are the same files at the newer prime-rl SHA. Differences that matter for
readers are called out at the end.

Prefix `P` is
`https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/`.
Prefix `V` is
`https://github.com/PrimeIntellect-ai/verifiers/blob/7407a3ea4f4f5e720bd5acfde007a41a33bb65a1/`.

## Where the files live

The file monitor is on by default
([P/packages/prime-rl-configs/src/prime_rl/configs/monitors.py#L55-L56](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/packages/prime-rl-configs/src/prime_rl/configs/monitors.py#L55-L56)).
Everything it writes sits under `<output_dir>/monitors/file/`
([P/src/prime_rl/utils/pathing.py#L309-L312](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/utils/pathing.py#L309-L312)).

```
<run_dir>/monitors/file/
  metrics.jsonl
  traces/stream/NNNNN.jsonl          live chunk
  traces/stream/NNNNN.jsonl.zst      sealed chunk
  traces/stream.index.jsonl          sibling of the stream directory
  traces/annotations/<producer>/NNNNN.jsonl[.zst]
  traces/annotations/<producer>.index.jsonl
```

Path helpers:
[P/src/prime_rl/monitors/file/traces/__init__.py#L13-L30](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/__init__.py#L13-L30).
The index path is `stream.with_name(stream.name.removesuffix(".jsonl") + ".index.jsonl")`.
For a directory named `stream`, that is the sibling `stream.index.jsonl`. Same
pattern for `annotations/orchestrator` → `annotations/orchestrator.index.jsonl`.

## Chunked JSONL

[P/src/prime_rl/monitors/file/traces/chunks.py](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/chunks.py)
is the writer.

- Chunk files are `{number:05d}.jsonl` ([L26-L27](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/chunks.py#L26-L27)).
- A line never spans chunks ([L6-L8](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/chunks.py#L6-L8)).
- Default roll size is 5 GiB, `compress=true`
  ([P/packages/prime-rl-configs/src/prime_rl/configs/monitors.py#L31-L38](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/packages/prime-rl-configs/src/prime_rl/configs/monitors.py#L31-L38)).
- A finished chunk is sealed beside itself as `NNNNN.jsonl.zst`, then the plain
  file is removed ([L48-L58](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/chunks.py#L48-L58)).
- Seal uses zstd **level 3** and **seekable frames of 4 MiB**
  (`LEVEL = 3`, `FRAME_BYTES = 4 << 20`, [L19-L23](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/chunks.py#L19-L23);
  `pyzstd.SeekableZstdFile(..., level_or_option=LEVEL, max_frame_content_size=FRAME_BYTES)`).
- The live chunk stays plain so it can be tailed ([L3-L4](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/chunks.py#L3-L4)).
- `append` writes the record bytes then `b"\n"`
  ([L81-L91](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/chunks.py#L81-L91)).
  A crash mid-write can leave a truncated last line with no trailing newline.
  Readers drop that tail.
- `open_chunk` opens the plain file, or `SeekableZstdFile` on `path.name + ".zst"`
  if the plain file is gone ([L39-L45](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/chunks.py#L39-L45)).
  Index offsets are uncompressed byte offsets in that view.

`zstd -dcf stream/* | jq` reads sealed and live chunks together
([L9](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/chunks.py#L9)).
A regular zstd decoder skips the seekable-format skippable footer, so
`zstandard.ZstdDecompressor` can read a whole sealed file as one stream.

## metrics.jsonl

[P/src/prime_rl/monitors/file/monitor.py#L64-L76](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/monitor.py#L64-L76).

One JSON object per `log_metrics` call, newline-terminated.

| Field | Rule |
| --- | --- |
| `step` | Training step, or `null` for time-keyed rows (inference metrics). |
| `time` | `time.time()` wall clock. |
| `producer` | Set when the monitor was inited with a producer (`orchestrator`, `trainer`, `eval`, `online-eval`). |
| other keys | The sanitized metric dict. Non-finite floats are dropped before write. |

Default path is `metrics.jsonl` relative to `monitors/file/`
([monitors.py L27-L29](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/packages/prime-rl-configs/src/prime_rl/configs/monitors.py#L27-L29)).

## traces/stream: one Episode per line

Every completed episode is serialized **once**, in arrival order, whatever its
fate (errored, rejected, dead, or shipped)
([P/src/prime_rl/monitors/file/monitor.py#L151-L178](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/monitor.py#L151-L178)).
The line is `episode.to_json(float_decimals=...)`
([L168](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/monitor.py#L168)).
Default `float_decimals` is 4
([monitors.py L40-L43](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/packages/prime-rl-configs/src/prime_rl/configs/monitors.py#L40-L43)).
`subset == "effective"` writes nothing here. Shipped membership arrives later as
annotations.

`to_json` is `to_record` as compact UTF-8 JSON with `exclude_none=True`
([V/verifiers/v1/episode.py#L121-L174](https://github.com/PrimeIntellect-ai/verifiers/blob/7407a3ea4f4f5e720bd5acfde007a41a33bb65a1/verifiers/v1/episode.py#L121-L174)).
NaN and infinity become `null`.

### Episode fields

[V/verifiers/v1/episode.py#L30-L69](https://github.com/PrimeIntellect-ai/verifiers/blob/7407a3ea4f4f5e720bd5acfde007a41a33bb65a1/verifiers/v1/episode.py#L30-L69).

| Field | Meaning |
| --- | --- |
| `id` | `uuid4().hex` (32 hex chars). |
| `attempt` | Attempt number, from 1. |
| `env` | `{id, name}`. `id` is `EnvConfig.env_id`. `name` is the orchestrator env key. |
| `task` | `TraceTask` (see below). Present even when no trace was produced. |
| `group` | `{id}` or omitted. Dispatcher sets `id=str(meta.group_id)` ([P/src/prime_rl/orchestrator/dispatcher.py#L784](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/orchestrator/dispatcher.py#L784)). |
| `run` | `TrainRunInfo`: `{type:"train", id, name?, work:{type:"train"\|"eval", step, policy?}}`. |
| `ok` | Episode completed successfully. |
| `is_timeout` | Env `run()`/`finalize()` deadline expired. |
| `errors` | `{type, message, status_code?, traceback?}`. |
| `traces` | Agent traces, completion order. |
| `usage`, `num_*` | Computed sums across traces. |

### TraceTask (episode.task and trace.task)

[V/verifiers/v1/trace.py#L298-L308](https://github.com/PrimeIntellect-ai/verifiers/blob/7407a3ea4f4f5e720bd5acfde007a41a33bb65a1/verifiers/v1/trace.py#L298-L308).

| Field | Meaning |
| --- | --- |
| `type` | Task class name (`type(task).__name__`). |
| `data` | Frozen task row (`TaskData` plus extra keys on the wire). |
| `key` | Stable identity. Default is sha256 of `data` **excluding** `idx` ([V/verifiers/v1/task.py#L171-L181](https://github.com/PrimeIntellect-ai/verifiers/blob/7407a3ea4f4f5e720bd5acfde007a41a33bb65a1/verifiers/v1/task.py#L171-L181)). |
| `hash` | sha256 of `data` **including** `idx` ([L165-L168](https://github.com/PrimeIntellect-ai/verifiers/blob/7407a3ea4f4f5e720bd5acfde007a41a33bb65a1/verifiers/v1/task.py#L165-L168)). Hash function: [L50-L52](https://github.com/PrimeIntellect-ai/verifiers/blob/7407a3ea4f4f5e720bd5acfde007a41a33bb65a1/verifiers/v1/task.py#L50-L52). |

`TaskData` itself has `idx`, `id`, `name`, `description`, `prompt`, plus scoring
and runtime fields
([V/verifiers/v1/task.py#L85-L136](https://github.com/PrimeIntellect-ai/verifiers/blob/7407a3ea4f4f5e720bd5acfde007a41a33bb65a1/verifiers/v1/task.py#L85-L136)).
There is no family field. This tool defaults family to `env.name`.

When `key` is missing, the reader falls back to `hash`.

### Trace fields used for signal

[V/verifiers/v1/trace.py#L566-L672](https://github.com/PrimeIntellect-ai/verifiers/blob/7407a3ea4f4f5e720bd5acfde007a41a33bb65a1/verifiers/v1/trace.py#L566-L672).

| Field | Meaning |
| --- | --- |
| `version` | `TRACE_VERSION = 1`. |
| `id` | `uuid4().hex`. |
| `rewards` | `{name: {score, weight} \| null}`. `null` means scoring did not run. `weight` defaults to 1.0 ([L315-L321](https://github.com/PrimeIntellect-ai/verifiers/blob/7407a3ea4f4f5e720bd5acfde007a41a33bb65a1/verifiers/v1/trace.py#L315-L321)). |
| `reward` | **Computed** `sum(score * weight)` over non-null heads ([L669-L672](https://github.com/PrimeIntellect-ai/verifiers/blob/7407a3ea4f4f5e720bd5acfde007a41a33bb65a1/verifiers/v1/trace.py#L669-L672)). Written into the JSON record. |
| `metrics` | Unweighted named floats. |
| `info` | Scratch dict. Arrival stamps go here. |
| `ok`, `dropped`, `is_completed`, `stop_condition`, `is_timeout`, `errors` | Outcome. `dropped` is a reason string or omitted. |
| `nodes`, `calls` | Message graph and model calls. Per-token float streams rounded to `float_decimals`. |
| `agent` | `{config, name, trainable, runtime?}`. |

prime-rl unit tests plant the correctness head as `rewards={"correct": Reward(score=...)}`
([P/tests/unit/orchestrator/test_train_sink.py#L61](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/tests/unit/orchestrator/test_train_sink.py#L61))
or `rewards={"reward": Reward(score=...)}`
([P/tests/unit/orchestrator/test_advantage.py#L105](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/tests/unit/orchestrator/test_advantage.py#L105)).
Heads are named by `record_reward(name, value, weight=1.0)`
([V/verifiers/v1/trace.py#L915-L920](https://github.com/PrimeIntellect-ai/verifiers/blob/7407a3ea4f4f5e720bd5acfde007a41a33bb65a1/verifiers/v1/trace.py#L915-L920)).

### Arrival stamps on `trace.info`

Written **before** the episode is logged
([P/src/prime_rl/orchestrator/annotations.py#L20-L30](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/orchestrator/annotations.py#L20-L30)):

```
info.kind = "train" | "eval"
info.dispatch = {step, time}   # dispatch time is trace.timing.start
info.arrival = {step, time}    # arrival step is the batch window
```

Arrival records do **not** contain advantages.

## Length penalty vs score vs final reward

This is the part that does not match a naive reading of `docs/PLAN.md`.

`Trace.reward` is only the weighted sum of named `rewards` heads. GRPO's
`length_penalty` is applied later, in `GRPOAlgorithm.score_group`
([P/src/prime_rl/orchestrator/algo/grpo.py#L66-L114](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/orchestrator/algo/grpo.py#L66-L114)):

```
penalty_frac = w_out * (out / max_out) + w_in * (in / max_in) + w_turns * (turns / max_turns)
penalty = rewards.mean() * penalty_frac
shaped_rewards = rewards - penalty
advantages = shaped_rewards - baseline
```

Defaults are `w_out=0.25`, `w_in=0.1`, `w_turns=0.1`
([P/packages/prime-rl-configs/src/prime_rl/configs/algorithm.py#L75-L87](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/packages/prime-rl-configs/src/prime_rl/configs/algorithm.py#L75-L87)).
`length_penalty` is off by default (`None`, [L188-L189](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/packages/prime-rl-configs/src/prime_rl/configs/algorithm.py#L188-L189)).

`score_group` assigns advantages onto nodes. It does **not** write `shaped_rewards`
back into `trace.reward` or into `rewards[*].score`. After a real run with
length penalty on, an all-correct group still has `rewards.correct.score == 1`
and `trace.reward == 1` on every line. The group is alive because advantages
are non-zero, and those scalars show up on the **annotation** (`info.advantage`)
and on node `advantages` streams, not as a different `trace.reward`.

The reader contract still tracks two metrics, because they can differ on disk
whenever an env records more than one head or a weight other than 1:

- **score** (correctness): `rewards["correct"].score` when that head exists.
- **final reward**: the on-disk `trace.reward` computed field.

The synthetic fixture plants a second head named `length` on the all-correct
groups, with `score = -mean_reward * w_out * (out / max_out)`, so the two
metrics actually diverge in the JSON. That head is a fixture encoding of the
GRPO formula above. A live prime-rl writer never emits it. The estimator in
`docs/PLAN.md` still needs to apply `length_penalty` itself when the run
config has it on and the stream only has a correctness head.

## traces/stream.index.jsonl

Written with each episode
([P/src/prime_rl/monitors/file/monitor.py#L171-L172](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/monitor.py#L171-L172)).
Row builder:
[P/src/prime_rl/monitors/file/traces/index.py#L48-L134](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/index.py#L48-L134).

`line` is 1-based in the stream. `chunk` and `offset` are the uncompressed
address `ChunkedJsonl.append` returned.

Fields written:

`line`, `offset`, `chunk`, `id`, `kind`, `trace_ids`, `env`, `group`, `ok`,
`num_errors`, `reward`, `advantage`, `input_tokens`, `output_tokens`, `turns`,
`branches`, `stop_condition`, `truncated`, `timeout`, `cost`, `dispatch`,
`arrival`, `duration`.

Notes:

- `env` is `episode.env.name` or `episode.env.id` ([L95](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/index.py#L95)).
- `group` is `episode.group.id` ([L96](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/index.py#L96)).
- `reward` is the **mean** of `trace.reward` over traces that have a rewards
  dict ([L66-L67, L99](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/index.py#L66-L99)).
- `dispatch` and `arrival` in the index are **timestamps only**
  (`info.dispatch.time`, `info.arrival.time`), not `{step, time}`
  ([L110-L112](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/index.py#L110-L112)).
- `advantage` is the mean of `trace.info.advantage` when present. Arrival
  records have none, so this is `null` until a later reader folds annotations.

### What the index does not have

`index_row` builds per-component `rewards` / `metrics` / `timing` maps inside
`summarize_episode`, then **pops them before write**
([L127-L134](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/index.py#L127-L134)).

The index has:

- no `task` object
- no task `key` or `hash`
- no per-component `rewards` (`{name: {score, weight}}`)
- no per-component `metrics`

The reader must parse the stream to group by task and to split correctness
score from final reward. The index is only a fast path for env, group, and mean
reward.

## traces/annotations/\<producer\>/

One chunked stream per producer
([P/src/prime_rl/monitors/file/monitor.py#L180-L196](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/monitor.py#L180-L196)).
Producers observed in-tree: `orchestrator`, `trainer`, `eval`, `online-eval`.

Record, from `make_update`
([P/src/prime_rl/monitors/file/traces/update.py#L15-L40](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/update.py#L15-L40)):

```
{"version": 1, "trace_id", "info", "branches": [{"index", advantages?, trainer_logprobs?, entropies?}]}
```

Index row: `{trace_id, chunk, offset, info}` ([L21-L25](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/update.py#L21-L25)).

Updates fold in write order, newest wins
([L61-L79](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/traces/update.py#L61-L79)).

The orchestrator writes, for every **shipped** trace
([P/src/prime_rl/orchestrator/annotations.py#L33-L49](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/orchestrator/annotations.py#L33-L49)):

```
info = {"effective": true, "ship": {"step", "time"}, "advantage"?}
```

`advantage` is copied from `trace.info` when metrics already stamped it. A train
trace with no `effective` annotation never trained (dead, rejected, stale, or
errored). File-monitor `log_episodes(..., subset="effective")` is a no-op
([monitor.py L157-L158](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/monitors/file/monitor.py#L157-L158)),
so there is no second copy of the episode.

## vf-run traces.jsonl

Same Episode JSON, one episode per line, uncompressed
([V/verifiers/v1/utils/trace_store.py#L19-L42](https://github.com/PrimeIntellect-ai/verifiers/blob/7407a3ea4f4f5e720bd5acfde007a41a33bb65a1/verifiers/v1/utils/trace_store.py#L19-L42)).
That path uses pydantic `dump_json(exclude_none=True)`, not `to_json`.

## How this differs from docs/PLAN.md

PLAN was written against prime-rl `bd326dca` / verifiers `7407a3ea`. At
`5af925d4` the files PLAN cites are still the writers above. The surprises:

1. **`stream.index.jsonl` has no task key and no per-component rewards.** PLAN
   said this. It is enforced by `index_row` popping `rewards` / `metrics` /
   `timing`. `summarize_episode` computes those maps and then throws them away.
2. **Index `dispatch` / `arrival` are times, not `{step, time}`.** Step lives
   only on the stream (`trace.info.dispatch.step`).
3. **Length penalty does not change `trace.reward` on disk.** PLAN's "apply the
   penalty before testing" is an estimator rule, not a reader rule. On disk,
   score and final reward differ only when the env recorded multiple heads or
   non-unit weights.
4. **`solved_none` / `solved_all` use `trace.reward == 1.0`**, not a named
   correctness head
   ([P/src/prime_rl/orchestrator/metrics.py#L291-L292](https://github.com/PrimeIntellect-ai/prime-rl/blob/5af925d4b1568559607a7b42b3679d9edf3d39de/src/prime_rl/orchestrator/metrics.py#L291-L292)).
5. **Group ids are UUID strings**, not opaque integers
   (`str(meta.group_id)`).
