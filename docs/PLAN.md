# grpo-signal plan (v1, Nov 12 2026)

## Goal
Before a GRPO run on verifiers + prime-rl, say how much of each task family will
actually produce a gradient, and what to change in the config if the answer is "not much".
It reads logs that already exist first, and only runs a model if the user explicitly asks it to.

## The problem
1. Wasted rollouts. GRPO compares the G rollouts of one task. If they all get the same
   reward, the group teaches nothing. For a pass/fail task with success rate p, that
   happens with probability p^G + (1-p)^G. At G=8, a task with p=0.9 is dead 43% of the time.
   prime-rl generates those rollouts, then throws them away.
2. Refill bias. With `constant_trainer_batch_size = true` (the default), prime-rl keeps
   drawing new tasks until the batch is full. Families that are too easy or too hard get
   drawn at their dataset rate but rarely survive, so the batch the model trains on tilts
   toward medium difficulty. A whole family can drop out of the gradient with no warning.
3. Constant non-0/1 rewards. `solved_none` / `solved_all` only count groups that are
   all 0 or all 1. A family where every rollout scores 0.5 is just as dead, but looks fine.
4. How groups are weighted. prime-rl's GRPO subtracts the group mean and does not divide
   by the std (the Dr. GRPO choice). So a group's weight grows with p(1-p): near-solved
   and near-impossible tasks count less even when they are alive. GRPO with std division
   gives every live group about the same weight. The report shows both.

## What verifiers and prime-rl do today (read in code)
Read at prime-rl `bd326dca348a1af68b6d543c237837354bd783b7` (its `deps/verifiers` is
pinned to `7407a3ea`) and verifiers `7407a3ea4f4f5e720bd5acfde007a41a33bb65a1`
(tag v0.4.1 = `540f40fa`). Code lines cited from earlier reads at prime-rl `b2ae73bf` / `b7770c93`
are unchanged in these files. Prefixes:
P = https://github.com/PrimeIntellect-ai/prime-rl/blob/bd326dca348a1af68b6d543c237837354bd783b7/
V = https://github.com/PrimeIntellect-ai/verifiers/blob/7407a3ea4f4f5e720bd5acfde007a41a33bb65a1/
CFG = packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py

- Advantage = reward minus the group mean, with no std division: P/src/prime_rl/orchestrator/algo/grpo.py#L105-L114.
  The only GRPO options are `length_penalty`, `length_weighted_baseline` and
  `loss_aggregation` (default "token"): P/packages/prime-rl-configs/src/prime_rl/configs/algorithm.py#L181-L202.
- Tokens with zero advantage are always dropped (no flag): P/src/prime_rl/orchestrator/train_sink.py#L51-L73.
  With `orchestrator.constant_trainer_batch_size = true` (CFG L541) this happens before the
  batch is counted, and the batch is refilled (train_sink.py L386-L395); with false, after
  (L511-L520), with no refill.
- Empty batch: warn and skip (P/src/prime_rl/orchestrator/orchestrator.py#L700-L702).
- Optional gate `curriculum.gates.<name>` with `type="advantage_range"`, `reject_min=0`,
  `reject_max=0` (CFG L210-L230). Off by default because `gates = {}`.
- Optional sampler `curriculum.sampler.type="difficulty_pool"` with pools hard ≤0.25 (weight 0.2),
  normal ≤0.75 (1.0), easy ≤1.0 (0.2), `seed=42` (CFG L168-L201). Per-task state is only
  the latest group mean, keyed by task key, and it can only be restored from a checkpoint:
  P/src/prime_rl/orchestrator/curriculum/samplers/pool.py#L46-L96.
- `group_size` defaults to 1 (CFG L242, L357). GRPO with G=1 is always dead, and this is only
  caught at run time (docs/algorithms.md L115).
- Nothing checks reward variance before launch. `dry_run` only validates the config.
  The old `online_difficulty_filtering` / `easy_threshold` / `hard_threshold` options were
  removed in prime-rl #2639 (commit f6563ff0e).
- Metrics only, never gates: `solved_none/all/some` (0/1 rewards only, metrics.py L286-L297),
  a warning when ≤10% of traces are trainable (orchestrator.py L711-L716), and counts of
  discarded and zero-output groups.
- Sampler and gate types are a fixed set (CFG L204-L207, L223), so there is no plug-in
  point without forking.
- verifiers has no trainer. `vf-validate`'s noop check marks a task invalid if it already
  scores ≥ 1.0 with no agent turn (V/verifiers/v1/cli/validate.py#L254-L305). That is the only
  check on reward sensitivity, and nothing in verifiers looks at reward variance.

## Use cases (agreed with JP)
1. Pre-launch check: `grpo-signal audit` on N rollouts per task from a local vLLM server,
   giving a per-family report plus a config diff.
2. Hub env author: a "signal card" for an env (dead rate vs G, per-family mix) to put in its README.
3. Mid-run diagnosis: point it at an existing prime-rl run directory and see which families
   are vanishing, compared with what the metrics show.

## Inputs and readers (on-disk formats checked in code)
prime-rl file monitor (on by default, CFG sibling monitors.py L55). Everything is under
`<output_dir>/<run.dir>/monitors/file/` (P/src/prime_rl/utils/pathing.py#L309-L312;
under `rl` the orchestrator's output_dir is the run dir, configs/rl.py L169-L171, L301-L302).
- `metrics.jsonl`: one JSON row per log call, `{"step", "time", "producer"?, ...metrics}`;
  non-finite values are dropped (P/src/prime_rl/monitors/file/monitor.py#L64-L76).
- `traces/stream/`: the episode stream, chunked JSONL. Chunks are `00000.jsonl`, `00001.jsonl`, ...;
  a finished chunk is sealed as `NNNNN.jsonl.zst` (zstd level 3, seekable 4 MiB frames, plain
  file removed afterwards); the live chunk stays plain; a line never spans chunks; default chunk
  size 5 GiB (`chunk_bytes`), `compress=true` (P/src/prime_rl/monitors/file/traces/chunks.py#L1-L118,
  monitors.py L31-L38). `zstd -dcf stream/* | jq` reads it all.
  - One line = one `vf.Episode` written once, on arrival, in arrival order, whatever its fate
    (errored, rejected, dead or shipped): orchestrator.py L592-L603, monitor.py L151-L178.
    Per-token floats are rounded to `float_decimals` (default 4).
  - Episode fields: `id`, `env` (name/id), `task` {type, data, key, hash}, `group.id`
    (set by the dispatcher, dispatcher.py L784), `run`, `ok`, `traces[]`
    (V/verifiers/v1/episode.py#L39-L68). Trace fields: `rewards` {name: {score, weight}},
    `metrics`, `info`, `reward` (computed: sum of score×weight), `ok`, `dropped`
    (V/verifiers/v1/trace.py#L298-L308, L598-L623, L669-L672).
  - At arrival, `trace.info` is stamped with `kind` ("train"/"eval"), `dispatch` {step, time}
    and `arrival` {step, time} (P/src/prime_rl/orchestrator/annotations.py#L20-L30).
    Arrival records do not contain advantages.
- `traces/stream.index.jsonl`: one row per episode, sitting next to the stream directory
  (traces/__init__.py L27-L30). Fields: `line`, `chunk`, `offset`, `id`, `kind`, `trace_ids`,
  `env`, `group`, `ok`, `num_errors`, `reward` (mean over the episode's traces), `advantage`,
  tokens, `turns`, `branches`, `stop_condition`, `truncated`, `timeout`, `cost`, `dispatch`,
  `arrival`, `duration`. It does NOT include the task key or the per-component
  rewards/metrics (index.py L48-L134). So the reader must use the stream to group by task;
  the index is only a fast path for env/group/reward.
- `traces/annotations/<producer>/` plus `traces/annotations/<producer>.index.jsonl`: one stream
  per producer (`orchestrator`, `trainer`, `eval`, `online-eval`), same chunked layout
  (monitor.py L180-L196). Record: `{"version": 1, "trace_id", "info", "branches": [{index,
  advantages?|trainer_logprobs?|entropies?}]}`. Index row: `{trace_id, chunk, offset, info}`
  (P/src/prime_rl/monitors/file/traces/update.py#L15-L40). Updates are merged in write order,
  newest wins (update.py L61-L79).
  The orchestrator writes `info = {"effective": true, "ship": {step, time}, "advantage"?}` for
  every trace that shipped (annotations.py L33-L49). A train trace with no `effective` annotation
  never trained (dead, rejected, stale or errored).
- vf-run `traces.jsonl`: the same Episode JSON, one episode per line, uncompressed
  (V/verifiers/v1/utils/trace_store.py#L19-L42). `-r` = rollouts per task, `--select.*` picks
  tasks, `client.base_url` = endpoint (default: paid Prime inference, V/verifiers/v1/configs/client.py#L23-L71).
- prime-rl `uv run eval -r N` writes the same file-monitor layout (producer "eval").
- Family mapping: verifiers TaskData has no family field. The user picks one: a TaskData field
  (`--family-field difficulty`), `name`, a regex, or a CSV of task_key,family. Default is one
  family per env name.

## Math
Per task, binary rewards (k successes out of N):
- Posterior on p: Beta(a=k+1, b=N-k+1).
- Expected dead rate at G: E[p^G + (1-p)^G] = (B(a+G,b) + B(a,b+G)) / B(a,b). Closed form, computed in log space.
- Wilson 95% interval on p, for display.
Continuous or non-0/1 rewards:
- Dead fraction = share of G-subsets of the N rewards with std < eps (eps = 1e-6, because pruning
  tests for exactly 0.0). Subsets are drawn without replacement, which is unbiased and needs N ≥ G;
  enumerate all when C(N,G) ≤ 20k, otherwise Monte Carlo. Resampling with replacement overstates
  the dead rate, so it is not used.
- Flag families that are "constant but not 0/1".
Group weight, binary (checked identities):
- Mean-only (prime-rl): E[sum |A|] = 2(G-1)p(1-p), E[sum A^2] = (G-1)p(1-p).
- With std division: each live group weighs about the same, whatever p is.
Caveats:
- With `length_penalty` set, penalty = mean reward × length fraction (grpo.py L98-L104). An all-1 group
  becomes alive (lengths differ), an all-0 group stays dead. Apply the penalty before testing.
- `max_rl`: zero advantage when the group mean ≤ 0 (algo/max_rl.py L27). `rae`: a running-average
  baseline per agent, so constant groups are not dead there (algo/rae.py L17-L37). v1 supports
  `grpo` and `max_rl`; for other algorithms it says it does not apply.
- Families are aggregated over tasks, with a bootstrap over tasks (not over rollouts) for intervals.

## Resampling simulation (copies prime-rl exactly)
- Every new group picks an env with `random.Random(42).choices(envs, weights=ratio)`
  (train_source.py L31, L46, L58-L68), then takes the next task from that env's sampler and
  uses that env's `group_size`.
- StandardSampler: least-sampled first, ties broken by source order, i.e. a fixed cycle; a dead
  task still counts as drawn and comes back next pass (samplers/standard.py L15-L65).
  DifficultyPoolSampler: weighted draw with replacement, unseen tasks weight 1, weights updated
  from the simulated group mean (pool.py L48-L82).
- The batch fills by trace count (`batch_size`, default 128) and is cut once full; leftovers
  carry over (train_sink.py L246-L250, L503-L509).
- Outputs per step: each family's share of the batch, steps per pass, rollouts per step.
  Analytic check: share_f ∝ ratio_e × G_e × live_f.
- Ignored in v1: the order in which rollouts finish (it only shifts things between batches)
  and staleness.

## Outputs
- Per-family table: dataset share, predicted gradient share (mean-only and std-divided), p and
  its interval, reward std, dead rate at G = 4/8/16, wasted-rollout fraction, "constant non-0/1" flag.
- Dead rate vs G curve, per family.
- Config diff (TOML) using existing prime-rl keys only:
  `orchestrator.train.group_size`; per source in `[[orchestrator.train.source]]`: `name`,
  `ratio`, `group_size`, `select.include.keys` / `select.exclude.keys`
  (`select.exclude.files.keys` only on verifiers newer than 0.4.1), `curriculum.sampler.type`
  / `pools.<n>.threshold|weight`, `curriculum.gates.<n>.type="advantage_range"`,
  `reject_min`, `reject_max`; `orchestrator.batch_size`, `orchestrator.constant_trainer_batch_size`.
  To restore a target mix: put each family in its own source, with ratio ∝ target share / (G × live).
- Predicted metrics to compare against the run, all broken out per env: `batch/{env}`
  (orchestrator.py L789-L790) vs `rollout/env_share` (metrics.py L645), `train/{env}/solved_*`,
  `curriculum/{env}/admission_rate`; and globally `rollout/queued/pruned`, `rollout/no_signal`.
  There are no per-task metrics; per-task numbers come from the stream.
- `report.json` (stable schema) + `report.html` (one static file) + a short table in the terminal.
- `--fail-if` for CI, e.g. `--fail-if "dead_rate>0.5" --fail-if "min_family_share<0.25*dataset_share"`.

## CLI (PROPOSAL)
    grpo-signal audit --from-run outputs/my-run [--steps 0:50] [--family-field difficulty] -G 8
    grpo-signal audit --from-traces outputs/vf-run/traces.jsonl --family-map families.csv -G 4,8,16
    grpo-signal audit --rollouts --taskset <id> -N 16 --base-url http://localhost:8000/v1 \
        --max-tokens 2e7 --max-usd 0        # later milestone, opt-in, requires a cap
    grpo-signal simulate report.json --config rl.toml      # predicted mix under a given config
    grpo-signal card report.json > SIGNAL.md               # Hub env author card

## Module layout (PROPOSAL)
    src/grpo_signal/
      types.py      Rollout, TaskStats, FamilyStats, Report (pydantic)
      readers/      prime_run.py, vf_traces.py, chunks.py (plain + .zst), families.py
      estimate.py   per-task and per-family estimators
      simulate.py   copy of the prime-rl sampler and batch logic
      recommend.py  config diff + predicted metrics
      report.py     JSON, HTML, terminal
      cli.py
Main signatures:
    def read_prime_run(run_dir: Path, *, kind: Literal["train","eval"]="train", steps: slice|None=None) -> Iterator[Rollout]
    def read_vf_traces(path: Path) -> Iterator[Rollout]
    def iter_chunked_jsonl(directory: Path) -> Iterator[bytes]
    def assign_family(r: Rollout, spec: FamilySpec) -> str
    def dead_rate_binary(k: int, n: int, G: int) -> float
    def dead_rate_subsets(rewards: Sequence[float], G: int, eps: float=1e-6, max_subsets: int=20_000, seed: int=0) -> float
    def wilson(k: int, n: int, z: float=1.96) -> tuple[float, float]
    def signal_weight(p: float, G: int, normalize_std: bool) -> float
    def simulate_mix(families: list[FamilyStats], cfg: PrimeConfig, steps: int, seed: int=42) -> MixResult
    def recommend(report: Report, target: Mix|None=None) -> ConfigDiff
`Rollout` = {env, task_key, task_idx, family, group_id, reward, rewards: dict[str,float], shipped: bool|None, step: int|None}.

## Privacy
- prime-rl's Prime monitor is on by default and uploads episodes (monitors.py L58-L61).
  For private sets, run with `--no-monitors.prime`. The tool never uploads; reports contain
  task keys and numbers only, never prompts or replies, unless `--include-text` is passed.
- vf-run's default endpoint is paid Prime inference; the docs point to a local vLLM server.

## Cost and N
- Rollouts = N × T (T = tasks audited). Tokens ≈ N × T × (input + output tokens per rollout).
  Dollars = N × T × (in_tok × in_price + out_tok × out_price); local GPU time ≈ N × T × out_tok / throughput.
  Overhead compared with training = N × T / (steps × batch_size).
- For family shares, sampling T_f ≈ 200 tasks per family is enough; per-task exclusion needs every task.
- N=16 supports G ≤ 16. At 16/16 correct, p ≥ 0.81 (Wilson 95%); at 0/16, p ≤ 0.19; at 8/16, [0.28, 0.72].
  N=32 gives ≥0.89 / ≤0.11 / [0.34, 0.66]. Advice: N=16, then 16 more rollouts only for tasks
  at 0/16 or 16/16 before excluding them.

## Milestones
1. Scaffold: types, CLI stubs, readers that pass on fixtures, CI (cloud-agent PR, JP merges).
2. Zero-cost path, which is the first real milestone: `--from-run` / `--from-traces` → estimator →
   JSON/HTML report + `--fail-if`. Checked on one existing prime-rl run's logs.
3. Simulation + config diff + predicted metrics; compare against `batch/{env}` from that run.
4. Signal card for Hub env authors.
5. Opt-in rollout path (local vLLM first) with token/dollar caps; nothing is spent without JP's OK.

## Tests
- Synthetic trace fixtures (stream chunks, both plain and .zst, index, annotations, vf-run file)
  with planted p per family; the estimator must recover p within its interval.
- Golden test: closed-form dead rate vs Monte Carlo from the Beta posterior; subset method vs
  exact enumeration; p^G + (1-p)^G at known values (p=0.9, G=8 → 0.4305).
- Simulation vs analytic share (ratio × G × live) within 1% over 10k steps; StandardSampler order
  matches prime-rl on a 5-task toy example.
- Reader contract test against prime-rl's own writer (`ChunkedJsonl`, `make_update`) in an
  optional CI job pinned to a prime-rl SHA, so format drift breaks CI instead of the report.

## Open questions for JP
1. Default family source: env name, a TaskData field, or a required `--family-map`?
2. Which real run's logs to use for milestone 2 (one with on-disk traces, nothing private)?
3. Python 3.11+ and a dependency on `verifiers` (for the Episode schema) or no dependency
   (raw JSON only)? Raw JSON keeps it light and works across verifiers versions.
4. License MIT and the name `grpo-signal`: OK as is?
5. Should the HTML report be a single static file or skipped in v1?
