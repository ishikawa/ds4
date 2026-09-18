# Lightweight quality and performance A/B

Run `scripts/smoke_quality_perf.py` after a large change to inference,
tokenization, prompting, caching, model loading, scheduling, or GPU kernels.
It starts one server at a time, checks two deterministic prompts for exact
output agreement, and measures a cached continuation. The command fails when
text differs or decode throughput regresses beyond the configured threshold.

Do not run it beside another model process. Stop the production server first,
confirm that the selected port is free, and use the same model, flags, host,
power state, and background workload for both sides. The script always stops
the server it starts and writes a JSON report plus one log per condition.

## Compare two binaries

Build the old and new revisions and preserve each `ds4-server` under a distinct
path. Then run:

```sh
python3 scripts/smoke_quality_perf.py \
  --model /path/to/model.gguf \
  --baseline-server /tmp/ds4-server-before \
  --candidate-server ./ds4-server \
  --server-arg=--metal \
  --server-arg=--ssd-streaming \
  --server-arg=--ssd-streaming-cache-experts \
  --server-arg=42GB \
  --output /tmp/ds4-smoke-ab.json
```

## Compare a feature switch

Use the same binary and change only the relevant environment variable:

```sh
python3 scripts/smoke_quality_perf.py \
  --model /path/to/model.gguf \
  --server-arg=--metal \
  --server-arg=--ssd-streaming \
  --server-arg=--ssd-streaming-cache-experts \
  --server-arg=42GB \
  --baseline-env DS4_METAL_GLM_STREAMING_GROW_CACHE_AFTER_PREFILL=0 \
  --candidate-env DS4_METAL_GLM_STREAMING_GROW_CACHE_AFTER_PREFILL=1 \
  --output /tmp/ds4-cache-grow-smoke.json
```

Environment values not listed for a side are inherited from the shell. Set
both sides explicitly when a shell setting could otherwise contaminate the A/B.

## Interpreting the report

- `quality_exact_match` must be `true`. Investigate any mismatch rather than
  accepting it as benchmark noise.
- `decode_change_percent` compares candidate throughput with baseline. The
  default failure floor is -10%; use `--max-regression-percent` to set a tighter
  project-specific bound.
- Compare TTFT separately in the JSON. The script reports it but does not fold
  it into the decode gate because prefill and decode regress for different
  reasons.
- This is a smoke gate. Release quality scoring, long-context tests, backend
  coverage, and repeated benchmark samples remain required where applicable.

The defaults keep the run short: two 48-token quality cases and one 128-token
performance continuation with a 1,000-word prefix. Use at least three
alternating repetitions and 256 or more decode tokens before making a precise
performance claim.
