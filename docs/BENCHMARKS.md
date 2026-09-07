# Benchmarks

Standardized benchmark suite for the V620 Fleet Engine. Every run produces a
machine-readable JSON report plus this kind of Markdown table, so results are
reproducible and comparable across machines, models, and llama.cpp versions.

## How it works

`bench/prompts.py` defines a **fixed, versioned prompt set**
(`v1.0-2026-09`): three deterministic cases (`short-64`, `medium-256`,
`long-512`) with pinned `max_tokens`. The client (`bench/client.py`) issues
non-streaming `/v1/chat/completions` requests and measures wall-clock time;
a concurrency sweep (`1,2,4` levels, `CONCURRENCY_LEVELS`) reuses the prompt
set per worker. Server-side truth (draft acceptance, tg_3s) is pulled from
the slot log by `bench/report.py` when a log path is supplied.

```sh
# one endpoint, all prompts, concurrency 1
python -m fleet_engine bench --endpoint 45798 --out-dir bench/results

# whole dev fleet (45798-45801)
python -m fleet_engine bench --all-dev --concurrency 1 --out-dir bench/results

# production view (read-only) on the docker bridge
python -m fleet_engine bench --all-prod --host 172.27.0.1 --out-dir bench/results
```

Per-endpoint model id defaults to the first `/v1/models` entry.

## Verified baseline (this workstation, 2026-09-07)

Qwen3.8-27B UD-IQ4_XS, spec = `draft-mtp,ngram-mod` (MTP3 + ngram), q4_0 KV,
YARN 2x, production lanes 45600-45603. Client-observed total rate,
`short-64` / `medium-256`:

| lane | short-64 t/s | medium-256 t/s | long-512 t/s |
|---|---|---|---|
| 45600 (g1) | 35.04 | 32.17 | **34.44** |
| 45601 (g2) | 9.23 | 20.84 | - (not run) |
| 45602 (g3) | 33.45 | 32.16 | - |
| 45603 (g4) | 32.16 | 31.05 | - |

**32.7 t/s decode baseline reproduced** (plan milestone) on g1/g3/g4.
Full reports: `bench/results/p18-prod-baseline.{json,md}`.

## Known anomalies (captain's attention)

- **Lane 45601 (g2) degraded**: 9.23 t/s short / 20.84 t/s medium vs ~32-35
  on the other three lanes, same model. Not an engine defect (client measured
  through the same code path on all four); likely a per-card or per-lane
  runtime issue. P1-11 power-cap sweep will re-characterize; until then
  treat g2 numbers as unverified.
- **Spec A/B (MTP vs ngram vs combined)**: requires a GPU. The cron host
  (hermes-web-app container) has no `/dev/dri`, so the A/B leg is pending
  until a tick runs on a host with ROCm access, or the captain authorizes
  a dev-port A/B on the workstation shell. The command builder reproduces
  the production spec flags exactly (unit-tested), so the A/B is a
  measurement task, not an implementation task.

## Caveats

- Non-streaming client timing lumps prefill + decode into `total`; the
  `prefill_tps` column is `prompt_tokens / total` (a lower bound) unless the
  slot log is supplied. Server-side rates (tg_3s, eval time) in the log are
  the ground truth and are parsed by `fleet_engine.metrics`.
- `cached_tokens` from `prompt_tokens_details` is recorded per run; a warm
  prefix cache inflates prefill and deflates decode — keep prompt sets
  distinct across concurrent lanes when measuring prefill.
