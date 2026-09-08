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

## P1-11 Power envelope (2026-09-08)

### Finding: the 250W cap is a HARDWARE FLOOR, not a knob
The plan's power sweep (250→225→200→180→160→150W, ±20% perf target) is
**infeasible on this hardware**: all four Radeon Pro V620s report
`Max Graphics Package Power (W): 250.0` and refuse every set below 250W
with `Value cannot be less than: 250W`. rocm-smi `--showminpower` returns
no row on this driver, so the floor only surfaces in the rejection.

Measured per card (rocm-smi indices, 2026-09-08):

| rocm GPU | PCI | card | max cap | settable below max |
|----------|------|------|---------|--------------------|
| 0 | 03:00 | V620 | 250W | **no (floor 250W)** |
| 1 | 43:00 | V620 | 250W | **no (floor 250W)** |
| 2 | 83:00 | V620 | 250W | **no (floor 250W)** |
| 3 | C7:00 | V620-class | 150W | **no (floor 150W)** |
| 4 | CA:00 | V620 | 250W | **no (floor 250W)** |

So the fleet's perf/watt is pinned at the TDP point; the sweep axis
degenerates to a single measurement: **250W (locked)**.

### Perf @ 250W (Qwen3.8-27B UD-IQ4_XS, c=1, prod lanes, 2026-09-08)

| lane | port | short-64 | medium-256 | long-512 |
|------|--------|----------|------------|----------|
| 1 | 45600 | 41.91 t/s | 31.93 t/s | 46.72 t/s |
| 2 | 45601 | 12.53 t/s | 22.24 t/s | 28.38 t/s |
| 3 | 45602 | 39.66 t/s | 34.22 t/s | 36.75 t/s |
| 4 | 45603 | 38.57 t/s | 26.11 t/s | 30.69 t/s |

Lane 2 (port 45601) is the re-confirmed **g2 anomaly**: short-64 at
12.53 t/s (~1/3 of the other lanes' ~40 t/s) while medium/long are within
normal band. Consistent with P1-8's finding (9.2 vs 33.4). Hypothesis
unchanged: something in that lane's spec-decode path (MTP draft) stalls on
short sequences — see STATE.md RISK log. Read-only: no lane restarted.

Perf/watt (decode, long-512, c=1) at the locked 250W cap:
- lane 1: 46.72 t/s · lane 2: 28.38 t/s · lane 3: 36.75 t/s · lane 4: 30.69 t/s
(No lower-cap row is measurable on this hardware.)

### Tooling delivered
- `fleet_engine/power.py` — `power_range()` / `set_power()` with floor
  detection; safe to call on locked cards (rejections are clean data, not
  errors).
- `python3 -m fleet_engine set-power [--gpu N | --all] [--watts W]` —
  omit `--watts` to report each card's settable range.
- 6 fake-rocm-smi unit tests (`fleet_engine/tests/test_p1_11_power.py`).

### What this means for the campaign
Power-cap tuning is dead on V620s; the envelope-push lever that remains is
**spec options** (P1-10 presets: mtp/ngram/mixed, draft-len, batch), which
trade decode latency against VRAM/quality at the fixed 250W point. That
sweep needs a free V620 (all four currently host prod lanes + RLM); it runs
the moment a lane is stood down or a dev box is available — recorded as the
remaining P1-11 slice.
