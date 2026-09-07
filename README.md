# cadodo-v620-fleet-engine

Open-source inference engine + fleet control app for the **Radeon Pro V620**
(gfx1031 / RDNA2, 32 GB) — built around an integrated **llama.cpp** backend.
Detects up to 16 V620 slots, serves OpenAI-compatible endpoints per card, and
plugs into Hermes Agent Desktop / OpenCode Desktop automatically.

> Status: **Phase 1 skeleton** (dev branch). Production fleet on this workstation
> still runs the legacy `gpuN.sh` launchers; this engine is developed on dev
> ports (45700+) until Runtime Manager parity is verified.

## Features (target surface)

| Module | State |
|---|---|
| Slot Detector (gfx1031 filter, up to 16) | **v1.0** (sysfs + engine HIP probe) |
| Config Store (`slots.json`, atomic, schema v1) | **v1.0** (+ detection merge) |
| Runtime Manager (spawn/stop/health/restart, supervisor) | **v1.0** (absorbs gpuN.sh logic; auto-restart w/ backoff) |
| Model Registry (GGUF folder scan) | **v1.0** (v3 header, UD quant names) |
| Control UI (16-slot grid, live rates, VRAM/power) | **v1.0** (tkinter, headless screenshot verified) |
| Connector (Hermes / OpenCode / custom) | **v1.0** (block-scoped writers, drift watch, E2E turns) |
| Benchmark suite (prefill/decode/spec-acceptance) | **v1.0** (prompt set v1.0-2026-09, 32.7 t/s baseline) |
| V620 Tuner (measured-safe option matrix) | **v1.0** (tuner-list/tuner-show, docs/TUNING.md) |
| Harness Auto-Link (PLAN II) | Phase 10 |
| Envelope push (power-cap sweep) | Phase 11 |

## Verified performance baseline (this workstation)

Qwen3.8-27B UD-IQ4_XS, MTP3 + ngram-mod combined speculation, q4_0 KV,
YARN 2x: **32.7 tok/s** decode (legacy launcher, commit ee41df8). The engine's
command builder reproduces those flags exactly (see
`fleet_engine/tests/test_config_and_cmd.py`).

## CLI

```sh
# detect V620 slots (sysfs scan + per-HIP llama-server --list-devices probe)
python -m fleet_engine detect \
  --llama-bin /path/to/llama-server \
  --roc-vendor /path/to/roc-vendor-libs \
  --slots-json ./slots.json
```

Without `--llama-bin` the detector falls back to a sysfs-only scan: slots are
numbered by PCI bus address and `hip_index` is left unknown for the user to
assign. With the probe, each HIP index is resolved to a card name and matched
against the V620s found in sysfs; the runtime re-verifies the pinned card at
start time, so a mis-match fails loudly.

### Runtime control (Phase 3)

Each managed fleet has a **state dir** holding `slots.json`, `pids/`, and
`logs/`. `slots.json` is the single source of truth for what runs.

```sh
# start slot 1 (or all slots) under a detached supervisor
python -m fleet_engine start --state-dir ./devstate \
  --slot 1 \
  --llama-bin /path/to/llama-server \
  --roc-vendor /path/to/roc-vendor-libs

# status table (pid, supervisor, port, /health)
python -m fleet_engine status --state-dir ./devstate

# stop (idempotent; SIGTERM supervisor + server, SIGKILL fallback)
python -m fleet_engine stop --state-dir ./devstate --slot 1
python -m fleet_engine restart --state-dir ./devstate --slot 1
```

Behavior notes:
- **Supervisor**: each slot runs under a detached `fleet_engine
  runtime-supervisor` process. On server crash it relaunches with backoff
  (2/4/8/16 s, capped; resets after 10 min uptime) and gives up after 8
  crashes in a rolling 10-min window, leaving a `GIVEUP` line in the log.
- **Stop** SIGTERMs the supervisor first (it owns and terminates its own
  server child), then the server, then waits and escalates to SIGKILL.
  Verified: kill -9 of a live server auto-recovers in ~5 s end-to-end.
- **Dev-only by default**: campaign dev slots use ports 45700+ and their own
  state dir; production lanes (45600-45603) are never touched.

## Development

```sh
uvx ruff check fleet_engine
python -m pytest fleet_engine/tests -v
```

## License

MIT — see [LICENSE](LICENSE). llama.cpp itself is MIT; this engine ships no
weights. Models and mmproj are excluded (LFS-tracked if you add them locally).
