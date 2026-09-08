# cadodo-v620-fleet-engine

Open-source inference engine + fleet control app for the **Radeon Pro V620**
(gfx1031 / RDNA2, 30-32 GB) — built around an integrated **llama.cpp**
backend. Detects up to 16 V620 slots, serves OpenAI-compatible endpoints per
card, and plugs into Hermes Agent Desktop / OpenCode Desktop automatically.

**v1.0** (2026-09-08) — clone-able and installable by another V620 owner;
see [docs/INSTALL.md](docs/INSTALL.md) for the 30-minute
clone-to-first-token path and [docs/TUNING.md](docs/TUNING.md) for the
measured option matrix.

## Verified performance baseline (reference workstation)

Qwen3.8-27B UD-IQ4_XS, MTP3 + ngram-mod combined speculation, q4_0 KV,
YARN 2x on a single 32 GiB V620: **~33-47 tok/s** decode depending on
sequence length (short 41.9 / medium 31.9 / long 46.7 t/s at c=1,
2026-09-08, `bench/`). The engine's command builder reproduces those flags
exactly (`fleet_engine/tests/test_config_and_cmd.py`).

**Power envelope:** all four reference V620s are locked at their 250W TDP
floor — `rocm-smi --setpower` refuses any cap below max on this hardware.
`python -m fleet_engine set-power` reports each card's settable range and is
the canonical way to discover (and, on settable cards, apply) caps;
see `docs/BENCHMARKS.md`.

## Modules

| Module | v1.0 surface |
|---|---|
| Slot Detector | sysfs gfx1031 scan + per-HIP `llama-server --list-devices` probe; PCI-ordered stable slots, up to 16 |
| Config Store | `slots.json` schema v1, atomic writes, detection merge preserves manual edits |
| Runtime Manager | supervisor-based spawn/stop/status/restart; crash backoff 2/4/8/16s, GIVEUP after 8 crashes/10 min |
| Model Registry | GGUF header parser (v3, UD quant names), `models` CLI |
| Control UI | tkinter 16-slot grid: live rates, VRAM/power bars, log viewer, dev start/restart, read-only prod view; headless `--screenshot` |
| Connector (Hermes) | block-scoped single-writer provider-block YAML edits, drift watch, reconnect probe |
| Connector (OpenCode) | OpenAI-compatible provider entries, drift, `opencode run` probe |
| Benchmark suite | versioned 3-prompt set, concurrency sweep, JSON+MD reports, spec-acceptance stats |
| V620 Tuner | measured-safe preset matrix (`tuner-list`/`tuner-show`), docs/TUNING.md |
| Harness Auto-Link | per-slot register/deregister into the omni-harness endpoint store |
| Power control | `power.py` range report + cap apply with hardware-floor detection |

## Quick start (summary — full path in docs/INSTALL.md)

```sh
git clone https://github.com/Cadododoom/cadodo-v620-fleet-engine
cd cadodo-v620-fleet-engine
./setup.sh            # read-only pre-flight: finds ROCm + llama-server + models

# detect your V620 slots
python -m fleet_engine detect --llama-bin /path/to/llama-server \
  --roc-vendor /path/to/rocm/vendor-libs --slots-json /opt/fleet/slots.json

# start a dev slot (45700+ ports; state dir holds slots.json + pids/ + logs/)
python -m fleet_engine start --state-dir /opt/fleet/state --slot 1 \
  --llama-bin /path/to/llama-server --roc-vendor /path/to/rocm/vendor-libs
python -m fleet_engine status --state-dir /opt/fleet/state

# standardized benchmark
python -m fleet_engine bench --endpoint 45800 --out-dir bench/results

# report / apply power caps (reports hardware floor when locked)
python -m fleet_engine set-power --gpu 0
python -m fleet_engine set-power --gpu 0 --watts 250
```

## Development

```sh
uvx ruff check fleet_engine
python -m pytest fleet_engine/tests -v
```

CI (GitHub Actions): ruff + full pytest suite on every push to `main`/`dev/*`.

## License

MIT — see [LICENSE](LICENSE). llama.cpp is MIT; this engine ships **no
weights**. Models/mmproj are git-ignored (LFS-tracked if you add them
locally).
