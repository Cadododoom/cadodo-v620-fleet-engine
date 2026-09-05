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
| Slot Detector (gfx1031 filter, up to 16) | Phase 2 |
| Config Store (`slots.json`, atomic, schema v1) | **v0.1** |
| Runtime Manager (spawn/stop/health/restart) | skeleton (cmd builder **v0.1**) |
| Model Registry (GGUF folder scan) | Phase 4 |
| Control UI (16-slot grid, live rates, VRAM/power) | Phase 5 |
| Connector (Hermes / OpenCode / custom) | Phases 6-7 |
| Benchmark suite (prefill/decode/spec-acceptance) | Phase 8 |
| V620 Tuner (measured-safe option matrix) | Phase 9 |
| Harness Auto-Link (PLAN II) | Phase 10 |
| Envelope push (power-cap sweep) | Phase 11 |

## Verified performance baseline (this workstation)

Qwen3.8-27B UD-IQ4_XS, MTP3 + ngram-mod combined speculation, q4_0 KV,
YARN 2x: **32.7 tok/s** decode (legacy launcher, commit ee41df8). The engine's
command builder reproduces those flags exactly (see
`fleet_engine/tests/test_config_and_cmd.py`).

## Development

```sh
uvx ruff check fleet_engine
python -m pytest fleet_engine/tests -v
```

## License

MIT — see [LICENSE](LICENSE). llama.cpp itself is MIT; this engine ships no
weights. Models and mmproj are excluded (LFS-tracked if you add them locally).
