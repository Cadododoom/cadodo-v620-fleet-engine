# v1.0 announcement (draft)

> Draft for the captain to review/publish. Not posted anywhere automatically.

---

**cadodo-v620-fleet-engine v1.0** — open-source inference engine + fleet
control for the Radeon Pro V620 (gfx1031, 32 GB).

Clone-to-first-token in ~30 minutes: detect your V620s, write `slots.json`,
start a supervised llama.cpp slot per card, and get OpenAI-compatible
endpoints that auto-wire into Hermes Agent Desktop and OpenCode Desktop.

What's in the box:

- **Slot detector** — sysfs gfx1031 scan + per-HIP probe, up to 16 slots,
  PCI-ordered and stable across restarts.
- **Runtime manager** — per-slot supervisor with crash backoff and GIVEUP;
  kill -9 recovers in ~5 s.
- **Model registry** — GGUF header parsing (v3, UD quant names) with a
  `models` picker.
- **Control UI** — 16-slot grid with live tok/s, VRAM/power bars, log viewer,
  dev start/restart, headless screenshots.
- **Connectors** — single-writer provider blocks for Hermes + OpenCode with
  drift watch and reconnect.
- **Benchmark suite** — versioned 3-prompt set, concurrency sweep, JSON/MD
  reports, spec-acceptance stats.
- **Tuner** — measured-safe preset matrix (MTP/ngram, KV quant, YARN).
- **Power control** — settable-range report + cap apply with hardware-floor
  detection.
- **Harness auto-link** — slot start/stop registers into the omni-harness
  endpoint store.

Reference performance (Qwen3.8-27B UD-IQ4_XS, single 32 GB V620, c=1):
41.9 / 31.9 / 46.7 tok/s at short/medium/long; ~33 t/s aggregate baseline.
Note: the reference cards are locked at their 250W TDP floor — power-cap
tuning is hardware-blocked there, and the tooling reports that cleanly
instead of failing.

MIT licensed, no weights shipped. Docs: INSTALL (clone-to-token), TUNING
(option matrix), BENCHMARKS (numbers + power envelope).

Repo: https://github.com/Cadododoom/cadodo-v620-fleet-engine (tag v1.0)
