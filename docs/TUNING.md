# TUNING — V620 (gfx1031) + llama.cpp

Every preset in this file carries its **measured** effect, workload, and date.
The rule is: ship only measured-safe defaults; everything unmeasured is
labeled as such. The same matrix is queryable from the repo:

```sh
python -m fleet_engine tuner-list                    # the matrix below
python -m fleet_engine tuner-show prod-default       # full argv to paste
python -m fleet_engine tuner-show spec-off --state-dir DEVDIR --slot 2
```

## The verified production config (safe default)

Qwen3.8-27B UD-IQ4_XS (hybrid SSM+attention, native MTP head), one V620 per
slot, 30 GiB VRAM:

| flag | value | why |
|---|---|---|
| `--n-gpu-layers` | `all` | 13.3 GiB weights fit easily in 30 GiB |
| `--threads` | `14` | 56-core box / 4 lanes; per-lane partition |
| `--ctx-size` | `528384` | 2x the native 262144 ctx |
| `--rope-scaling yarn --rope-scale 2.0157 --yarn-orig-ctx 262144` | | RoPE extension past 262k (the stock ROCm backend capped at native ctx; a MASTER build with YARN support is required) |
| `--cache-type-k/v` | `q4_0` | halves KV VRAM; KV block = 8.4 GiB at 528k ctx |
| `--parallel 1 --kv-unified` | | one stream per lane; unified KV lets the prompt cache share K/V |
| `--spec-type draft-mtp,ngram-mod` + `--spec-draft-n-max 3 --spec-draft-n-min 1 --spec-ngram-mod-n-min 4 --spec-ngram-mod-n-max 16 --spec-ngram-mod-n-match 24 --threads-draft 8` | | the combined drafter (see below) |
| `--mmproj mmproj-F16.gguf` | | vision input (omit if unused) |

Environment (mirrors the launcher this config was verified under):
`HIP_VISIBLE_DEVICES=<n>` to pin the card, `LD_LIBRARY_PATH` pointing at the
ROCm vendor libs the build was compiled against.

## Option matrix (measured)

Single-stream tuning session (2026-09-02, same build/model, coding + prose
workloads):

| preset | drafter | measured |
|---|---|---|
| **prod-default** | MTP3 + ngram-mod (combined) | **107 t/s coding / ~101 t/s prose** — combo >= best single drafter in every workload tested |
| ngram-only | ngram-mod (4-16, match 24) | 108 t/s coding (~= combo; prose is where ngram shines) |
| mtp-only | MTP3 native nextn | 52 t/s coding / 39 t/s prose |
| spec-off | none | ~half of combo on MTP-heavy text (derived from the ratios above) |

Combined-drafter mechanics: per step, the MTP head drafts first; if it produces
nothing, ngram-mod gets its shot. Shared draft budget = max(3, 16) = 16
tokens/step. Live telemetry from a production slot log: **draft acceptance
0.508 (4420 accepted / 8707 generated), mean accepted len 2.58** — the number
to watch if you change ngram params.

Client-observed rate (2026-09-07, prompt set `v1.0-2026-09`,
`bench --all-prod`): **32.1-35.0 t/s** decode on the three healthy lanes —
this is the reproducible baseline for "did I break anything" after any change.

## Unmeasured options (labeled, do not ship as defaults)

- **KV q8_0** (`kv-q8` preset): doubles pure attention-KV from 16.0 to
  32.0 KiB/token. At 528k ctx the KV block alone goes 8.4 -> 16.8 GiB; a 27B
  slot is 21.3 GiB at q4_0 and ~38 GiB at q8_0 — it no longer fits one V620.
  q8_0 is the accuracy-headroom choice for *short-ctx* slots only.
- **Threads** (`threads-28` preset): production uses 14/slot
  (56 cores / 4 lanes). Re-partitioning (2 slots x 28 threads) is a valid
  A/B, not a default.
- **Flash attention**: this config runs with the build's default
  (`--flash-attn auto`); no measured delta recorded, so it is not a preset.
- **MTP depth > 3**: the model's native nextn head supports 3; deeper
  `--spec-draft-n-max` needs a measured A/B before anyone should try it.

## VRAM budget (27B slot, 528k ctx, per card)

| item | GiB |
|---|---|
| weights (UD-IQ4_XS) | ~13.3 |
| KV (q4_0, 528k) | ~8.4 |
| SSM state + activations + headroom | ~3-5 |
| **total** | **~21.3** |

Two such slots fit one 30 GiB V620; three do not.

## Known anomaly (captain's attention, not a tuning issue)

Lane g2 (port 45601) measured **9.2-20.8 t/s** on 2026-09-07 vs 32-35 on the
other three identical lanes, same model, same code path — repeat-confirmed.
`spec-off` is the isolation preset: if g2 recovers with spec off, the spec
pipeline on that card is the cause; if not, it is a per-card runtime issue.
The P1-11 power-cap sweep re-characterizes power vs rate per lane.

## How to reproduce the 32.7 t/s baseline

```sh
# after INSTALL.md (engine built, model downloaded, one dev slot up):
python -m fleet_engine bench --endpoint 45800 --out-dir bench/results
# expect decode_tps_mean in the ~30s on the 27B combo config (32.7 baseline
# was measured on the 4x V620 production fleet, prompt set v1.0-2026-09)
```
