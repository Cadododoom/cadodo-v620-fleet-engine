# INSTALL — clone to first token on a V620 box

Target: a V620 (gfx1031, 30-32 GiB) owner goes from clone to first token in
< 30 minutes. Tested on: Ubuntu-class Linux, 56-core, ROCm 6.x,
llama.cpp ROCm MASTER build (YARN support required for > native ctx).

## 0. Prerequisites

1. **ROCm** with working HIP (this box pins LM Studio's vendored ROCm
   libraries; a stock ROCm 6.x install also works for llama.cpp builds).
2. **Python 3.11+** (3.12 ok).
3. A **GGUF model**. Verified: Qwen3.8-27B UD-IQ4_XS
   (~14.3 GB; needs `nextn`/MTP head in the weights for the combo drafter —
   any qwen3.5-family hybrid works; a plain decoder also works with
   `spec-type ngram-mod` or `none`).
4. llama.cpp **ROCm build** of `llama-server` with YARN support. Build once:
   `cmake -B build-rocm -DGGML_ROCM=ON && cmake --build build-rocm -j$(nproc)`.
   (A build capped at the model's native ctx cannot extend past it — that is
   what YARN is for.)

## 1. Install the engine

```sh
git clone https://github.com/Cadododoom/cadodo-v620-fleet-engine
cd cadodo-v620-fleet-engine
./setup.sh            # checks ROCm + llama-server + python, prints your paths
```

`setup.sh` is a check, not a builder: it locates `llama-server`, the ROCm
vendor lib dir, and your GGUF folder, then prints the exact commands for
steps 2-4 with your paths filled in.

## 2. Detect your V620s

```sh
python -m fleet_engine detect \
  --llama-bin /path/to/llama-server \
  --roc-vendor /path/to/rocm/vendor-libs \
  --slots-json /opt/fleet/slots.json
```

Expected: one row per V620, `hip` resolved 1..N (PCI-ordered, stable),
~30 GiB VRAM. Non-V620 cards (e.g. an RX 5700 display card) are filtered by
PCI device ID. On other systems the detector degrades to "N unknown GPUs,
user assigns" — set `gpu` in slots.json by hand.

## 3. Write a dev slot

Edit `/opt/fleet/slots.json` (schema v1) — minimal working slot, dev port:

```json
{
  "schema_version": 1,
  "slots": {
    "1": {
      "slot": 1, "name": "dev1", "gpu": 2, "port": 45800,
      "model": "/opt/models/Qwen3.8-27B-UD-IQ4_XS.gguf",
      "ctx": 528384, "concurrency": 1,
      "rope_scale": 2.0157, "yarn_orig_ctx": 262144,
      "kv_type": "q4_0", "kv_unified": true,
      "threads": 14
    }
  }
}
```

For a short-ctx slot drop `rope_scale` to 1.0 and `ctx` to the model's
native context. `gpu` = the HIP index from step 2.

## 4. Launch

```sh
python -m fleet_engine start --state-dir /opt/fleet \
  --llama-bin /path/to/llama-server \
  --roc-vendor /path/to/rocm/vendor-libs
# first load of a 27B: ~2-4 min. READY line = first token available.
curl -s http://127.0.0.1:45800/health
```

Stop with `python -m fleet_engine stop --state-dir /opt/fleet`. The
supervisor auto-restarts crashes (backoff 2/4/8/16 s, giveup after 8 in
10 min) — see the per-slot log in `/opt/fleet/logs/`.

## 5. First token + verify the tuning baseline

```sh
curl -s http://127.0.0.1:45800/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"<first /v1/models id>","messages":[{"role":"user","content":"Reply with: ok"}],"max_tokens":4}'

python -m fleet_engine bench --endpoint 45800 --out-dir bench/results
python -m fleet_engine tuner-list
```

With the combo spec config (the safe default, see `docs/TUNING.md`) expect
**~30+ t/s** decode on a 27B-class model. The full measured matrix, VRAM
budget, and the g2-lane anomaly are in TUNING.md.

## 6. Connect a client (optional)

- **Hermes Agent**: `python -m fleet_engine conn-apply --state-dir /opt/fleet --config ~/.hermes/config.yaml`
  (dry-run first: add `--dry-run`; drift check: `conn-drift`).
- **OpenCode**: `python -m fleet_engine opc-apply --state-dir /opt/fleet --config ~/.config/opencode/opencode.json`
  then `opc-turn --opencode-bin $(which opencode)` for a real end-to-end turn.

Both writers are single-writer block-scoped: they only touch the
`v620-<slot>` blocks/entries they own and back up the file first.

## Production-parity note

The engine reproduces the verified production launcher flags exactly
(unit-tested in `tests/test_config_and_cmd.py`). The legacy `gpuN.sh`
launchers on this workstation remain untouched; the engine is developed on
dev ports (45700+) per the campaign rules.

## Troubleshooting

| symptom | fix |
|---|---|
| `--list-devices` probe finds no V620 | check `HIP_VISIBLE_DEVICES` order vs `rocm-smi`; the engine probe (not rocm-smi) is authoritative |
| server dies at load, log says ctx | your build lacks YARN; rebuild llama.cpp master or set `ctx` <= native |
| OOM at 528k ctx, 27B | q4_0 KV needs ~8.4 GiB; drop ctx or `parallel`/`concurrency` |
| lane much slower than peers | run `tuner-show spec-off`, restart the lane, re-bench (see TUNING.md anomaly) |
