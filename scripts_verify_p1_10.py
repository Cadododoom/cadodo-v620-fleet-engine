"""P1-10 LIVE verification: Harness Auto-Link (PLAN I phase 10).

Milestone: "slot start -> harness shows endpoint; slot stop -> deregistered."

Live flow against the real dev slot :45798 (qwen2.5-0.5b, GPU 2):
  1. `fleet_engine start --harness-state-dir` -> slot up AND profile.json
     gains port_map {g1: 45798} + last_state=verified (1-token round-trip
     through the registered endpoint).
  2. harness-sync CLI: one harness watch cycle over the engine slots.
  3. `fleet_engine stop --harness-state-dir` -> slot down AND the slot is
     removed from the profile (profile disabled when it goes empty).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

ENG = "/home/theworks/AI_Workstation_Work/cadodo-v620-fleet-engine"
LLAMA = "/home/theworks/AI_Workstation_Work/llama.cpp/build-rocm/bin/llama-server"
ROC = "/home/theworks/.lmstudio/extensions/backends/vendor/linux-llama-rocm-vendor-v4"
PORT = 45798
MODEL = "/home/theworks/models/draft_models/qwen2.5-0.5b-instruct-q4_k_m.gguf"


def run_cli(args, **kw):
    p = subprocess.run(["python3", "-m", "fleet_engine"] + args, cwd=ENG,
                       capture_output=True, text=True, timeout=300, **kw)
    print(f"$ fleet_engine {' '.join(args[:6])}... rc={p.returncode}")
    if p.stdout.strip():
        print(p.stdout.strip()[:500])
    if p.stderr.strip():
        print(p.stderr.strip()[:500], file=sys.stderr)
    return p


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="p110_live_")
    state = os.path.join(tmp, "state")
    hstate = os.path.join(tmp, "hstate")
    os.makedirs(state)
    slots = {
        "schema_version": 1,
        "slots": {
            "1": {
                "slot": 1, "name": "dev1", "gpu": 2, "host": "127.0.0.1",
                "port": PORT, "model": MODEL, "ctx": 4096, "concurrency": 1,
                "kv_type": "q4_0", "kv_unified": True, "num_gpu_layers": "all",
                "threads": 8, "rope_scale": 1.0, "harness_link": True,
                "spec": {"spec_type": "none"},
            }
        },
    }
    with open(os.path.join(state, "slots.json"), "w") as f:
        json.dump(slots, f, indent=2)

    base = ["--state-dir", state, "--llama-bin", LLAMA, "--roc-vendor", ROC]

    # 1) start with harness link -> registered + verified
    p = run_cli(["start"] + base + ["--harness-state-dir", hstate])
    assert p.returncode == 0, "start failed"
    with open(os.path.join(hstate, "profiles.json")) as f:
        profs = json.load(f)["profiles"]
    prof = next(d for d in profs if d["id"] == "engine-fleet")
    assert prof["port_map"] == {"g1": PORT}, prof
    print(f"[link] profile after start: port_map={prof['port_map']} "
          f"last_state={prof['last_state']}")
    assert prof["last_state"] == "verified", "start must verify the endpoint"

    # 2) harness-sync: one harness-side cycle over engine slots
    p = run_cli(["harness-sync", "--state-dir", state,
                 "--harness-dir", "/home/theworks/AI_Workstation_Work/cadodo-core-omni-harness",
                 "--harness-state-dir", hstate])
    assert p.returncode == 0, f"harness-sync failed: {p.stdout}"
    print("[sync] harness-side cycle over engine slots ok")

    # 3) stop with harness link -> deregistered, profile disabled
    p = run_cli(["stop"] + base + ["--harness-state-dir", hstate])
    assert p.returncode == 0, "stop failed"
    with open(os.path.join(hstate, "profiles.json")) as f:
        profs = json.load(f)["profiles"]
    prof = next(d for d in profs if d["id"] == "engine-fleet")
    print(f"[link] profile after stop: port_map={prof['port_map']} enabled={prof['enabled']}")
    assert prof["port_map"] == {}, "slot must be deregistered on stop"
    assert prof["enabled"] is False, "empty profile must be disabled"

    print("\nP1-10 LIVE VERIFICATION: ALL CHECKS PASSED")
    print(f"state: {tmp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
