"""P1-7 OpenCode connector tests: provider entry shape, byte-preservation of
other providers, drift, idempotence, chat probe against a live local server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from fleet_engine import opencode
from fleet_engine.config_store import SlotConfig

GGUF = "/home/theworks/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf"
MODEL_ID = "Qwen3.8-27B-UD-IQ4_XS.gguf"

SAMPLE = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
        "lmstudio": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "LM Studio (Local)",
            "options": {"baseURL": "http://127.0.0.1:1234/v1", "apiKey": "lm-studio"},
            "models": {"qwen3.6-35b": {"name": "Qwen 35B Local", "max_tokens": 16384}},
        },
        "v620_tp4_dflash": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "4x AMD V620 ROCm TP=4 + DFlash-2",
            "options": {"baseURL": "http://127.0.0.1:8000/v1", "apiKey": "v620-rocm-local"},
            "models": {"Qwen3.8-27B-UD-Q6_K_XL": {"name": "Qwen 3.8 27B"}},
        },
    },
    "model": "v620_tp4_dflash/Qwen3.8-27B-UD-Q6_K_XL",
    "compaction": {"auto": False},
    "permission": {"bash": "allow"},
}


def make_slot(slot: int, port: int) -> SlotConfig:
    return SlotConfig(slot=slot, name=f"dev{slot}", gpu=slot - 1, port=port, model=GGUF,
                      ctx=262144)


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "opencode.json"
    p.write_text(json.dumps(SAMPLE, indent=2) + "\n")
    return str(p)


def test_provider_entry_shape(cfg):
    slot = make_slot(1, 45701)
    e = opencode.provider_entry(slot)
    assert e["npm"] == "@ai-sdk/openai-compatible"
    assert e["options"]["baseURL"] == "http://127.0.0.1:45701/v1"
    assert e["options"]["apiKey"] == opencode.API_KEY
    assert MODEL_ID in e["models"]
    assert e["models"][MODEL_ID]["context_length"] == 262144
    assert e["models"][MODEL_ID]["max_tokens"] == 32768  # capped at 32k for max_tokens field


def test_apply_preserves_other_providers(cfg):
    slot = make_slot(1, 45701)
    changed = opencode.apply(cfg, [slot], dry_run=False)
    assert changed == ["v620-1"]
    data = json.load(open(cfg))
    # untouched providers survive
    assert data["provider"]["lmstudio"]["options"]["baseURL"] == "http://127.0.0.1:1234/v1"
    assert data["provider"]["v620_tp4_dflash"]["models"] == SAMPLE["provider"]["v620_tp4_dflash"]["models"]
    assert data["compaction"] == {"auto": False}
    assert data["permission"] == {"bash": "allow"}
    # new entry present
    assert data["provider"]["v620-1"]["options"]["baseURL"] == "http://127.0.0.1:45701/v1"
    # default model untouched without --make-default
    assert data["model"] == "v620_tp4_dflash/Qwen3.8-27B-UD-Q6_K_XL"


def test_apply_make_default_and_idempotent(cfg):
    slot = make_slot(1, 45701)
    changed = opencode.apply(cfg, [slot], dry_run=False, make_default=True)
    assert "model" in changed
    data = json.load(open(cfg))
    assert data["model"] == f"v620-1/{MODEL_ID}"
    before = open(cfg).read()
    assert opencode.apply(cfg, [slot], dry_run=False, make_default=True) == []
    assert open(cfg).read() == before


def test_apply_dry_run_no_write(cfg):
    slot = make_slot(1, 45701)
    before = open(cfg).read()
    assert opencode.apply(cfg, [slot], dry_run=True) == ["v620-1"]
    assert open(cfg).read() == before


def test_apply_creates_backup(cfg):
    import glob
    import os
    opencode.apply(cfg, [make_slot(1, 45701)], dry_run=False)
    assert len(glob.glob(os.path.join(os.path.dirname(cfg), "opencode.json.bak-*"))) == 1


def test_drift_detects_port_and_missing(cfg):
    slot = make_slot(1, 45702)
    # missing entry
    assert any(d.field == "provider" for d in opencode.check_drift(cfg, slot))
    # write it, then verify in sync
    opencode.apply(cfg, [slot], dry_run=False)
    assert opencode.check_drift(cfg, slot) == []
    # now desired port changes -> base_url drift
    moved = make_slot(1, 45703)
    drifts = opencode.check_drift(cfg, moved)
    assert any(d.field == "options" for d in drifts)


def test_chat_probe_ok_and_down():
    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            assert body["model"] == MODEL_ID
            out = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):  # noqa: N802
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        ok, content = opencode.chat_probe(make_slot(1, port))
        assert ok and content == "ok"
        ok2, detail = opencode.chat_probe(make_slot(2, port + 77), timeout=2.0)
        assert not ok2 and "failed" in detail
    finally:
        srv.shutdown()
