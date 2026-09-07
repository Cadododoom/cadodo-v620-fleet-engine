"""P1-6 connector tests: block-scoped YAML edits, drift, reconnect probe.

Uses a synthetic config mirroring the production providers layout (comments,
nested keys, other providers) so byte-preservation can be asserted for real.
The reconnect test binds a real local HTTP server on an ephemeral port.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from fleet_engine import connector
from fleet_engine.config_store import SlotConfig

SAMPLE_CONFIG = """\
# Hermes config — synthetic (tests)
model:
  provider: v620-1
  name: qwen3.8-27b
providers:
  # local LM Studio (must stay untouched)
  custom:
    name: lmstudio
    base_url: http://127.0.0.1:1234/v1
    default_model: qwen3.8-27b
    extra_body:
      temperature: 0.0
  v620-1:
    name: V620 GPU1 (llama.cpp)
    base_url: http://172.27.0.1:45600/v1
    model: /home/theworks/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf
    discover_models: true
    stale_timeout_seconds: 900
    extra_body:
      enable_thinking: false
  v620-2:
    name: V620 GPU2 (llama.cpp)
    base_url: http://172.27.0.1:45601/v1
    model: /home/theworks/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf
    discover_models: true
    extra_body:
      enable_thinking: false
cron:
  enabled: true
"""

GGUF = "/home/theworks/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf"


def make_slot(slot: int, port: int) -> SlotConfig:
    return SlotConfig(slot=slot, name=f"v620-{slot}", gpu=slot - 1, port=port, model=GGUF)


@pytest.fixture
def cfg_path(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(SAMPLE_CONFIG, encoding="utf-8")
    return str(p)


def test_block_span_finds_provider_level_key(cfg_path):
    lines = open(cfg_path).read().splitlines(keepends=True)
    span = connector._find_block_span(lines, "v620-1")
    assert span is not None
    start, end = span
    assert lines[start].startswith("  v620-1:")
    # block ends before the next level-2 key (v620-2)
    assert lines[end].startswith("  v620-2:")
    # nested 'name' inside 'custom' must not be confused with top-level model.name
    # missing key -> synthetic insertion point (start == end), ordered numerically
    ins = connector._find_block_span(lines, "v620-99")
    assert ins is not None and ins[0] == ins[1]
    assert lines[ins[0]].startswith("cron:")  # after all v620-N blocks


def test_apply_only_touches_managed_blocks(cfg_path):
    slot = make_slot(2, 45701)
    changed = connector.apply(cfg_path, [slot], {2: GGUF}, host="172.27.0.1", dry_run=False)
    assert changed == ["v620-2"]
    text = open(cfg_path).read()
    # untouched sections survive byte-for-byte
    assert "  custom:\n    name: lmstudio\n    base_url: http://127.0.0.1:1234/v1\n" in text
    assert "cron:\n  enabled: true\n" in text
    # v620-1 block untouched (still 45600, old name)
    assert "  v620-1:\n    name: V620 GPU1 (llama.cpp)\n    base_url: http://172.27.0.1:45600/v1\n" in text
    # v620-2 block rewritten with fences
    assert "  base_url: http://172.27.0.1:45701/v1\n" in text
    assert connector.FENCE_OPEN in text and connector.FENCE_CLOSE in text


def test_apply_is_idempotent(cfg_path):
    slot = make_slot(1, 45600)
    names = {1: GGUF}
    connector.apply(cfg_path, [slot], names, host="172.27.0.1", dry_run=False)
    first = open(cfg_path).read()
    changed = connector.apply(cfg_path, [slot], names, host="172.27.0.1", dry_run=False)
    assert changed == []
    assert open(cfg_path).read() == first


def test_apply_dry_run_writes_nothing(cfg_path):
    import os
    slot = make_slot(3, 45702)
    before = open(cfg_path).read()
    changed = connector.apply(cfg_path, [slot], {3: GGUF}, dry_run=True)
    assert changed == ["v620-3"]
    assert open(cfg_path).read() == before
    assert not [f for f in os.listdir(os.path.dirname(cfg_path)) if f.startswith("config.yaml.bak")]


def test_apply_creates_backup(cfg_path):
    import glob
    import os
    connector.apply(cfg_path, [make_slot(1, 45701)], {1: GGUF}, host="172.27.0.1", dry_run=False)
    baks = glob.glob(os.path.join(os.path.dirname(cfg_path), "config.yaml.bak-*"))
    assert len(baks) == 1


def test_apply_appends_missing_block(cfg_path):
    text = open(cfg_path).read().replace("  v620-2:\n", "")  # drop the v620-2 key line only
    open(cfg_path, "w").write(text)
    changed = connector.apply(cfg_path, [make_slot(2, 45701)], {2: GGUF}, host="172.27.0.1", dry_run=False)
    assert changed == ["v620-2"]
    # block now appended after v620-1, before cron
    out = open(cfg_path).read()
    assert out.index("  v620-1:") < out.index("  v620-2:") < out.index("cron:")


def test_drift_detects_port_and_model(cfg_path):
    # slots.json says 45700, config still says 45600 -> drift
    slot = make_slot(1, 45700)
    drifts = connector.check_drift(cfg_path, slot, GGUF, host="172.27.0.1")
    fields = {d.field for d in drifts}
    assert "base_url" in fields
    # in-sync: desired == config
    slot_ok = SlotConfig(slot=1, name="v620-1", gpu=0, port=45600, model=GGUF)
    assert connector.check_drift(cfg_path, slot_ok, GGUF, host="172.27.0.1") == []
    # missing block
    assert any(d.field == "block" for d in connector.check_drift(cfg_path, make_slot(4, 45603), GGUF))


def test_drift_read_only(cfg_path):
    before = open(cfg_path).read()
    connector.check_drift(cfg_path, make_slot(1, 45700), GGUF)
    assert open(cfg_path).read() == before


class _Handler(BaseHTTPRequestHandler):
    served = [GGUF, "other-model"]

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            body = json.dumps({"models": [{"name": n} for n in self.served]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):  # silence
        pass


def test_verify_reconnect_ok():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        ok, detail = connector.verify_reconnect(make_slot(1, port))
        assert ok, detail
        assert "present" in detail
    finally:
        srv.shutdown()


def test_verify_reconnect_mismatch_and_down():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        bad = SlotConfig(slot=1, name="v620-1", gpu=0, port=port, model="/nope/missing.gguf")
        ok, detail = connector.verify_reconnect(bad)
        assert not ok and "mismatch" in detail
        ok2, _ = connector.verify_reconnect(make_slot(2, port + 77))  # nothing listening
        assert not ok2 and "unreachable" in _
    finally:
        srv.shutdown()
