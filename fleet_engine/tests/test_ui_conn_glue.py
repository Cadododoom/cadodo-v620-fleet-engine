"""P1-6 UI glue: slot start via Panel._ctrl with --hermes-config syncs the
provider block automatically (PLAN I phase-6: 'change in UI -> provider
updates without a manual step')."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from fleet_engine.app import Panel
from fleet_engine.config_store import ConfigStore, SlotConfig, slot_to_dict
from fleet_engine.ui import FleetModel

GGUF = "/home/theworks/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf"

CFG = """\
providers:
  v620-1:
    name: old
    base_url: http://127.0.0.1:45001/v1
    model: /old/path.gguf
"""


class _Handler(BaseHTTPRequestHandler):
    served = [GGUF]

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            body = json.dumps({"models": [{"name": n} for n in self.served]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):  # noqa: N802
        pass


class _FakeRuntimePaths:
    def __init__(self, sd: str):
        self._sd = sd

    def slots_json(self):
        return self._sd + "/slots.json"


class _FakeRuntime:
    """Just enough surface for Panel._ctrl: paths.slots_json + start/stop."""

    def __init__(self, state_dir: str):
        self.paths = _FakeRuntimePaths(state_dir)

    def start(self, cfg: SlotConfig, wait_ready: bool = True,
              ready_timeout: float = 240.0, log_fn=print):
        self._started = cfg

    def stop(self, cfg: SlotConfig, timeout: float = 30.0, log_fn=print):
        pass


def test_panel_start_syncs_provider(tmp_path):
    """UI start on a dev slot -> provider block in the Hermes config is
    rewritten to the slot's live port/model with no manual step."""
    import os
    import time
    import tkinter as tk

    if not os.environ.get("DISPLAY"):
        import pytest
        pytest.skip("no DISPLAY (run under Xvfb)")

    state = tmp_path / "state"
    state.mkdir()
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    try:
        slot = SlotConfig(slot=1, name="dev1", gpu=0, port=port, model=GGUF)
        store = ConfigStore(str(state / "slots.json"))
        data = store.load()
        data["slots"]["1"] = slot_to_dict(slot)
        store.save(data)

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(CFG)

        model = FleetModel(dev_slots=[slot], dev_state_dir=str(state))
        model.refresh()
        model.build_views()
        rt = _FakeRuntime(str(state))
        root = tk.Tk()
        try:
            panel = Panel(root, model, runtime=rt, state_dir=str(state),
                          hermes_config=str(cfg_path))
            view = [v for v in panel.model._views if v.slot == 1][0]
            panel._ctrl("start", view)
            for _ in range(200):
                if not panel._busy:
                    break
                time.sleep(0.05)
            assert not panel._busy, "ctrl thread did not settle"
            text = cfg_path.read_text()
            assert f"base_url: http://127.0.0.1:{port}/v1" in text
            assert f"model: {GGUF}" in text
            assert "managed block" in text
            assert "http://127.0.0.1:45001/v1" not in text
            # backup created (atomic apply path)
            baks = list(tmp_path.glob("config.yaml.bak-*"))
            assert len(baks) == 1
        finally:
            root.destroy()
    finally:
        srv.shutdown()
