"""P1-10 unit tests: harness auto-link (register/deregister/sync)."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fleet_engine.harness_link import (
    HarnessLink,
    _chat_one_token,
    _probe_models,
    deregister_endpoint,
    register_endpoint,
)


class _Slot(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: ANN001
        pass

    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/models"):
            self._send(200, json.dumps({"models": [
                {"name": "/m/qwen2.5-0.5b-instruct.gguf",
                 "model": "/m/qwen2.5-0.5b-instruct.gguf"}]}).encode())
        else:
            self._send(404, b"{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        assert req["max_tokens"] == 1
        self._send(200, json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode())


def test_register_and_deregister(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 45880), _Slot)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        link = HarnessLink(slot=1, name="dev1", port=45880, model="")
        res = register_endpoint(link, str(tmp_path))
        assert res["ok"] and res["registered"] and res["verified"], res
        # model was adopted from the served name (full path as llama.cpp serves)
        assert res["model"] == "/m/qwen2.5-0.5b-instruct.gguf"

        prof = json.loads((tmp_path / "profiles.json").read_text())["profiles"][0]
        assert prof["port_map"] == {"g1": 45880}
        assert prof["last_state"] == "verified"

        # re-register with explicit model is idempotent (no duplicate profile)
        link2 = HarnessLink(slot=1, name="dev1", port=45880, model="custom-name")
        register_endpoint(link2, str(tmp_path))
        profs = json.loads((tmp_path / "profiles.json").read_text())["profiles"]
        assert len(profs) == 1

        res = deregister_endpoint(link, str(tmp_path))
        assert res["removed"] == 1
        prof = json.loads((tmp_path / "profiles.json").read_text())["profiles"][0]
        assert prof["port_map"] == {} and prof["enabled"] is False
    finally:
        srv.shutdown()


def test_register_slot_never_serving(tmp_path):
    link = HarnessLink(slot=2, name="dev2", port=45881, model="")
    res = register_endpoint(link, str(tmp_path), wait_ready=3)
    assert res["ok"] is False and res["registered"] and not res["verified"]
    assert "not serving" in res["error"]


def test_probe_and_chat():
    srv = ThreadingHTTPServer(("127.0.0.1", 45882), _Slot)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        up, served = _probe_models("http://127.0.0.1:45882/v1")
        # engine API: (up, first_served_name) - the name llama.cpp serves it under
        assert up and served == "/m/qwen2.5-0.5b-instruct.gguf"
        r = _chat_one_token("http://127.0.0.1:45882/v1", served)
        assert r["ok"]
        up2, _ = _probe_models("http://127.0.0.1:45883/v1")
        assert not up2
    finally:
        srv.shutdown()
