"""Runtime manager tests: fake-server spawn/stop/status + supervisor relaunch."""

from __future__ import annotations

import os
import stat
import time

import pytest

from fleet_engine.config_store import ConfigStore, SlotConfig
from fleet_engine.runtime import Runtime, _alive, _health


def make_state_dir(tmp_path) -> str:
    base = str(tmp_path / "state")
    os.makedirs(base)
    cfg = SlotConfig(slot=1, name="dev1", gpu=1, port=45799, model=str(tmp_path / "fake.gguf"))
    ConfigStore(os.path.join(base, "slots.json")).set_slot(cfg)
    return base


def make_fake_server(tmp_path) -> str:
    """A fake llama-server: serves /health, exits on SIGTERM (like the real
    llama-server does). All CLI args are ignored."""
    script = tmp_path / "llama-server"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import http.server, os, signal\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        if self.path == '/health':\n"
        "            b = b'{\"status\":\"ok\"}'\n"
        "            self.send_response(200); self.send_header('Content-Length', str(len(b)))\n"
        "            self.end_headers(); self.wfile.write(b)\n"
        "        else:\n"
        "            self.send_response(404); self.end_headers()\n"
        "    def log_message(self, *a): pass\n"
        "def _term(signum, frame):\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, _term)\n"
        "srv = http.server.HTTPServer(('127.0.0.1', 45799), H)\n"
        "srv.serve_forever()\n"
    )
    script.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    (tmp_path / "fake.gguf").write_text("x")
    return str(script)


@pytest.fixture
def state(tmp_path):
    return make_state_dir(tmp_path)


def test_start_stop_roundtrip(tmp_path, state, capsys):
    bin_ = make_fake_server(tmp_path)
    rt = Runtime(state_dir=state, llama_bin=bin_)
    cfg = rt.store.get_slot(1)
    t0 = time.time()
    pid = rt.start(cfg, wait_ready=True, ready_timeout=20)
    ready_s = time.time() - t0
    assert pid is not None and _alive(pid)
    assert ready_s < 10
    st = rt.status(cfg)
    assert st.running and st.health == '{"status":"ok"}' and st.supervisor_pid is not None
    rt.stop(cfg)
    st = rt.status(cfg)
    assert not st.running and st.pid is None
    assert not os.path.exists(os.path.join(state, "pids", "dev1.pid"))


def test_start_refuses_missing_model(tmp_path, state):
    bin_ = make_fake_server(tmp_path)
    rt = Runtime(state_dir=state, llama_bin=bin_)
    cfg = rt.store.get_slot(1)
    cfg.model = str(tmp_path / "nope.gguf")
    with pytest.raises(FileNotFoundError):
        rt.start(cfg)


def test_supervisor_relives_killed_server(tmp_path):
    """Restart storm test: kill the server, supervisor relaunches it.
    Recovery must complete well under 10 s."""
    state = make_state_dir(tmp_path)
    bin_ = make_fake_server(tmp_path)
    rt = Runtime(state_dir=state, llama_bin=bin_)
    cfg = rt.store.get_slot(1)
    pid = rt.start(cfg, wait_ready=True, ready_timeout=20)
    # kill the server hard (simulates a crash)
    os.kill(pid, 9)
    t0 = time.time()
    recovered = False
    trace = []
    while time.time() - t0 < 30:
        st = rt.status(cfg)
        h = _health(45799)
        trace.append((round(time.time() - t0, 1), st.pid, h))
        if st.running and st.pid != pid and h is not None:
            recovered = True
            break
        time.sleep(0.25)
    rt.stop(cfg)
    dt = time.time() - t0
    logtxt = open(os.path.join(state, "logs", "dev1.log")).read()
    if not recovered:
        raise AssertionError(f"no recovery in window. trace={trace}\nlog:\n{logtxt}")
    # recovered within window; dt includes the rest of the loop iteration
    assert dt < 10, f"recovery took {dt:.1f}s, expected < 10s. trace={trace}\nlog:\n{logtxt}"
