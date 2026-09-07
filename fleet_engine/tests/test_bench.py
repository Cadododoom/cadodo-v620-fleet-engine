"""Unit tests for the benchmark suite (fake OpenAI server, no GPU needed)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from bench.client import chat_once, run_suite
from bench.prompts import all_prompts, get_prompt
from bench.report import build_report, render_markdown, spec_stats_from_log, write_report

HERE = Path(__file__).parent
LOGS = HERE.parent / "devstate" / "logs"


# ---------------------------------------------------------------------------
# fake OpenAI-compatible server
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        n_gen = 4
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "model": body.get("model", "fake"),
            "created": 1788775000,
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": n_gen,
                      "prompt_tokens_details": {"cached_tokens": 2}},
        }).encode())

    def log_message(self, *a):  # silence
        pass


@pytest.fixture(scope="module")
def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

def test_prompt_set_complete():
    names = [p.name for p in all_prompts()]
    assert len(names) == 3
    assert "short-64" in names and "medium-256" in names and "long-512" in names
    for p in all_prompts():
        assert p.system and p.user and p.max_tokens > 0


def test_get_prompt_unknown_raises():
    with pytest.raises(KeyError):
        get_prompt("nope")


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------

def test_chat_once_ok(fake_server):
    p = get_prompt("short-64")
    r = chat_once(fake_server, "fake-model", p)
    assert r.ok, r.error
    assert r.completion_tokens == 4
    assert r.prompt_tokens == 12
    assert r.cached_tokens == 2
    assert r.total_ms > 0
    assert r.total_tps > 0


def test_chat_once_bad_endpoint():
    p = get_prompt("short-64")
    r = chat_once("http://127.0.0.1:1/v1", "fake", p, timeout=2.0)
    assert not r.ok
    assert r.error


def test_run_suite_concurrency(fake_server):
    res = run_suite(fake_server, "fake-model", prompts=["short-64"],
                    concurrency=2, timeout=10.0, log=lambda *a: None)
    # 2 reps x 1 prompt
    assert len(res.results) == 2
    assert res.ok_count == 2
    assert res.decode_tps_mean() is not None
    assert res.concurrency == 2


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def test_spec_stats_from_real_log():
    # real prod/dev log shapes: dev1.log contains draft acceptance lines
    log = LOGS / "dev1.log"
    if not log.is_file():
        pytest.skip("dev1.log not present on this host")
    stats = spec_stats_from_log(str(log))
    if stats:
        assert 0.0 <= stats["mean"] <= 1.0
        assert stats["samples"] >= 1
        assert stats["accepted_total"] >= 0


def test_spec_stats_missing_log():
    assert spec_stats_from_log("/nonexistent/log.txt") == {}


def test_build_and_render_markdown(fake_server):
    res = run_suite(fake_server, "fake-model", prompts=["short-64"], timeout=10.0,
                    log=lambda *a: None)
    report = build_report([res], meta={"host": "test", "model": "fake"})
    assert report["report_version"] == "1.0"
    assert len(report["slots"]) == 1
    md = render_markdown(report)
    assert "# V620 Fleet Benchmark Report" in md
    assert "model: fake" in md
    assert "short-64" in md


def test_write_report_files(tmp_path, fake_server):
    res = run_suite(fake_server, "fake-model", prompts=["short-64"], timeout=10.0,
                    log=lambda *a: None)
    report = build_report([res])
    jpath, mpath = write_report(report, str(tmp_path), stem="testbench")
    assert Path(jpath).is_file() and Path(mpath).is_file()
    loaded = json.loads(Path(jpath).read_text())
    assert loaded["slots"][0]["port"] == res.port
