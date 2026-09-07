"""Benchmark client: runs the standardized prompt set against one OpenAI-compatible
slot endpoint and collects per-run metrics.

Pure stdlib (urllib), unit-testable against a local fake HTTP server.
All measurements are wall-clock from the client side:
  - prefill_tps  = prompt_tokens / (time-to-first-byte)
  - decode_tps   = completion_tokens / (total - ttft)   [client-observed]
  - total_tps    = completion_tokens / total
Spec-acceptance telemetry and server-side rates come from the slot log
(fleet_engine.metrics), not from this client.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Optional

from .prompts import PROMPT_SET_VERSION, BenchPrompt, all_prompts, get_prompt


@dataclass
class RunResult:
    """One prompt execution against one endpoint."""

    prompt: str
    port: int
    concurrency: int
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_ms: float
    prefill_tps: float
    decode_tps: float
    total_tps: float
    ok: bool
    error: str = ""
    cached_tokens: int = 0
    created_ts: Optional[int] = None


@dataclass
class SlotBenchResult:
    """All runs for one endpoint at one concurrency level."""

    port: int
    endpoint: str
    model: str = ""
    concurrency: int = 1
    prompt_set: str = PROMPT_SET_VERSION
    started_ts: float = 0.0
    finished_ts: float = 0.0
    results: list[RunResult] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.ok)

    def decode_tps_mean(self) -> Optional[float]:
        vals = [r.decode_tps for r in self.results if r.ok and r.decode_tps > 0]
        return sum(vals) / len(vals) if vals else None

    def prefill_tps_mean(self) -> Optional[float]:
        vals = [r.prefill_tps for r in self.results if r.ok and r.prefill_tps > 0]
        return sum(vals) / len(vals) if vals else None


def chat_once(
    endpoint: str,
    model: str,
    prompt: BenchPrompt,
    timeout: float = 600.0,
) -> RunResult:
    """One non-streaming chat completion; wall-clock TTFT + total.

    endpoint is the base URL of the /v1 API, e.g. http://127.0.0.1:45600/v1
    """
    url = endpoint.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "max_tokens": prompt.max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    port = _port_of(endpoint)
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
        total_s = time.time() - t0
        data = json.loads(raw)
        usage = data.get("usage") or {}
        n_prompt = int(usage.get("prompt_tokens") or 0)
        n_gen = int(usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
        created = data.get("created")
        if n_gen <= 0:
            return RunResult(
                prompt=prompt.name, port=port, concurrency=1,
                prompt_tokens=n_prompt, completion_tokens=0,
                ttft_ms=0.0, total_ms=total_s * 1000.0,
                prefill_tps=0.0, decode_tps=0.0, total_tps=0.0,
                ok=False, error="no completion tokens", cached_tokens=cached,
            )
        # non-streaming: TTFT unknown -> approximate with total (documented in report)
        ttft = total_s
        prefill_tps = n_prompt / ttft if ttft > 0 else 0.0
        # decode_tps for non-streaming is a total-rate lower bound; server log is truth
        decode_tps = n_gen / total_s
        total_tps = n_gen / total_s
        return RunResult(
            prompt=prompt.name, port=port, concurrency=1,
            prompt_tokens=n_prompt, completion_tokens=n_gen,
            ttft_ms=total_s * 1000.0, total_ms=total_s * 1000.0,
            prefill_tps=prefill_tps, decode_tps=decode_tps, total_tps=total_tps,
            ok=True, cached_tokens=cached, created_ts=created,
        )
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError) as e:
        return RunResult(
            prompt=prompt.name, port=port, concurrency=1,
            prompt_tokens=0, completion_tokens=0,
            ttft_ms=0.0, total_ms=0.0,
            prefill_tps=0.0, decode_tps=0.0, total_tps=0.0,
            ok=False, error=f"{type(e).__name__}: {e}",
        )


def _port_of(endpoint: str) -> int:
    from urllib.parse import urlparse
    p = urlparse(endpoint)
    return p.port or (443 if p.scheme == "https" else 80)


def run_suite(
    endpoint: str,
    model: str,
    prompts: Optional[list[str]] = None,
    concurrency: int = 1,
    timeout: float = 600.0,
    log=print,
) -> SlotBenchResult:
    """Run the prompt set (or a subset by name) against one endpoint.

    concurrency > 1 reuses the same prompt set C times (simultaneous requests
    are issued sequentially per worker via threads).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    names = prompts or [p.name for p in all_prompts()]
    plist = [get_prompt(n) for n in names]
    reps = max(1, concurrency)
    result = SlotBenchResult(
        port=_port_of(endpoint), endpoint=endpoint, model=model,
        concurrency=concurrency, started_ts=time.time(),
    )
    jobs = [p for _ in range(reps) for p in plist]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(chat_once, endpoint, model, p, timeout): p for p in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            r.concurrency = concurrency
            result.results.append(r)
            log(
                f"[{endpoint}:{result.port}] {r.prompt!r} c={concurrency} "
                f"{'OK' if r.ok else 'FAIL'} gen={r.completion_tokens} "
                f"total={r.total_ms:.0f}ms rate={r.total_tps:.2f} t/s"
                + (f" err={r.error[:80]}" if r.error else "")
            )
    result.finished_ts = time.time()
    result.results.sort(key=lambda r: (r.prompt, r.concurrency))
    return result


def result_to_dict(res: SlotBenchResult) -> dict:
    d = asdict(res)
    return d


def load_result(d: dict) -> SlotBenchResult:
    out = SlotBenchResult(
        port=d["port"], endpoint=d["endpoint"], model=d.get("model", ""),
        concurrency=d.get("concurrency", 1), prompt_set=d.get("prompt_set", PROMPT_SET_VERSION),
        started_ts=d.get("started_ts", 0.0), finished_ts=d.get("finished_ts", 0.0),
    )
    out.results = [RunResult(**r) for r in d.get("results", [])]
    return out
