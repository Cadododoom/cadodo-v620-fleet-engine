"""Harness auto-link (PLAN I phase 10, spec from PLAN II P2-6).

The engine owns SLOT config; the harness owns ENDPOINT config. When a slot
starts, the engine registers its endpoint with the harness by upserting the
slot into the harness's ``endpoints/profiles.json`` (the harness-side
single source of truth) and running one unattended sync cycle
(probe -> drift -> re-register -> 1-token verify). When the slot stops,
the engine unlinks it: the slot is removed from every profile's port_map
and the profile is disabled if it goes empty.

The engine never edits the harness's config files (profiles.json only);
the harness's own watcher daemon owns all desktop-app config files
(hermes/opencode). This keeps the single-writer rule per file intact on
both sides.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class HarnessLink:
    """One engine slot linked to a harness endpoint profile."""

    slot: int
    name: str
    port: int
    model: str
    profile_id: str = "engine-fleet"   # default harness profile id
    host: str = "127.0.0.1"

    @property
    def slot_id(self) -> str:
        return f"g{self.slot}"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


def _profiles_path(harness_state_dir: str) -> str:
    return os.path.join(harness_state_dir, "profiles.json")


def _load_profiles(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {"schema_version": 1, "profiles": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("schema_version", 1)
    data.setdefault("profiles", [])
    return data


def _save_profiles(path: str, data: dict[str, Any]) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    import tempfile

    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _probe_models(base_url: str, timeout: float = 4.0) -> tuple[bool, Optional[str]]:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8", "replace"))
        names = [m.get("id") or m.get("model") or m.get("name") for m in body.get("models", [])]
        return True, (names[0] if names else None)
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
        return False, None


def _chat_one_token(base_url: str, model: str, timeout: float = 30.0) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the word ok"}],
            "max_tokens": 1,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode("utf-8", "replace"))
        out["choices"][0]["message"]["content"]
        return {"ok": True, "latency_ms": round((time.time() - t0) * 1000.0, 1)}
    except (urllib.error.URLError, OSError, KeyError, IndexError, json.JSONDecodeError) as e:
        return {"ok": False, "error": str(e)}


def register_endpoint(
    link: HarnessLink, harness_state_dir: str, wait_ready: float = 60.0
) -> dict[str, Any]:
    """Upsert the slot into the harness profile + run one sync cycle.

    Waits up to ``wait_ready`` seconds for the slot to serve (the engine
    start path may call this right after spawn). Returns
    {ok, registered, verified, model, latency_ms?, error?}.
    """
    path = _profiles_path(harness_state_dir)
    data = _load_profiles(path)

    profile = None
    for d in data["profiles"]:
        if d.get("id") == link.profile_id:
            profile = d
            break
    if profile is None:
        profile = {
            "id": link.profile_id,
            "kind": "custom",
            "enabled": True,
            "config_path": "",
            "host": link.host,
            "model": link.model or "",
            "port_map": {},
        }
        data["profiles"].append(profile)

    sid = link.slot_id
    profile.setdefault("port_map", {})[sid] = link.port
    if link.model and not profile.get("model"):
        profile["model"] = link.model

    # wait for the slot to actually serve, then verify 1 token
    t0 = time.time()
    up, served = False, None
    while time.time() - t0 < wait_ready:
        up, served = _probe_models(link.base_url)
        if up:
            break
        time.sleep(1.0)
    if not up:
        _save_profiles(path, data)
        return {"ok": False, "registered": True, "verified": False,
                "error": "slot not serving after wait"}
    # llama.cpp serves the model by whatever name was launched with; if the
    # link carried no explicit model, adopt the served one verbatim.
    model = link.model or (served or "")
    if model:
        profile["model"] = model
    chk = _chat_one_token(link.base_url, model)
    profile["last_state"] = "verified" if chk["ok"] else "verify-fail"
    _save_profiles(path, data)
    return {
        "ok": chk["ok"],
        "registered": True,
        "verified": chk["ok"],
        "model": model,
        "latency_ms": chk.get("latency_ms"),
        "error": chk.get("error"),
    }


def deregister_endpoint(link: HarnessLink, harness_state_dir: str) -> dict[str, Any]:
    """Remove the slot from the harness profile (disable if it goes empty)."""
    path = _profiles_path(harness_state_dir)
    data = _load_profiles(path)
    removed = 0
    for d in data["profiles"]:
        pm = d.get("port_map") or {}
        if link.slot_id in pm:
            del pm[link.slot_id]
            removed += 1
            d["port_map"] = pm
            if not pm and d.get("id") == link.profile_id:
                d["enabled"] = False
    _save_profiles(path, data)
    return {"ok": True, "removed": removed}


def run_sync_cycle(
    harness_dir: str, state_dir: str, slots: dict[str, int], python: str | None = None
) -> dict[str, Any]:
    """Run one harness watch cycle in-process (probe/drift/apply/verify).

    Used by the engine's `harness-sync` CLI; the harness daemon does the
    same thing continuously. `harness_dir` is the cadodo-core-omni-harness
    repo root; `state_dir` the harness runtime state dir (profiles.json).
    """
    code = (
        "import sys, json, os; sys.path.insert(0, {h!r});\n"
        "from endpoints.profiles import ProfileStore\n"
        "from endpoints.verify import sync_profile\n"
        "from persist.persistence import PersistenceStore\n"
        "ps = ProfileStore(os.path.join({s!r}, 'profiles.json'))\n"
        "st = PersistenceStore(path=os.path.join({s!r}, 'endpoints.db'))\n"
        "out = []\n"
        "for p in ps.profiles(enabled_only=True):\n"
        "    out.append(sync_profile(p, {slots!r}, store=st))\n"
        "print(json.dumps(out))\n"
    ).format(h=harness_dir, s=state_dir, slots=slots)
    py = python or sys.executable
    proc = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip()[-400:]}
    try:
        results = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"ok": False, "error": "bad sync output"}
    return {"ok": all(r["ok"] for r in results), "results": results}
