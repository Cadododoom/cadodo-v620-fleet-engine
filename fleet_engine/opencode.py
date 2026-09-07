"""OpenCode connector (P1-7): write/update OpenAI-compatible provider entries.

OpenCode's config is JSON (~/.config/opencode/opencode.json). Each engine slot
maps to one provider key ``v620-<slot>`` using the same single-writer rule as
the Hermes connector: the engine owns its provider entries; everything else in
the file is preserved by a full JSON round-trip (OpenCode writes its own config,
so a canonical json.dump indent-2 rewrite is how the file already looks).

Milestone: an OpenCode turn completes via an engine slot. The verification
path is a chat completion POST to the written baseURL with the written model
id - exactly what the OpenCode client sends - plus (when the binary is
available) ``opencode run`` for a real end-to-end turn.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config_store import SlotConfig

PROVIDER_KEY = "v620-{slot}"
NPM_PACKAGE = "@ai-sdk/openai-compatible"
API_KEY = "v620-fleet-local"  # llama.cpp ignores it; OpenCode requires non-empty


def provider_key(slot: SlotConfig) -> str:
    return PROVIDER_KEY.format(slot=slot.slot)


def model_id(slot: SlotConfig) -> str:
    """Model id as served by llama.cpp: the GGUF filename (what /v1/models returns)."""
    return os.path.basename(slot.model)


def provider_entry(slot: SlotConfig, host: str = "127.0.0.1") -> dict:
    """The provider dict the engine writes for one slot."""
    return {
        "npm": NPM_PACKAGE,
        "name": f"V620 slot {slot.slot} ({slot.name})",
        "options": {
            "baseURL": f"http://{host}:{slot.port}/v1",
            "apiKey": API_KEY,
        },
        "models": {
            model_id(slot): {
                "name": f"V620 slot {slot.slot} ({os.path.basename(slot.model)})",
                "max_tokens": min(slot.ctx, 32768),
                "context_length": slot.ctx,
                "attachment": False,
                "modalities": {"input": ["text"], "output": ["text"]},
            }
        },
    }


@dataclass
class Drift:
    key: str
    field: str
    config_value: object
    desired_value: object

    def __str__(self) -> str:
        return (f"{self.key}.{self.field}: config={self.config_value!r} "
                f"desired={self.desired_value!r}")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(path: str, data: dict, backup: bool = True) -> None:
    if backup:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        shutil.copy2(path, f"{path}.bak-{stamp}")
    dirn = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=dirn, prefix=".opencode-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def check_drift(config_path: str, slot: SlotConfig, host: str = "127.0.0.1") -> list[Drift]:
    """Read-only drift check for one slot's provider entry."""
    key = provider_key(slot)
    want = provider_entry(slot, host)
    data = load_config(config_path)
    providers = data.get("provider", {})
    if key not in providers:
        return [Drift(key, "provider", None, "(missing provider entry)")]
    live = providers[key]
    drifts = []
    for field in ("npm", "name"):
        if live.get(field) != want[field]:
            drifts.append(Drift(key, field, live.get(field), want[field]))
    if live.get("options") != want["options"]:
        drifts.append(Drift(key, "options", live.get("options"), want["options"]))
    mid = model_id(slot)
    if mid not in live.get("models", {}):
        drifts.append(Drift(key, f"models.{mid}", None, "(missing model)"))
    else:
        lm = live["models"][mid]
        for f in ("max_tokens", "context_length"):
            if lm.get(f) != want["models"][mid][f]:
                drifts.append(Drift(key, f"models.{mid}.{f}", lm.get(f), want["models"][mid][f]))
    return drifts


def apply(
    config_path: str,
    slots: list[SlotConfig],
    host: str = "127.0.0.1",
    dry_run: bool = True,
    backup: bool = True,
    make_default: bool = False,
) -> list[str]:
    """Write provider entries for all slots. Returns changed keys.

    ``make_default`` also sets the top-level ``model`` to the first slot's
    provider/model id (used for the dev verification turn).
    """
    data = load_config(config_path)
    providers = data.setdefault("provider", {})
    changed: list[str] = []
    for slot in slots:
        key = provider_key(slot)
        entry = provider_entry(slot, host)
        if providers.get(key) != entry:
            providers[key] = entry
            changed.append(key)
    if make_default and changed:
        first = slots[0]
        want_model = f"{provider_key(first)}/{model_id(first)}"
        if data.get("model") != want_model:
            data["model"] = want_model
            changed.append("model")
    if changed and not dry_run:
        save_config(config_path, data, backup=backup)
    return changed


def chat_probe(slot: SlotConfig, host: str = "127.0.0.1", timeout: float = 90.0,
               prompt: str = "Reply with the single word: ok") -> tuple[bool, str]:
    """Send the exact request OpenCode would send to this slot (chat completion
    to baseURL with the written model id). Returns (ok, detail-or-reply)."""
    url = f"http://{host}:{slot.port}/v1/chat/completions"
    body = json.dumps({
        "model": model_id(slot),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "stream": False,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        content = out["choices"][0]["message"]["content"]
        return True, content
    except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as e:
        return False, f"probe failed: {e}"


def opencode_turn(opencode_bin: str, model_ref: str, prompt: str,
                  timeout: float = 300.0) -> tuple[bool, str]:
    """Real end-to-end turn via the opencode binary (when available)."""
    import subprocess
    try:
        p = subprocess.run(
            [opencode_bin, "run", "--model", model_ref, prompt],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"opencode run failed: {e}"
    out = (p.stdout or "").strip()
    if p.returncode != 0:
        return False, f"rc={p.returncode} err={p.stderr.strip()[:300]}"
    return True, out[-500:]
