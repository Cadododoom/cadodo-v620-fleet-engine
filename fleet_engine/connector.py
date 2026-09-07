"""Connector: Hermes provider auto-write + drift watch + reconnect.

PLAN I Phase 6. The engine is the SINGLE WRITER for its managed provider
block(s) in the Hermes config (``providers.v620-<slot>``); everything else
in the file is byte-preserved.

Design
------
- Block-scoped editing: only the lines of a managed ``v620-*:`` key at the
  ``providers:`` mapping level are rewritten. Comments, ordering, and every
  other file section stay untouched (no full-file YAML re-dump).
- Dry-run is the default. ``--apply`` (or ``apply=True``) writes the file,
  with an atomic tmp+rename and a timestamped ``.bak`` alongside.
- Drift watch: ``check_drift`` is READ-ONLY — it compares live config values
  to desired and reports; ``apply`` then re-writes the block. This implements
  the plan's "drift-detect is read-only + alert" race rule.
- Reconnect: ``verify_reconnect`` hits ``GET /v1/models`` on the slot port;
  the provider block is only considered healthy when the model the server
  actually serves matches the one written into the config.

Campaign rule: live Hermes config (~/.../config.yaml) is only touched with
explicit --apply and only after the dry-run diff has been reviewed.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config_store import SlotConfig

MANAGED_KEY_RE = re.compile(r"^v620-\d+$")
FENCE_OPEN = "# >>> v620-fleet-engine managed block — do not edit inside <<<"
FENCE_CLOSE = "# <<< v620-fleet-engine managed block end <<<"


def provider_key(slot: SlotConfig) -> str:
    """Provider key for a slot: v620-<slot> (1..16)."""
    return f"v620-{slot.slot}"


def block_yaml(slot: SlotConfig, model_name: str, host: str = "127.0.0.1",
               indent: int = 2) -> str:
    """Render the managed provider block for one slot as YAML text.

    ``indent`` is the column of the ``v620-N:`` key (2 = under ``providers:``).
    Values mirror the production fleet providers: llama.cpp base_url, full GGUF
    path as model, thinking disabled for Qwen3.x.
    """
    pad = " " * indent
    pad2 = " " * (indent + 2)
    fence = " " * (indent + 2)
    return (
        f"{pad}{provider_key(slot)}:\n"
        f"{fence}{FENCE_OPEN}\n"
        f"{pad2}name: V620 slot {slot.slot} ({slot.name})\n"
        f"{pad2}base_url: http://{host}:{slot.port}/v1\n"
        f"{pad2}model: {model_name}\n"
        f"{pad2}discover_models: true\n"
        f"{pad2}extra_body:\n"
        f"{pad2}  enable_thinking: false\n"
        f"{fence}{FENCE_CLOSE}\n"
    )


@dataclass
class Drift:
    key: str
    field: str
    config_value: object
    desired_value: object

    def __str__(self) -> str:
        return (f"{self.key}.{self.field}: config={self.config_value!r} "
                f"desired={self.desired_value!r}")


def _providers_span(lines: list[str]) -> tuple[int, int] | None:
    """Line span of the top-level ``providers:`` mapping (indent 0)."""
    start = None
    for i, ln in enumerate(lines):
        s = ln.rstrip("\n")
        if s == "providers:" or (s.startswith("providers:") and s[9] == " "):
            start = i
            break
    if start is None:
        return None
    for j in range(start + 1, len(lines)):
        s = lines[j].rstrip("\n")
        if not s.strip():
            continue
        if not s.startswith(" "):
            return start, j
    return start, len(lines)


def _provider_keys(lines: list[str]) -> list[tuple[str, int, int]]:
    """(key, block_start, block_end) for every level-2 key under providers:.

    A block ends at the next level-2 key, the next level-0 line, or EOF.
    """
    span = _providers_span(lines)
    if span is None:
        return []
    s0, e0 = span
    keys: list[tuple[str, int, int]] = []
    i = s0 + 1
    while i < e0:
        s = lines[i].rstrip("\n")
        if not s.strip():
            i += 1
            continue
        m = re.match(r"^  (\S[^:]*):", s)
        if m:
            key = m.group(1)
            j = i + 1
            while j < e0:
                t = lines[j].rstrip("\n")
                if t.strip() and len(t) - len(t.lstrip(" ")) <= 2:
                    break
                j += 1
            keys.append((key, i, j))
            i = j
        else:
            i += 1
    return keys


def _key_numeric(key: str) -> int | None:
    m = re.fullmatch(r"v620-(\d+)", key)
    return int(m.group(1)) if m else None


def _find_block_span(lines: list[str], key: str) -> tuple[int, int] | None:
    """(start, end) of ``key``'s block at providers level.

    If the key is absent, returns a synthetic (ins, ins) insertion point:
    numerically ordered among v620-N keys when possible, else end of the
    providers mapping. Returns None only if the file has no providers: mapping.
    """
    for k, s, e in _provider_keys(lines):
        if k == key:
            return s, e
    span = _providers_span(lines)
    if span is None:
        return None
    keys = _provider_keys(lines)
    if not keys:
        return span[1], span[1]
    num = _key_numeric(key)
    if num is not None:
        lower = [t for t in keys if (_key_numeric(t[0]) or 0) < num]
        if lower:
            return lower[-1][2], lower[-1][2]
        higher = [t for t in keys if (_key_numeric(t[0]) or 10**9) > num]
        if higher:
            return higher[0][1], higher[0][1]
    return keys[-1][2], keys[-1][2]


def _desired_fields(slot: SlotConfig, model_name: str, host: str) -> dict:
    return {
        "base_url": f"http://{host}:{slot.port}/v1",
        "model": model_name,
    }


def check_drift(config_path: str, slot: SlotConfig, model_name: str,
                host: str = "127.0.0.1") -> list[Drift]:
    """Read-only: compare the live provider block to desired. Empty = in sync.

    A missing block is reported as one drift on the ``<key>`` field.
    """
    key = provider_key(slot)
    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines(keepends=True)
    if not any(k == key for k, _s, _e in _provider_keys(lines)):
        desired = _desired_fields(slot, model_name, host)
        return [Drift(key, "block", None, f"(missing block; base_url={desired['base_url']})")]
    span = _find_block_span(lines, key)
    assert span is not None
    start, end = span
    import yaml
    block = yaml.safe_load("".join(lines[start:end]))
    if not isinstance(block, dict) or key not in block:
        return [Drift(key, "block", None, "(unparseable block)")]
    live = block[key]
    desired = _desired_fields(slot, model_name, host)
    drifts = []
    for field, want in desired.items():
        got = live.get(field)
        if got != want:
            drifts.append(Drift(key, field, got, want))
    return drifts


def _splice(lines: list[str], key: str, block_text: str) -> list[str]:
    """Replace (or append) the managed block for ``key`` in place."""
    span = _find_block_span(lines, key)
    new_lines = block_text.splitlines(keepends=True)
    if span is None:
        # append at end of file (after ``providers:`` level mapping)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        return lines + new_lines
    start, end = span
    return lines[:start] + new_lines + lines[end:]


def apply(
    config_path: str,
    slots: list[SlotConfig],
    model_names: dict[int, str],
    host: str = "127.0.0.1",
    dry_run: bool = True,
    backup: bool = True,
) -> list[str]:
    """Write managed provider blocks for all slots. Returns the changed keys.

    ``model_names`` maps slot number -> the model string to register (usually
    the full GGUF path, matching production convention).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.splitlines(keepends=True)
    changed: list[str] = []
    for slot in slots:
        key = provider_key(slot)
        block_text = block_yaml(slot, model_names[slot.slot], host=host)
        before = "".join(_block_or_empty(lines, key))
        after_lines = _splice(lines, key, block_text)
        after = "".join(_block_or_empty(after_lines, key))
        if before != after:
            changed.append(key)
        lines = after_lines
    if changed and not dry_run:
        new_content = "".join(lines)
        if backup:
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            shutil.copy2(config_path, f"{config_path}.bak-{stamp}")
        dirn = os.path.dirname(os.path.abspath(config_path)) or "."
        fd, tmp = tempfile.mkstemp(dir=dirn, prefix=".fleetcfg-", suffix=".yaml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_content)
            os.replace(tmp, config_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return changed


def _block_or_empty(lines: list[str], key: str) -> list[str]:
    span = _find_block_span(lines, key)
    if span is None:
        return []
    return lines[span[0]:span[1]]


def diff_preview(config_path: str, slots: list[SlotConfig],
                 model_names: dict[int, str], host: str = "127.0.0.1") -> str:
    """Human-readable preview of what apply() would change (no writes)."""
    lines = open(config_path, "r", encoding="utf-8").read().splitlines(keepends=True)
    out = []
    for slot in slots:
        key = provider_key(slot)
        before = _block_or_empty(lines, key)
        after = _block_or_empty(_splice(lines, key, block_yaml(slot, model_names[slot.slot], host)), key)
        status = "UNCHANGED" if before == after else "CHANGED"
        out.append(f"--- {key} [{status}] ---")
        for ln in after:
            out.append(ln.rstrip("\n"))
    return "\n".join(out)


def verify_reconnect(slot: SlotConfig, timeout: float = 3.0,
                     host: str = "127.0.0.1") -> tuple[bool, str]:
    """Reconnect probe: is the slot serving, and with the expected model?

    Returns (ok, detail). Model match is checked against GET /v1/models.
    """
    url = f"http://{host}:{slot.port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as e:
        return False, f"unreachable: {e}"
    import json
    try:
        models = json.loads(body).get("models", [])
    except (json.JSONDecodeError, AttributeError):
        return False, "unparseable /v1/models"
    names = {m.get("name") for m in models}
    if slot.model and slot.model in names:
        return True, f"ok (model {slot.model} present)"
    return False, f"model mismatch: slot={slot.model!r} served={sorted(names)[:3]}"


def wait_reconnect(slot: SlotConfig, timeout: float = 60.0,
                   host: str = "127.0.0.1") -> tuple[bool, str]:
    """Poll verify_reconnect until ok or timeout (auto-reconnect loop)."""
    deadline = time.time() + timeout
    last = "not probed"
    while time.time() < deadline:
        ok, last = verify_reconnect(slot, host=host)
        if ok:
            return True, last
        time.sleep(1.0)
    return False, f"timed out after {timeout:.0f}s: {last}"
