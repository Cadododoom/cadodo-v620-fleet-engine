"""Power-cap control for V620 slots (P1-11 envelope push).

Wraps `rocm-smi --setpower` so the engine can cap a slot's package power
from slots.json (`power_cap_watts`) or the CLI.

Hardware reality found on this box (verified 2026-09-08, rocm-smi 6.x):
  * every V620 (gfx1031) reports a 250W MAXIMUM package power and REFUSES
    to be set below 250W: `--setpower 200` -> "Value cannot be less than:
    250W". The sweep axis of the plan (250->140W) is therefore
    hardware-blocked on standard firmware; the 250W cap is the only
    achievable point on this hardware and is already the default.
  * one card (rocm GPU3 on this machine) is factory-locked at a 150W MAX.
    That is a firmware/BIOS limit, not a rocm-smi one.

So `set_power` here:
  * detects the floor/max per GPU and reports it (so the UI/docs can show
    why a lower cap is unavailable instead of silently failing),
  * applies a cap only when it is within the hardware's settable range,
  * never touches the production lanes (caller passes the target GPU index;
    the engine resolves slot->GPU, and the campaign rule keeps dev work on
    dev ports).

Pure subprocess + string parsing (no ROCm python deps), unit-testable with
a fake rocm-smi.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Optional

# rocm-smi format: "GPU[3]\t: Max Graphics Package Power (W): 250.0"
# The number comes AFTER the "(W):" label, not before it.
_POWER_RE = re.compile(r"Power\s*\(W\)\s*:\s*(\d+(?:\.\d+)?)")


@dataclass
class PowerRange:
    """Settable power window for one GPU (watts)."""

    gpu: int
    min_w: Optional[float] = None   # lowest the card allows (floor)
    max_w: Optional[float] = None   # highest the card allows (cap)
    error: Optional[str] = None

    @property
    def settable(self) -> bool:
        return self.min_w is not None and self.max_w is not None


def _run_rocm_smi(rocm_smi: str, args: list[str], timeout: float = 10.0) -> dict:
    """Run rocm-smi; returns {ok, stdout, error}. Note: rocm-smi exits 0 even
    when it refuses a set, so `ok` is parsed from the output text, not rc."""
    try:
        p = subprocess.run([rocm_smi, *args], capture_output=True, text=True,
                           timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return {"ok": False, "stdout": "", "error": str(e)}
    out = p.stdout + p.stderr
    ok = not re.search(r"Unable|Invalid|Error|cannot be", out, re.I)
    return {"ok": ok, "stdout": out, "error": None if ok else out[-300:]}


def _first_watts(text: str) -> Optional[float]:
    m = _POWER_RE.search(text)
    return float(m.group(1)) if m else None


def power_range(rocm_smi: str = "rocm-smi", gpu: int = 0) -> PowerRange:
    """Read a GPU's settable min/max package power from rocm-smi."""
    try:
        out_min = _run_rocm_smi(rocm_smi, ["-d", str(gpu), "--showminpower"])
        out_max = _run_rocm_smi(rocm_smi, ["-d", str(gpu), "--showmaxpower"])
        return PowerRange(gpu=gpu,
                          min_w=_first_watts(out_min["stdout"]),
                          max_w=_first_watts(out_max["stdout"]))
    except (OSError, subprocess.SubprocessError) as e:
        return PowerRange(gpu=gpu, error=str(e))


def set_power(gpu: int = 0, watts: int = 250, rocm_smi: str = "rocm-smi") -> dict:
    """Try to set a package power cap on `gpu`.

    Returns {ok, gpu, requested_w, applied, reason?, range?}. Never raises
    for a rejected cap (that's a normal, expected outcome on locked cards);
    only raises for a missing/unusable rocm-smi binary.

    If the requested watts is below the card's floor or above its max, the
    call is NOT made (rocm-smi would reject it) and the reason is reported.
    """
    rng = power_range(rocm_smi, gpu)
    if rng.error:
        return {"ok": False, "gpu": gpu, "requested_w": watts,
                "applied": False, "reason": f"rocm-smi error: {rng.error}"}
    # Pre-flight checks only when the range is readable; otherwise fall
    # through to the set attempt (rocm-smi's own rejection is authoritative -
    # this driver reports no --showminpower, so the floor surfaces there).
    if rng.settable:
        if watts < (rng.min_w or 0):
            return {"ok": False, "gpu": gpu, "requested_w": watts,
                    "applied": False,
                    "reason": f"below hardware floor {rng.min_w:.0f}W "
                              f"(card refuses < {rng.min_w:.0f}W)",
                    "range": {"min": rng.min_w, "max": rng.max_w}}
        if watts > rng.max_w:
            return {"ok": False, "gpu": gpu, "requested_w": watts,
                    "applied": False,
                    "reason": f"above hardware max {rng.max_w:.0f}W",
                    "range": {"min": rng.min_w, "max": rng.max_w}}
    res = _run_rocm_smi(rocm_smi, ["-d", str(gpu), "--setpower", str(watts)])
    if res["ok"]:
        return {"ok": True, "gpu": gpu, "requested_w": watts, "applied": True,
                "reason": "", "range": {"min": rng.min_w, "max": rng.max_w}}
    return {"ok": False, "gpu": gpu, "requested_w": watts, "applied": False,
            "reason": _clean_reason(res.get("stdout", "") or res.get("error") or ""),
            "range": {"min": rng.min_w, "max": rng.max_w}}


def _clean_reason(out: str) -> str:
    """Collapse a rocm-smi rejection to its essential message."""
    m = re.search(r"Value cannot be less than:\s*(\d+(?:\.\d+)?)W", out)
    if m:
        return f"below hardware floor {m.group(1)}W (rocm-smi refused)"
    if re.search(r"Invalid|out of range|greater than", out, re.I):
        return "above hardware max (rocm-smi refused)"
    s = (out or "").strip()
    return s[-300:] if s else "rocm-smi returned nonzero"
