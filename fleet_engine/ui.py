"""Control UI: 16-slot grid, live rates, VRAM/power bars, log viewer.

Phase 5 of PLAN I. Polished tkinter front end over the engine modules
(P1-2 detector, P1-3 runtime, metrics log/power parsers). Zero
third-party deps: stdlib tkinter only.

Design notes
------------
* The grid always renders MAX_SLOTS cells (16); undetected slots are shown
  dimmed as "empty" so the layout is stable across machines.
* Two data sources, both read-only:
    - dev slots: slots.json in a state dir (this machine's own dev fleet,
      45700+), with ON/OFF/RESTART buttons wired to Runtime.
    - prod lanes (optional --fleet-dir): read-only view of the production
      fleet (gpuN.sh parsed by metrics.parse_prod_lanes) with live rates,
      VRAM and power from the same log/rocm-smi sources. No control
      buttons: the engine never touches prod.
* Live numbers: header_decode()/header_prefill() (fresh < 5 s, else last
  completed), spec acceptance, VRAM used/total bar, package power bar.
* Headless verification: --screenshot PATH builds the real widget tree on
  X (Xvfb) and dumps it to PNG; --no-interactive does one refresh + exit.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from .metrics import (
    LogTailer,
    PowerTable,
    ProdLane,
    SlotLogStats,
    parse_log_chunk,
    parse_prod_lanes,
    rocm_gpu_indices_by_pci,
    rocm_smi_power,
    sysfs_vram_bytes,
)

MAX_SLOTS = 16
COLS = 4
REFRESH_MS = 3000
LIVE_WINDOW_S = 5.0

# palette (parity with the original fleet_panel.py)
BG = "#101418"
CARD = "#171d23"
BORDER = "#2a333c"
FG = "#dce3ea"
DIM = "#7d8b97"
CYAN = "#25c8e0"
MAGENTA = "#e05ce0"
GOLD = "#f0b429"
GREEN = "#4ad07a"
RED = "#e05555"
EMPTY = "#12161a"
MONO = "DejaVu Sans Mono"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

@dataclass
class SlotView:
    """Renderable state for one grid cell (dev, prod, or empty)."""

    slot: int
    name: str
    kind: str  # "dev" | "prod" | "empty"
    pci: str = ""
    hip: Optional[int] = None
    port: int = 0
    model: str = ""
    running: bool = False
    pid: Optional[int] = None
    health: Optional[str] = None
    log: str = ""
    stats: Optional[SlotLogStats] = None
    vram_total: int = 0
    vram_used: int = 0
    power_w: Optional[float] = None
    power_cap: Optional[int] = None


class FleetModel:
    """Collects the data the UI renders. Pure python; unit-testable."""

    def __init__(self, dev_slots: Optional[list] = None,
                 prod_lanes: Optional[list[ProdLane]] = None,
                 power: Optional[PowerTable] = None,
                 dev_state_dir: str = "") -> None:
        self.dev_slots = dev_slots or []  # list[SlotConfig]
        self.prod_lanes = prod_lanes or []
        self.power = power or PowerTable()
        self.dev_state_dir = dev_state_dir
        self.pci_to_idx = rocm_gpu_indices_by_pci()
        self._tailers: dict[tuple, LogTailer] = {}
        self._stats: dict[tuple, SlotLogStats] = {}
        self._view_cache: dict[int, SlotView] = {}
        self._views: list[SlotView] = []

    # -- hardware ----------------------------------------------------------

    def refresh(self) -> None:
        """Re-poll the rocm-smi power table and attach it to the views."""
        self.power = rocm_smi_power()
        self._views = self.build_views()
        for v in self._views:
            idx = self._view_idx(v)
            if idx is not None:
                v.power_w = self.power.power_w.get(idx)
                v.vram_total = self.power.vram_total.get(idx) or v.vram_total
                v.vram_used = self.power.vram_used.get(idx) or v.vram_used
            elif v.pci:
                t, u = sysfs_vram_bytes(v.pci)
                v.vram_total, v.vram_used = t or v.vram_total, u or v.vram_used

    def _view_idx(self, v: SlotView) -> Optional[int]:
        if v.pci and v.pci in self.pci_to_idx:
            return self.pci_to_idx[v.pci]
        return None

    # -- slots -------------------------------------------------------------

    def build_views(self) -> list[SlotView]:
        """16 SlotViews in slot order: dev slots + prod lanes, rest empty.

        Views are cached per slot (self._view_cache) so live stats/health
        survive refresh cycles (only config-level fields are updated in place).
        """
        cache = self._view_cache
        for cfg in self.dev_slots:
            v = cache.get(cfg.slot)
            if v is None or v.kind != "dev":
                v = SlotView(slot=cfg.slot, name=cfg.name, kind="dev")
                cache[cfg.slot] = v
            v.name = cfg.name
            v.pci = cfg.pci_addr or ""
            v.hip = cfg.gpu
            v.port = cfg.port
            v.model = os.path.basename(cfg.model or "")
            v.log = os.path.join(self.dev_state_dir, "logs", f"{cfg.name}.log")
            v.power_cap = cfg.power_cap_watts
        for lane in self.prod_lanes:
            slot = lane.slot_number or 0
            if not slot or (slot in cache and cache[slot].kind == "dev"):
                continue
            v = cache.get(slot)
            if v is None or v.kind != "prod":
                v = SlotView(slot=slot, name=lane.name, kind="prod")
                cache[slot] = v
            v.name = lane.name
            v.pci = lane.pci_addr or ""
            v.hip = lane.hip_pin
            v.port = lane.port
            v.model = os.path.basename(lane.model or "")
            v.log = lane.log
        # drop slots no longer configured (dev/prod), keep empty + active
        for i in list(cache):
            v = cache[i]
            if v.kind in ("dev", "prod") and not self._still_configured(i, v.kind):
                del cache[i]
        for i in range(1, MAX_SLOTS + 1):
            if i not in cache:
                cache[i] = SlotView(slot=i, name=f"v620-{i}", kind="empty")
        self._views = [cache[i] for i in sorted(cache)]
        return self._views

    def _still_configured(self, slot: int, kind: str) -> bool:
        if kind == "dev":
            return any(c.slot == slot for c in self.dev_slots)
        return any((lane.slot_number or 0) == slot for lane in self.prod_lanes)

    def update_logs(self, now: Optional[float] = None) -> None:
        """Incrementally tail every slot/lane log into SlotLogStats."""
        now = now or time.time()
        for v in self.build_views():
            if v.kind == "empty" or not v.log or not os.path.exists(v.log):
                continue
            key = (v.kind, v.slot)
            tailer = self._tailers.get(key)
            if tailer is None:
                tailer = LogTailer(v.log)
                self._tailers[key] = tailer
            chunk = tailer.poll()
            if chunk:
                st = self._stats.get(key)
                if st is None:
                    st = SlotLogStats(path=v.log)
                    self._stats[key] = st
                parse_log_chunk(chunk, st, now=now)
                v.stats = st

    def update_health(self) -> None:
        """Probe /health for every view that has a port (read-only)."""
        import urllib.request

        for v in self._views:
            if v.kind == "empty" or not v.port:
                continue
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{v.port}/health", timeout=2
                ) as r:
                    v.health = r.read(64).decode("utf-8", "replace").strip()
                    v.running = True
            except Exception:  # noqa: BLE001 - down is a normal state
                v.health = None
                v.running = False

    # -- construction ------------------------------------------------------

    @classmethod
    def for_dev(cls, state_dir: str) -> "FleetModel":
        from .config_store import ConfigStore, slot_from_dict

        store = ConfigStore(os.path.join(state_dir, "slots.json"))
        data = store.load()
        cfgs = [slot_from_dict(d, slot=k) for k, d in sorted(data.get("slots", {}).items())]
        return cls(dev_slots=cfgs, dev_state_dir=state_dir)

    @classmethod
    def for_prod(cls, fleet_dir: str) -> "FleetModel":
        return cls(prod_lanes=parse_prod_lanes(fleet_dir))


# --------------------------------------------------------------------------
# formatting helpers (pure, testable)
# --------------------------------------------------------------------------

def fmt_rate(tps: Optional[float]) -> str:
    return f"{tps:.1f} t/s" if tps is not None else "—"


def fmt_gib(b: int) -> str:
    return f"{b / 1024**3:.1f}G"


def bar_fraction(used: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, used / total))


def slot_status_text(v: SlotView) -> str:
    """One-line status string for a slot view (used by UI + tests)."""
    parts = [v.name]
    if v.port:
        parts.append(f":{v.port}")
    if v.hip is not None:
        parts.append(f"HIP{v.hip}")
    if v.running:
        parts.append("UP")
    elif v.kind != "empty":
        parts.append("down")
    return "  ".join(parts)


@dataclass
class UiState:
    """Everything the UI needs for one refresh cycle."""

    views: list[SlotView] = field(default_factory=list)
    now: float = 0.0
    log_lines: dict[int, list[str]] = field(default_factory=dict)

    def decode_of(self, slot: int) -> Optional[float]:
        v = self.views[slot - 1] if slot <= len(self.views) else None
        if v and v.stats:
            return v.stats.header_decode(LIVE_WINDOW_S, self.now)
        return None

    def prefill_of(self, slot: int) -> Optional[float]:
        v = self.views[slot - 1] if slot <= len(self.views) else None
        if v and v.stats:
            return v.stats.header_prefill(LIVE_WINDOW_S, self.now)
        return None

    def spec_of(self, slot: int) -> Optional[float]:
        v = self.views[slot - 1] if slot <= len(self.views) else None
        if v and v.stats:
            return v.stats.spec_accept
        return None
