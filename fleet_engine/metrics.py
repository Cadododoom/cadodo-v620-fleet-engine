"""Live metrics for the control UI: llama-server log parsing + rocm-smi power.

Pure Python, no display required — everything here is unit-testable headless.

Sources (all read-only):
  * per-slot server log (llama.cpp "slot print_timing" lines):
      - completed-task summary: eval time = X ms / N tokens (ms per token,
        tokens per second)  -> last decode rate
      - live progress:       tg_3s = Y t/s   -> 3-second rolling decode rate
        while a request is generating (the "fresh" number, same approach the
        original fleet_panel used)
      - prompt processing:   t = X s / Y tokens per second -> last prefill rate
      - draft acceptance = A (B accepted / C generated)  -> spec telemetry
  * rocm-smi --showpower (or --showuse) for package power and VRAM usage.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional

V620_PCI_DEVICE_ID = "73A1"  # Radeon Pro V620 (gfx1031), same filter as detector

# llama.cpp log line shapes (verified against production fleet logs, 2026-09-06)
RE_EVAL = re.compile(
    r"eval time\s*=\s*(?P<ms>[\d.]+)\s*ms\s*/\s*(?P<n>\d+)\s*tokens\s*"
    r"\(\s*(?P<mspt>[\d.]+)\s*ms per token,\s*(?P<tps>[\d.]+)\s*tokens per second"
)
# in-flight prefill: "prompt processing, n_tokens = 2048, progress = 0.35, "
# "t = 0.12 s / 16824.51 tokens per second"
RE_PROMPT_LIVE = re.compile(
    r"prompt processing, n_tokens\s*=\s*\d+, progress\s*=\s*[\d.]+,?\s*"
    r"t\s*=\s*[\d.]+\s*s\s*/\s*(?P<tps>[\d.]+)\s*tokens per second"
)
# completed prefill: "prompt eval time = 233288.82 ms / 71457 tokens (...)"
RE_PROMPT_DONE = re.compile(
    r"prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(?P<n>\d+)\s*tokens\s*\(.*?"
    r"(?P<tps>[\d.]+)\s*tokens per second"
)
RE_SPEC = re.compile(
    r"draft acceptance\s*=\s*(?P<acc>[\d.]+)\s*\(\s*(?P<accepted>\d+)\s*accepted"
    r"\s*/\s*(?P<generated>\d+)\s*generated\)"
    r"(?:.*?mean len\s*=\s*(?P<mlen>[\d.]+))?"
)
RE_TG3S = re.compile(
    r"slot print_timing: id\s+\d+ \| task\s+\d+ \| n_gen\s*=\s*\d+,"
    r".*?tg_3s\s*=\s*(?P<tps>[\d.]+)\s*t/s"
)


@dataclass
class SlotLogStats:
    """Latest rates for one slot, parsed from the tail of its log file."""

    path: str = ""
    last_pos: int = 0
    decode_tps: Optional[float] = None       # last completed-task decode rate
    decode_n: Optional[int] = None           # tokens in that task
    decode_ts: float = 0.0
    live_decode_tps: Optional[float] = None  # in-flight tg_3s (3s rolling)
    live_decode_ts: float = 0.0
    prefill_tps: Optional[float] = None      # last completed prompt-eval rate
    prefill_n: Optional[int] = None
    prefill_ts: float = 0.0
    live_prefill_tps: Optional[float] = None  # in-flight "prompt processing" rate
    live_prefill_ts: float = 0.0
    spec_accept: Optional[float] = None      # latest draft acceptance
    spec_mean_len: Optional[float] = None
    spec_accepted: int = 0
    spec_generated: int = 0
    spec_ts: float = 0.0

    def header_decode(self, live_window_s: float = 5.0, now: Optional[float] = None) -> Optional[float]:
        """Fresh decode rate: live tg_3s if the request is still generating
        (seen within live_window_s), else the last completed-task rate."""
        import time

        now = now if now is not None else time.time()
        if self.live_decode_tps is not None and now - self.live_decode_ts < live_window_s:
            return self.live_decode_tps
        return self.decode_tps

    def header_prefill(self, live_window_s: float = 5.0, now: Optional[float] = None) -> Optional[float]:
        """Fresh prefill rate: in-flight rate while a prompt is streaming
        (seen within live_window_s), else the last completed prompt-eval rate."""
        import time

        now = now if now is not None else time.time()
        if self.live_prefill_tps is not None and now - self.live_prefill_ts < live_window_s:
            return self.live_prefill_tps
        return self.prefill_tps


def parse_log_chunk(chunk: str, stats: SlotLogStats, now: Optional[float] = None) -> None:
    """Update stats in place from one chunk of llama-server log text."""
    import time

    now = now if now is not None else time.time()
    for line in chunk.splitlines():
        m = RE_EVAL.search(line)
        if m:
            stats.decode_tps = float(m.group("tps"))
            stats.decode_n = int(m.group("n"))
            stats.decode_ts = now
        m = RE_PROMPT_LIVE.search(line)
        if m:
            stats.live_prefill_tps = float(m.group("tps"))
            stats.live_prefill_ts = now
        m = RE_PROMPT_DONE.search(line)
        if m:
            stats.prefill_tps = float(m.group("tps"))
            stats.prefill_n = int(m.group("n"))
            stats.prefill_ts = now
            stats.live_prefill_tps = None  # prompt finished
        m = RE_SPEC.search(line)
        if m:
            stats.spec_accept = float(m.group("acc"))
            stats.spec_mean_len = float(m.group("mlen")) if m.group("mlen") else None
            stats.spec_accepted = int(m.group("accepted"))
            stats.spec_generated = int(m.group("generated"))
            stats.spec_ts = now
        m = RE_TG3S.search(line)
        if m:
            stats.live_decode_tps = float(m.group("tps"))
            stats.live_decode_ts = now


class LogTailer:
    """Incremental tail of one log file (at most one chunk per poll).

    Tracks a character offset (text mode); if the file shrinks
    (rotation/truncation) we restart from 0.
    """

    def __init__(self, path: str, max_chars: int = 64 * 1024) -> None:
        self.path = path
        self.max_chars = max_chars
        self.pos = 0

    def poll(self) -> Optional[str]:
        """Return new text since the last poll, or None."""
        import os

        try:
            size = os.path.getsize(self.path)
        except OSError:
            return None
        if size < self.pos:  # truncated/rotated (byte size vs char offset)
            self.pos = 0
        if size == self.pos:
            return None
        read_from = max(self.pos, max(0, size - self.max_chars))
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(read_from)
                chunk = f.read()
                self.pos += len(chunk)
        except OSError:
            return None
        return chunk or None


@dataclass
class PowerTable:
    """rocm-smi snapshot: per-GPU-index package power (W) and VRAM used/total."""

    power_w: dict[int, float] = field(default_factory=dict)
    vram_used: dict[int, int] = field(default_factory=dict)
    vram_total: dict[int, int] = field(default_factory=dict)


_RE_POWER = re.compile(
    r"GPU\[(\d+)\].*?Package Power\s*\(\w+\)\s*:\s*([\d.]+)"
)
_RE_VRAM_TOTAL = re.compile(r"GPU\[(\d+)\].*?VRAM Total Memory\s*\(\w+\)\s*:\s*([\d.]+)")
_RE_VRAM_USED = re.compile(
    r"GPU\[(\d+)\].*?VRAM Total Used Memory\s*\(\w+\)\s*:\s*([\d.]+)"
)


def rocm_smi_power(rocm_smi_bin: str = "rocm-smi", timeout: float = 10.0) -> PowerTable:
    """Parse `rocm-smi --showpower --showmeminfo vram` (output format varies by
    ROCm version; we regex whatever is present). Read-only, best-effort."""
    table = PowerTable()
    try:
        out = subprocess.run(
            [rocm_smi_bin, "--showpower", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=timeout,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return table
    for line in out.splitlines():
        m = _RE_POWER.search(line)
        if m:
            table.power_w[int(m.group(1))] = float(m.group(2))
        m = _RE_VRAM_TOTAL.search(line)
        if m:
            table.vram_total[int(m.group(1))] = int(float(m.group(2)))
        m = _RE_VRAM_USED.search(line)
        if m:
            table.vram_used[int(m.group(1))] = int(float(m.group(2)))
    return table


def sysfs_vram_bytes(pci_addr: str, drm_dir: str = "/sys/class/drm") -> tuple[int, int]:
    """(vram_total, vram_used) bytes for a card by PCI address, from sysfs.
    (0, 0) if not found. Read-only."""
    for card in _card_dirs(drm_dir):
        dev = card / "device"
        ue = dev / "uevent"
        if not ue.is_file():
            continue
        try:
            fields = dict(
                line.split("=", 1) for line in ue.read_text().splitlines() if "=" in line
            )
        except OSError:
            continue
        if fields.get("PCI_SLOT_NAME") != pci_addr:
            continue
        vals = []
        for fname in ("mem_info_vram_total", "mem_info_vram_used"):
            try:
                vals.append(int((dev / fname).read_text().strip()))
            except (OSError, ValueError):
                vals.append(0)
        return (vals[0], vals[1])
    return (0, 0)


def _card_dirs(drm_dir: str) -> list:
    """Real DRM card dirs only (card0, card1, ...), skipping connectors like
    card1-DP-1 / card2-Writeback-2."""
    from pathlib import Path

    out = []
    try:
        for c in sorted(Path(drm_dir).glob("card*")):
            if re.fullmatch(r"card\d+", c.name):
                out.append(c)
    except OSError:
        pass
    return out


def rocm_gpu_indices_by_pci(drm_dir: str = "/sys/class/drm") -> dict[str, int]:
    """Map PCI address -> rocm-smi GPU index (order of /sys/class/drm/card*).

    rocm-smi enumerates cards in the same order the kernel exposes them, which
    is the sorted cardN order on this box. Covers ALL amdgpu cards (including
    non-V620s), since the power table is keyed by that same index."""
    out: dict[str, int] = {}
    for i, card in enumerate(_card_dirs(drm_dir)):
        ue = card / "device" / "uevent"
        try:
            fields = dict(line.split("=", 1) for line in ue.read_text().splitlines() if "=" in line)
        except OSError:
            continue
        if fields.get("DRIVER") != "amdgpu":
            continue
        addr = fields.get("PCI_SLOT_NAME", "")
        if addr:
            out[addr] = i
    return out


def v620_rocm_indices(drm_dir: str = "/sys/class/drm") -> dict[str, int]:
    """Map PCI address -> rocm-smi index, V620 cards only, in rocm-smi order."""
    all_map = rocm_gpu_indices_by_pci(drm_dir)
    return {
        addr: idx
        for addr, idx in all_map.items()
        if _v620_pci(drm_dir, addr)
    }


def _v620_pci(drm_dir: str, pci_addr: str) -> bool:
    for c in _card_dirs(drm_dir):
        ue = c / "device" / "uevent"
        try:
            fields = dict(line.split("=", 1) for line in ue.read_text().splitlines() if "=" in line)
        except OSError:
            continue
        if fields.get("PCI_SLOT_NAME") == pci_addr:
            try:
                dev_id = (c / "device" / "device").read_text().strip()[2:].upper()
            except OSError:
                return False
            return dev_id == V620_PCI_DEVICE_ID
    return False


@dataclass
class ProdLane:
    """One production fleet lane (read-only view from the gpuN.sh scripts)."""

    name: str
    port: int
    log: str
    hip_pin: Optional[int] = None
    model: Optional[str] = None
    slot_number: Optional[int] = None  # 1..N from the gpuN.sh filename
    pci_addr: str = ""
    vram_total: int = 0
    vram_used: int = 0


_RE_HIP = re.compile(r"^\s*GPU_INDEX\s*=\s*(\d+)", re.M)
_RE_PORT = re.compile(r"^\s*PORT\s*=\s*(\d+)", re.M)
_RE_NAME = re.compile(r'^\s*NAME\s*=\s*"?([\w.\-]+)', re.M)
_RE_MODEL_ENV = re.compile(r'^\s*MODEL\s*=\s*"?([^"\s]+)', re.M)


def parse_prod_lanes(fleet_dir: str, max_lanes: int = 16) -> list[ProdLane]:
    """Parse gpu1.sh..gpuN.sh + config.env from a production fleet dir
    (read-only) into ProdLane entries: name, port, log path, pinned HIP index,
    model path (MODEL comes from config.env, which the scripts source)."""
    from pathlib import Path

    base = Path(fleet_dir)
    if not base.is_dir():
        return []
    model: Optional[str] = None
    try:
        env = (base / "config.env").read_text()
        m = _RE_MODEL_ENV.search(env)
        model = m.group(1) if m else None
    except OSError:
        pass
    lanes: list[ProdLane] = []
    for i in range(1, max_lanes + 1):
        script = base / f"gpu{i}.sh"
        if not script.is_file():
            continue
        try:
            text = script.read_text()
        except OSError:
            continue
        m_hip = _RE_HIP.search(text)
        m_port = _RE_PORT.search(text)
        m_name = _RE_NAME.search(text)
        name = f"prod-{i}"
        if m_name:
            name = f"prod-{m_name.group(1)}"
        lanes.append(
            ProdLane(
                name=name,
                port=int(m_port.group(1)) if m_port else 45600 + i - 1,
                log=str(base / "logs" / f"gpu{i}.log"),
                hip_pin=int(m_hip.group(1)) if m_hip else None,
                model=model,
                slot_number=i,
            )
        )
    # attribute PCI addresses: HIP-pinned lanes match detected V620s by order
    # (same ordering rule as the detector; read-only, best-effort)
    from .detector import detect_v620_slots

    det = detect_v620_slots(drm_dir="/sys/class/drm", llama_bin=None)
    for lane in lanes:
        if lane.hip_pin and 1 <= lane.hip_pin <= len(det.slots):
            lane.pci_addr = det.slots[lane.hip_pin - 1].pci_addr
    return lanes
