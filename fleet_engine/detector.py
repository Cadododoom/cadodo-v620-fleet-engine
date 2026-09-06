"""Slot detector: enumerate GPUs, keep V620 (gfx1031) only, assign slots 1..N.

Two evidence sources, combined:
  1. sysfs PCI scan of /sys/class/drm/card*/device - vendor/device/subsys IDs,
     PCI address, VRAM total/used (mem_info_vram_*).
  2. llama-server --list-devices probe, one per HIP index - the engine's own
     view (HIP order is NOT rocm-smi order, and can differ across ROCm
     installs; the production launcher pins by HIP index and verifies the card
     string this same way).

The probe is the authoritative hip_index source; sysfs supplies identity and
VRAM. When the llama binary is absent or the probe fails, the detector
degrades to "N V620s detected, hip_index unknown - user assigns" and records
that on each slot (per plan section 6 risk row).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

V620_PCI_DEVICE_ID = "73A1"
V620_VENDOR_ID = "1002"
MAX_SLOTS = 16

_HIP_RE = re.compile(
    r"ROCm\d+:\s+(?P<name>[^(]+)\((?P<total>\d+)\s+MiB,\s*(?P<free>\d+)\s+MiB free\)"
)


@dataclass
class DetectedSlot:
    slot: int
    name: str
    pci_addr: str
    hip_index: int | None
    device_name: str
    vram_total_bytes: int
    vram_used_bytes: int


@dataclass
class Detection:
    slots: list[DetectedSlot]
    probe_ok: bool
    probe_error: str | None = None


def _read_id_file(path: Path) -> str:
    try:
        raw = path.read_text().strip()
        return raw[2:].upper() if raw.startswith("0x") else raw.upper()
    except OSError:
        return ""


def scan_sysfs_v620(drm_dir: str = "/sys/class/drm") -> dict[str, dict]:
    """PCI scan: return {pci_addr: {vendor, device, subsys, vram_total, vram_used}}
    for every amdgpu card under drm_dir. Read-only.

    Vendor/device come from device/vendor + device/device (the uevent on this
    box omits PCI_VENDOR_ID); PCI address + driver from uevent.
    """
    out: dict[str, dict] = {}
    base = Path(drm_dir)
    if not base.is_dir():
        return out
    for card in sorted(base.glob("card*")):
        dev = card / "device"
        uevent = dev / "uevent"
        if not uevent.is_file():
            continue
        fields: dict[str, str] = {}
        try:
            for line in uevent.read_text().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    fields[k] = v
        except OSError:
            continue
        if fields.get("DRIVER") != "amdgpu":
            continue
        vendor = _read_id_file(dev / "vendor")
        device = _read_id_file(dev / "device")
        if vendor != V620_VENDOR_ID:
            continue
        addr = fields.get("PCI_SLOT_NAME", "")
        info: dict[str, object] = {
            "vendor": vendor,
            "device": device,
            "subsys": fields.get("PCI_SUBSYS_ID", "").split(":")[-1],
        }
        for key, field in (("vram_total", "mem_info_vram_total"),
                           ("vram_used", "mem_info_vram_used")):
            try:
                info[key] = int((dev / field).read_text().strip())
            except (OSError, ValueError):
                info[key] = 0
        out[addr] = info
    return out


def probe_hip_devices(
    llama_bin: str,
    hip_index: int,
    roc_vendor: str | None = None,
    timeout: int = 30,
) -> tuple[str, int, int] | None:
    """Run `llama-server --list-devices` pinned to one HIP index.

    Returns (device_name, vram_total_mib, vram_free_mib) for the first ROCm
    line, or None if the binary is missing/failed. Read-only: the engine
    enumerates and exits.
    """
    if not llama_bin or not Path(llama_bin).is_file():
        return None
    env = {"PATH": "/usr/bin:/bin", "HIP_VISIBLE_DEVICES": str(hip_index)}
    if roc_vendor:
        env["LD_LIBRARY_PATH"] = roc_vendor
    try:
        proc = subprocess.run(
            [llama_bin, "--list-devices"],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    text = proc.stdout + proc.stderr
    for line in text.splitlines():
        m = _HIP_RE.search(line)
        if m:
            return m.group("name").strip(), int(m.group("total")), int(m.group("free"))
    return None


def _v620_name_match(name: str) -> bool:
    return "V620" in name.upper()


def _pci_sort_key(addr: str) -> tuple:
    out = []
    for p in addr.split(":"):
        try:
            out.append(int(p, 16))
        except ValueError:
            out.append(0)
    return tuple(out)


def detect_v620_slots(
    drm_dir: str = "/sys/class/drm",
    llama_bin: str | None = None,
    roc_vendor: str | None = None,
    max_hip: int = 16,
    probe: Callable[[int], tuple[str, int, int] | None] | None = None,
) -> Detection:
    """Detect V620s, assign stable slots 1..N by PCI bus address, and resolve
    hip_index via the engine probe when a llama-server binary (or injected
    probe) is available.

    The probe reports one card per HIP index but cannot distinguish identical
    V620s by name alone, so HIP indices are attributed in order of PCI address
    (first probed V620 -> first sorted address). The runtime re-verifies the
    pinned card at start time (same check as the production launcher), so a
    mis-attribution fails loudly rather than silently.
    """
    v620s = {
        addr: info
        for addr, info in scan_sysfs_v620(drm_dir).items()
        if str(info.get("device")) == V620_PCI_DEVICE_ID
    }
    sorted_addrs = sorted(v620s, key=_pci_sort_key)
    probe_ok = False
    probe_error: str | None = None

    hip_by_addr: dict[str, int] = {}
    if llama_bin or probe is not None:
        probe_fn = probe or (lambda i: probe_hip_devices(llama_bin or "", i, roc_vendor))
        for hip in range(max_hip):
            res = probe_fn(hip)
            if res is None or not _v620_name_match(res[0]):
                continue
            next_unmatched = next((a for a in sorted_addrs if a not in hip_by_addr), None)
            if next_unmatched is None:
                break
            hip_by_addr[next_unmatched] = hip
        probe_ok = len(hip_by_addr) == len(sorted_addrs)
        if not probe_ok:
            probe_error = (
                f"probe resolved {len(hip_by_addr)} of {len(sorted_addrs)} sysfs "
                "V620s; hip_index unknown for the rest (user assigns)"
            )
    else:
        probe_error = "no llama-server binary given; hip_index unknown (user assigns)"

    slots = [
        DetectedSlot(
            slot=i + 1,
            name=f"v620-{i + 1}",
            pci_addr=addr,
            hip_index=hip_by_addr.get(addr),
            device_name="AMD Radeon Pro V620",
            vram_total_bytes=int(v620s[addr].get("vram_total", 0)),
            vram_used_bytes=int(v620s[addr].get("vram_used", 0)),
        )
        for i, addr in enumerate(sorted_addrs[:MAX_SLOTS])
    ]
    return Detection(slots=slots, probe_ok=probe_ok, probe_error=probe_error)
