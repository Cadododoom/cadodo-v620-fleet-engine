"""Unit tests for the slot detector and its merge into the config store.

The probe is injected (no engine subprocess in unit tests); one live
sysfs-only detection test runs against the real machine and asserts the
fleet size and slot stability.
"""

from __future__ import annotations

from fleet_engine.config_store import ConfigStore, slot_from_dict
from fleet_engine.detector import (
    DetectedSlot,
    Detection,
    _pci_sort_key,
    detect_v620_slots,
    scan_sysfs_v620,
)

VRAM_32G = 32195477504


def _fake_probe(hip_to_name: dict[int, str]):
    def probe(hip: int):
        name = hip_to_name.get(hip)
        if name is None:
            return None
        return name, VRAM_32G // (1024**2), 1000
    return probe


def _fake_sysfs(tmp_path, addrs: list[str]) -> str:
    drm = tmp_path / "drm"
    for i, addr in enumerate(addrs):
        dev = drm / f"card{i}" / "device"
        dev.mkdir(parents=True)
        (dev / "uevent").write_text(
            "DRIVER=amdgpu\n"
            "PCI_ID=1002:73A1\n"
            "PCI_SUBSYS_ID=1002:0E34\n"
            f"PCI_SLOT_NAME={addr}\n"
        )
        (dev / "vendor").write_text("0x1002\n")
        (dev / "device").write_text("0x73a1\n")
        (dev / "mem_info_vram_total").write_text(str(VRAM_32G))
        (dev / "mem_info_vram_used").write_text("0")
    return str(drm)


def test_scan_sysfs_filters_to_amdgpu_only(tmp_path):
    drm = tmp_path / "drm"
    (drm / "card0" / "device").mkdir(parents=True)
    (drm / "card0" / "device" / "uevent").write_text(
        "DRIVER=nouveau\nPCI_ID=10DE:2684\nPCI_SLOT_NAME=0000:01:00.0\n"
    )
    (drm / "card1" / "device").mkdir(parents=True)
    (drm / "card1" / "device" / "uevent").write_text(
        "DRIVER=amdgpu\nPCI_ID=1002:731F\nPCI_SLOT_NAME=0000:c7:00.0\n"  # RX 5600, not V620
    )
    (drm / "card1" / "device" / "vendor").write_text("0x1002\n")
    (drm / "card1" / "device" / "device").write_text("0x731f\n")
    found = scan_sysfs_v620(str(drm))
    assert "0000:01:00.0" not in found
    assert "0000:c7:00.0" in found  # amdgpu card present in raw scan
    assert str(found["0000:c7:00.0"]["device"]) == "731F"


def test_detect_assigns_slots_and_hip(tmp_path):
    drm = _fake_sysfs(tmp_path, ["0000:c7:00.0", "0000:83:00.0", "0000:03:00.0"])
    # non-V620 card present (c7) - must be excluded by device ID filter
    probe = _fake_probe({1: "AMD Radeon Pro V620", 2: "AMD Radeon Pro V620", 3: "AMD Radeon Pro V620"})
    det = detect_v620_slots(drm_dir=drm, probe=probe)
    assert len(det.slots) == 3
    assert det.probe_ok
    assert [s.slot for s in det.slots] == [1, 2, 3]
    # slots sorted by PCI address; HIP indices attributed in discovery order
    assert [s.pci_addr for s in det.slots] == ["0000:03:00.0", "0000:83:00.0", "0000:c7:00.0"]
    assert [s.hip_index for s in det.slots] == [1, 2, 3]
    assert all(s.vram_total_bytes == VRAM_32G for s in det.slots)
    assert [s.name for s in det.slots] == ["v620-1", "v620-2", "v620-3"]


def test_detect_degrades_without_probe(tmp_path):
    drm = _fake_sysfs(tmp_path, ["0000:83:00.0", "0000:03:00.0"])
    det = detect_v620_slots(drm_dir=drm)
    assert len(det.slots) == 2
    assert not det.probe_ok
    assert det.probe_error is not None and "user assigns" in det.probe_error
    assert all(s.hip_index is None for s in det.slots)


def test_detect_partial_probe_marks_not_ok(tmp_path):
    drm = _fake_sysfs(tmp_path, ["0000:83:00.0", "0000:03:00.0"])
    probe = _fake_probe({1: "AMD Radeon Pro V620"})  # only 1 of 2 resolves
    det = detect_v620_slots(drm_dir=drm, probe=probe)
    assert not det.probe_ok
    assert "1 of 2" in (det.probe_error or "")
    assert det.slots[0].hip_index == 1
    assert det.slots[1].hip_index is None


def test_pci_sort_key_orders_addresses():
    keys = sorted(["0000:c7:00.0", "0000:83:00.0", "0000:03:00.0"], key=_pci_sort_key)
    assert keys == ["0000:03:00.0", "0000:83:00.0", "0000:c7:00.0"]


def test_merge_detection_populates_and_preserves(tmp_path):
    drm = _fake_sysfs(tmp_path, ["0000:83:00.0", "0000:03:00.0"])
    probe = _fake_probe({2: "AMD Radeon Pro V620", 3: "AMD Radeon Pro V620"})
    det = detect_v620_slots(drm_dir=drm, probe=probe)
    store = ConfigStore(str(tmp_path / "slots.json"))

    # pre-existing user slot: custom name + gpu must survive the merge
    store.set_slot(slot_from_dict({"slot": 1, "name": "custom-a", "gpu": 7}))

    data = store.merge_detection(det)
    store.save(data)
    loaded = store.load()
    assert loaded["slots"]["1"]["name"] == "custom-a"
    assert loaded["slots"]["1"]["gpu"] == 7
    assert loaded["slots"]["1"]["pci_addr"] == "0000:03:00.0"
    assert loaded["slots"]["2"]["name"] == "v620-2"
    assert loaded["slots"]["2"]["gpu"] == 3
    assert loaded["detected"]["count"] == 2
    assert loaded["detected"]["probe_ok"] is True
    # merged entry round-trips through the typed loader
    assert store.get_slot(2).gpu == 3


def test_detect_live_sysfs_only():
    """Live, read-only: this box has exactly 4 V620s (plus 1 RX 5700 that
    must be filtered out). No engine probe here - that is the CLI's job."""
    det = detect_v620_slots()
    assert len(det.slots) == 4
    assert all(s.hip_index is None for s in det.slots)
    assert not det.probe_ok
    # 32 GB card, ~30.7 GiB usable (engine probe reports 30704 MiB)
    assert all(s.vram_total_bytes > 29 * (1024**3) for s in det.slots)
    # stable slot numbering by PCI order
    addrs = [s.pci_addr for s in det.slots]
    assert addrs == sorted(addrs, key=_pci_sort_key)


def _slot(hip: int, addr: str, slot: int) -> DetectedSlot:
    return DetectedSlot(
        slot=slot, name=f"v620-{slot}", pci_addr=addr, hip_index=hip,
        device_name="AMD Radeon Pro V620", vram_total_bytes=VRAM_32G, vram_used_bytes=0,
    )


def test_detection_dataclass_roundtrip():
    d = Detection(
        slots=[_slot(1, "0000:03:00.0", 1)],
        probe_ok=True,
    )
    assert d.slots[0].hip_index == 1
    assert d.probe_error is None
