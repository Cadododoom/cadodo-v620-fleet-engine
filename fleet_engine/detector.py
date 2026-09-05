"""Slot detector (skeleton): enumerate GPUs, keep V620 (gfx1031), assign slots 1..16.

Phase 2 implements the real scan (rocm-smi + /sys/kernel/drm). The production
launcher pins by HIP index and verifies the card string with
`llama-server --list-devices`; the detector must reproduce that filter.
"""

from __future__ import annotations

from dataclasses import dataclass

V620_DEVICE_ID = "gfx1031"
MAX_SLOTS = 16


@dataclass
class DetectedSlot:
    hip_index: int
    device_name: str
    vram_bytes: int
    slot: int


def detect_slots() -> list[DetectedSlot]:
    """Scan for V620 GPUs and assign slots. Phase 2: implement via sysfs/rocm-smi."""
    raise NotImplementedError("Phase 2: slot detector scan (rocm-smi + /sys/kernel/drm)")
