"""Config store: slots.json schema v1.

Per-slot config, atomic JSON persistence, round-trip safe.
Phase 2 will add the detector that populates `gpu` and `name`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = 1
MAX_SLOTS = 16
DEFAULT_DEV_PORT_BASE = 45700  # campaign rule: dev servers on 45700+ only

VALID_SPEC_TYPES = ("none", "mtp", "ngram-mod", "draft-mtp,ngram-mod")


@dataclass
class SpecConfig:
    """Speculative decoding knobs, mirroring the production launcher flags.

    spec_type is a comma-joined llama.cpp --spec-type value.
    """

    spec_type: str = "draft-mtp,ngram-mod"
    mtp_n_max: int = 3
    mtp_n_min: int = 1
    ngram_n_min: int = 4
    ngram_n_max: int = 16
    ngram_match: int = 24
    draft_threads: int = 8
    draft_model: Optional[str] = None


@dataclass
class SlotConfig:
    """One llama-server slot on one V620."""

    slot: int
    name: str
    gpu: int
    host: str = "0.0.0.0"
    port: int = DEFAULT_DEV_PORT_BASE
    model: str = ""
    mmproj: Optional[str] = None
    ctx: int = 262144
    concurrency: int = 1
    rope_scale: float = 1.0
    yarn_orig_ctx: int = 262144
    kv_type: str = "q4_0"
    kv_unified: bool = True
    num_gpu_layers: str = "all"
    threads: int = 14
    power_cap_watts: Optional[int] = None
    harness_link: bool = False
    spec: SpecConfig = field(default_factory=SpecConfig)

    def __post_init__(self) -> None:
        if not 1 <= self.slot <= MAX_SLOTS:
            raise ValueError(f"slot must be in 1..{MAX_SLOTS}, got {self.slot}")
        if self.port < 1024:
            raise ValueError(f"port must be >= 1024, got {self.port}")
        if self.spec.spec_type not in VALID_SPEC_TYPES:
            raise ValueError(f"spec_type must be one of {VALID_SPEC_TYPES}")


def slot_from_dict(data: dict[str, Any], slot: Optional[int] = None) -> SlotConfig:
    """Build a SlotConfig from a parsed slots.json entry."""
    spec = SpecConfig(**data.get("spec", {}))
    cfg = {k: v for k, v in data.items() if k != "spec"}
    if slot is not None:
        cfg.setdefault("slot", slot)
    return SlotConfig(spec=spec, **cfg)


def slot_to_dict(cfg: SlotConfig) -> dict[str, Any]:
    return asdict(cfg)


class ConfigStore:
    """Reads/writes slots.json with atomic replacement."""

    def __init__(self, path: str) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        if not os.path.exists(self.path):
            return {"schema_version": SCHEMA_VERSION, "slots": {}}
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("slots", {})
        return data

    def save(self, data: dict[str, Any]) -> None:
        data["schema_version"] = SCHEMA_VERSION
        d = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def get_slot(self, slot: int) -> SlotConfig:
        data = self.load()
        entry = data["slots"].get(str(slot))
        if entry is None:
            raise KeyError(f"no slot {slot} in {self.path}")
        return slot_from_dict(entry, slot=slot)

    def set_slot(self, cfg: SlotConfig) -> None:
        data = self.load()
        data["slots"][str(cfg.slot)] = slot_to_dict(cfg)
        self.save(data)
