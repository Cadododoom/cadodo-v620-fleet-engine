"""Model registry (skeleton): scan local GGUF folders, parse quant/size metadata.

Phase 4 implements the gguf header reader + picker. The registry replaces the
hand-picked MODEL variable from config.env.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelInfo:
    path: str
    name: str
    size_bytes: int
    quant: str = "unknown"
    arch: str = "unknown"
    native_ctx: int = 0


def scan_models(dirs: list[str]) -> list[ModelInfo]:
    """Scan GGUF files under the given directories. Phase 4: implement."""
    raise NotImplementedError("Phase 4: GGUF folder scan + metadata parse")
