"""Connector (skeleton): Hermes + OpenCode provider auto-write, drift watch, reconnect.

Phase 6/7 implement the writers. Single-writer rule: only the engine touches the
provider block; drift detection is read-only and alerts.
"""

from __future__ import annotations


def register_hermes_provider(slot_name: str, port: int, model: str) -> None:
    """Write/update the v620-<slot> provider entry. Phase 6: implement."""
    raise NotImplementedError("Phase 6: Hermes provider auto-write")


def register_opencode_endpoint(port: int, model: str) -> None:
    """Write the OpenCode desktop endpoint config. Phase 7: implement."""
    raise NotImplementedError("Phase 7: OpenCode endpoint write")
