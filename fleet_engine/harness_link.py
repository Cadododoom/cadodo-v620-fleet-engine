"""Harness auto-link (skeleton): per-slot endpoint register/deregister with the
PLAN II omni-harness. Phase 10 implements once the PLAN II endpoint spec lands.
"""

from __future__ import annotations


def register_endpoint(slot: int, port: int) -> None:
    raise NotImplementedError("Phase 10: harness endpoint registration (needs PLAN II spec)")


def deregister_endpoint(slot: int) -> None:
    raise NotImplementedError("Phase 10: harness endpoint deregistration")
