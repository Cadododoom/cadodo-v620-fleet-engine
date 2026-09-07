"""Standardized benchmark prompts for reproducible throughput measurement.

Each prompt has a fixed input (system + user) and a fixed max_tokens so
results are comparable across runs, models, and hardware.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchPrompt:
    """One standardized benchmark case."""

    name: str
    system: str
    user: str
    max_tokens: int


PROMPT_SET_VERSION = "v1.0-2026-09"

PROMPTS: list[BenchPrompt] = [
    BenchPrompt(
        name="short-64",
        system="You are a helpful assistant. Reply concisely.",
        user="List the first 16 prime numbers, one per line.",
        max_tokens=64,
    ),
    BenchPrompt(
        name="medium-256",
        system=(
            "You are a senior software engineer. Explain concepts clearly "
            "with short code examples in Python when helpful."
        ),
        user=(
            "Explain how speculative decoding works in LLM inference, "
            "covering both draft-model and MTP (multi-token prediction) "
            "approaches. Include the acceptance/rejection loop and why "
            "it preserves output distribution."
        ),
        max_tokens=256,
    ),
    BenchPrompt(
        name="long-512",
        system=(
            "You are a systems architect. Write thorough, well-structured "
            "answers with headings. Cite trade-offs explicitly."
        ),
        user=(
            "Design a distributed key-value store that supports: "
            "(1) consistent reads within a single shard, "
            "(2) eventual consistency across shards, "
            "(3) CRDT-based conflict-free merge for concurrent writes, "
            "(4) TTL-based expiration with lazy + eager cleanup, "
            "(5) a gossip protocol for membership and failure detection. "
            "Cover the data model, partitioning strategy, replication "
            "protocol, failure handling, and a concrete sequence for a "
            "two-node split-brain scenario."
        ),
        max_tokens=512,
    ),
]

# Concurrency sweep levels
CONCURRENCY_LEVELS: list[int] = [1, 2, 4]


def get_prompt(name: str) -> BenchPrompt:
    for p in PROMPTS:
        if p.name == name:
            return p
    raise KeyError(f"unknown prompt {name!r}; available: {[p.name for p in PROMPTS]}")


def all_prompts() -> list[BenchPrompt]:
    return list(PROMPTS)
