"""V620 Tuner: curated option matrix with MEASURED effects.

Design rule (plan P1-9): ship only measured-safe defaults. Every preset
carries its measured numbers, workload, and date so the UI/docs can label
unmeasured variants as such. Numbers come from:

- ~/Documents/llama-v620-fleet/config.env (2026-09-01/02 tuning session,
  single-stream, coding + prose workloads on the production build)
- bench/results/p18-prod-baseline.json (2026-09-07, prompt set v1.0-2026-09)
- production slot log draft-acceptance line (MTP3+ngram: 0.508, mean len 2.58)

This module only builds configs/argv; it never touches GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from .config_store import SlotConfig, SpecConfig
from .runtime_cmd import build_llama_server_cmd

#: Preset defaults are anchored to the verified production config
#: (Qwen3.8-27B UD-IQ4_XS, 528384 ctx = 262144 * 2.0157 YARN).
BASE_SPEC = SpecConfig(
    spec_type="draft-mtp,ngram-mod",
    mtp_n_max=3,
    mtp_n_min=1,
    ngram_n_min=4,
    ngram_n_max=16,
    ngram_match=24,
    draft_threads=8,
)


@dataclass(frozen=True)
class Preset:
    """One tested option combination."""

    name: str
    summary: str
    spec: SpecConfig
    kv_type: str = "q4_0"
    measured: Optional[str] = None  # human-readable measured effect
    measured_date: Optional[str] = None
    safe_default: bool = False
    caveats: str = ""


PRESETS: list[Preset] = [
    Preset(
        name="prod-default",
        summary="Combined MTP3 + ngram-mod, q4_0 KV, YARN 2x, 528k ctx (production)",
        spec=BASE_SPEC,
        kv_type="q4_0",
        measured=(
            "32.1-35.0 t/s client-observed (prompt set v1.0-2026-09, 2026-09-07); "
            "single-stream tuning 2026-09-02: combo 107 t/s coding / ~101 t/s prose, "
            ">= best single drafter in every workload; draft acceptance 0.508 "
            "(4420 accepted / 8707 generated, mean len 2.58)"
        ),
        measured_date="2026-09-07",
        safe_default=True,
    ),
    Preset(
        name="spec-off",
        summary="No speculative decoding (safe fallback when a lane is degraded)",
        spec=replace(BASE_SPEC, spec_type="none"),
        kv_type="q4_0",
        measured=(
            "combo is ~2.05x mtp-only and ~1.0x ngram-only in the 2026-09-02 "
            "single-stream session, so expect roughly half of the combo rate on "
            "mtp-heavy text; exact lane rate unmeasured on 2026-09-07"
        ),
        measured_date=None,
        caveats=(
            "Use when a lane degrades (e.g. g2 at 9.2 t/s in the 2026-09-07 baseline) "
            "to isolate whether the spec pipeline is the cause. Not a speed preset."
        ),
    ),
    Preset(
        name="ngram-only",
        summary="ngram-mod only (prose-heavy workloads)",
        spec=replace(
            BASE_SPEC,
            spec_type="ngram-mod",
            ngram_n_min=4,
            ngram_n_max=16,
            ngram_match=24,
        ),
        kv_type="q4_0",
        measured="108 t/s single-stream coding-text session (2026-09-02), ~= combo",
        measured_date="2026-09-02",
    ),
    Preset(
        name="mtp-only",
        summary="MTP3 only (native nextn head, up to 3 drafts/step)",
        spec=replace(BASE_SPEC, spec_type="mtp"),
        kv_type="q4_0",
        measured="52 t/s coding (39 t/s prose) single-stream session (2026-09-02)",
        measured_date="2026-09-02",
    ),
    Preset(
        name="kv-q8",
        summary="q8_0 KV cache instead of q4_0 (accuracy headroom, 2x KV VRAM)",
        spec=BASE_SPEC,
        kv_type="q8_0",
        measured=None,
        caveats=(
            "UNMEASURED. q4_0 KV is what production runs; q8_0 doubles attention-KV "
            "VRAM (16.0 -> 32.0 KiB/token pure KV): at 528k ctx the KV block alone "
            "goes 8.4 -> 16.8 GiB, so a 27B slot (21.3 GiB at q4_0, ~38 GiB at q8_0) "
            "no longer fits one V620 card."
        ),
    ),
    Preset(
        name="threads-28",
        summary="28 decode threads instead of 14 (one 56-core box / 2 slots per core group)",
        spec=BASE_SPEC,
        kv_type="q4_0",
        measured=None,
        caveats=(
            "UNMEASURED. Production uses 14 threads/slot (56 cores / 4 lanes). "
            "Worth an A/B only if the captain wants to re-partition CPU across lanes."
        ),
    ),
]


def preset(name: str) -> Preset:
    for p in PRESETS:
        if p.name == name:
            return p
    raise KeyError(f"unknown preset {name!r}; choose from {[p.name for p in PRESETS]}")


def apply_preset(cfg: SlotConfig, p: Preset) -> SlotConfig:
    """Return a new SlotConfig with the preset's spec block + KV type applied.
    ctx/rope/yarn/threads/model are untouched (preset scope is the spec+KV matrix)."""
    return replace(cfg, spec=p.spec, kv_type=p.kv_type)


def preset_cmd(cfg: SlotConfig, p: Preset, llama_bin: str = "llama-server") -> list[str]:
    """Full llama-server argv for the preset applied to cfg."""
    return build_llama_server_cmd(apply_preset(cfg, p), llama_bin=llama_bin)


def safe_default_preset() -> Preset:
    for p in PRESETS:
        if p.safe_default:
            return p
    raise AssertionError("PRESETS must contain exactly one safe default")


def summary_lines() -> list[str]:
    """Render the matrix as CLI-friendly text lines."""
    out: list[str] = []
    for p in PRESETS:
        tag = " [SAFE DEFAULT]" if p.safe_default else ""
        out.append(f"{p.name}{tag}: {p.summary}")
        if p.measured:
            out.append(f"  measured ({p.measured_date}): {p.measured}")
        else:
            out.append("  measured: (not measured)")
        if p.caveats:
            out.append(f"  caveats: {p.caveats}")
    return out
