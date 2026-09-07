"""Tuner preset matrix: measured-safe defaults, argv parity, unknown preset."""

from __future__ import annotations

from fleet_engine.config_store import SlotConfig, SpecConfig
from fleet_engine.tuner import (
    BASE_SPEC,
    PRESETS,
    apply_preset,
    preset,
    preset_cmd,
    safe_default_preset,
    summary_lines,
)


def _slot() -> SlotConfig:
    return SlotConfig(
        slot=2, name="dev2", gpu=2, port=45800,
        model="/m/qwen.gguf", ctx=528384, rope_scale=2.0157, yarn_orig_ctx=262144,
        spec=SpecConfig(),
    )


def test_safe_default_is_prod():
    sd = safe_default_preset()
    assert sd.name == "prod-default"
    assert sd.spec == BASE_SPEC
    assert sd.kv_type == "q4_0"
    assert sd.measured_date == "2026-09-07"
    assert [p for p in PRESETS if p.safe_default] == [sd]


def test_prod_preset_argv_matches_production_flags():
    s = " ".join(preset_cmd(_slot(), preset("prod-default"), llama_bin="/ll/llama-server"))
    for token in (
        "--spec-type draft-mtp,ngram-mod",
        "--spec-draft-n-max 3",
        "--spec-draft-n-min 1",
        "--spec-ngram-mod-n-min 4",
        "--spec-ngram-mod-n-max 16",
        "--spec-ngram-mod-n-match 24",
        "--threads-draft 8",
        "--cache-type-k q4_0",
        "--cache-type-v q4_0",
    ):
        assert token in s, f"missing {token!r}"


def test_spec_off_preset_strips_spec_block():
    cfg = apply_preset(_slot(), preset("spec-off"))
    assert cfg.spec.spec_type == "none"
    s = " ".join(preset_cmd(cfg, preset("spec-off")))
    assert "--spec-type" not in s
    assert "--spec-draft-n-max" not in s


def test_single_drafter_presets():
    ngram = preset("ngram-only")
    assert ngram.spec.spec_type == "ngram-mod"
    mtp = preset("mtp-only")
    assert mtp.spec.spec_type == "mtp"
    s = " ".join(preset_cmd(_slot(), mtp))
    assert "--spec-ngram-mod-n-min" not in s


def test_kv_q8_changes_cache_types_only():
    kv = preset("kv-q8")
    cfg = apply_preset(_slot(), kv)
    assert cfg.kv_type == "q8_0"
    assert cfg.spec == BASE_SPEC  # spec block untouched by a KV preset
    s = " ".join(preset_cmd(cfg, kv))
    assert "--cache-type-k q8_0" in s and "--cache-type-v q8_0" in s
    assert "--spec-type draft-mtp,ngram-mod" in s


def test_unknown_preset_raises():
    try:
        preset("bogus")
    except KeyError:
        return
    raise AssertionError("unknown preset should raise KeyError")


def test_every_preset_has_measurement_or_caveat():
    for p in PRESETS:
        assert p.measured or p.caveats, f"{p.name} has neither measured nor caveats"
        if p.measured_date:
            assert p.measured, f"{p.name} has a date but no measurement"


def test_summary_lines_cover_all_presets():
    lines = "\n".join(summary_lines())
    for p in PRESETS:
        assert p.name in lines
    assert "[SAFE DEFAULT]" in lines
