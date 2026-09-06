"""Round-trip + parity tests for config store and command builder."""

from __future__ import annotations

import json

from fleet_engine.config_store import (
    SCHEMA_VERSION,
    ConfigStore,
    SlotConfig,
    SpecConfig,
    slot_from_dict,
    slot_to_dict,
)
from fleet_engine.runtime_cmd import build_llama_server_cmd


def production_like_slot() -> SlotConfig:
    """Mirror of the verified production config.env values (Qwen3.8-27B)."""
    return SlotConfig(
        slot=1,
        name="gpu1",
        gpu=1,
        host="0.0.0.0",
        port=45600,
        model="/models/Qwen3.8-27B-UD-IQ4_XS.gguf",
        mmproj="/models/mmproj-F16.gguf",
        ctx=528384,
        concurrency=1,
        rope_scale=2.0157,
        yarn_orig_ctx=262144,
        kv_type="q4_0",
        kv_unified=True,
        num_gpu_layers="all",
        threads=14,
        spec=SpecConfig(
            spec_type="draft-mtp,ngram-mod",
            mtp_n_max=3,
            mtp_n_min=1,
            ngram_n_min=4,
            ngram_n_max=16,
            ngram_match=24,
            draft_threads=8,
        ),
    )


def test_round_trip(tmp_path):
    store = ConfigStore(str(tmp_path / "slots.json"))
    cfg = production_like_slot()
    store.set_slot(cfg)
    loaded = store.get_slot(1)
    assert json.loads(json.dumps(slot_to_dict(loaded))) == json.loads(json.dumps(slot_to_dict(cfg)))
    assert store.load()["schema_version"] == SCHEMA_VERSION


def test_build_cmd_matches_production_flags():
    cmd = build_llama_server_cmd(production_like_slot(), llama_bin="/opt/llama/llama-server")
    s = " ".join(cmd)
    assert cmd[0] == "/opt/llama/llama-server"
    # core serving flags
    for token in (
        "--host 0.0.0.0",
        "--port 45600",
        "--n-gpu-layers all",
        "--threads 14",
        "--ctx-size 528384",
        "--rope-scaling yarn",
        "--rope-scale 2.0157",
        "--yarn-orig-ctx 262144",
        "--cache-type-k q4_0",
        "--cache-type-v q4_0",
        "--mmproj /models/mmproj-F16.gguf",
        "--parallel 1",
        "--kv-unified",
    ):
        assert token in s, f"missing {token!r} in: {s}"
    # spec block (combined MTP3 + ngram-mod, verified at 32.7 tok/s)
    for token in (
        "--spec-type draft-mtp,ngram-mod",
        "--spec-draft-n-max 3",
        "--spec-draft-n-min 1",
        "--spec-ngram-mod-n-min 4",
        "--spec-ngram-mod-n-max 16",
        "--spec-ngram-mod-n-match 24",
        "--threads-draft 8",
    ):
        assert token in s, f"missing {token!r} in: {s}"


def test_spec_off_omits_spec_block():
    cfg = production_like_slot()
    cfg.spec.spec_type = "none"
    s = " ".join(build_llama_server_cmd(cfg))
    assert "--spec-type" not in s
    assert "--spec-draft-n-max" not in s


def test_invalid_slot_rejected():
    try:
        SlotConfig(slot=17, name="x", gpu=1)
    except ValueError:
        return
    raise AssertionError("slot 17 should raise")


def test_invalid_spec_type_rejected():
    try:
        slot_from_dict({"slot": 1, "name": "x", "gpu": 1, "spec": {"spec_type": "bogus"}})
    except ValueError:
        return
    raise AssertionError("bogus spec_type should raise")
