"""Regression tests for the GGUF header reader + model registry.

The header parser was fixed against the real local corpus (unsloth
Qwen3.8-27B GGUFs). The synthetic header below encodes every value-type
branch of the parser (u8/i16/u32/i64/f32/f64/bool/string/array-of-string/
array-of-f32) so a future type-table regression fails here, not at scan time.
"""

from __future__ import annotations

import struct

from fleet_engine.model_registry import (
    GgufError,
    ModelInfo,
    _quant_from_filename,
    read_gguf_metadata,
    scan_models,
)


def _kv(key: str, vtype: int, payload: bytes) -> bytes:
    # header format: keylen(u64) key type(u32) payload
    return struct.pack("<Q", len(key)) + key.encode("utf-8") + struct.pack("<I", vtype) + payload


def build_gguf_v3(kv_pairs: list[bytes], n_tensors: int = 0) -> bytes:
    body = b"".join(kv_pairs)
    return (
        b"GGUF"
        + struct.pack("<I", 3)  # version
        + struct.pack("<Q", len(kv_pairs))
        + struct.pack("<Q", n_tensors)
        + body
    )


def make_header_bytes() -> bytes:
    pairs = [
        _kv("general.architecture", 8, struct.pack("<Q", 6) + b"qwen35"),
        _kv("a.u8", 0, struct.pack("<B", 7)),
        _kv("a.i16", 3, struct.pack("<h", -1234)),
        _kv("a.u32", 4, struct.pack("<I", 40)),
        _kv("a.i64", 11, struct.pack("<q", -9876543210)),
        _kv("a.f32", 6, struct.pack("<f", 0.9)),
        _kv("a.f64", 12, struct.pack("<d", 2.0157)),
        _kv("a.bool", 7, struct.pack("<I", 1)),
        # string array of 2 tokens (tokenizer blob shape in real models)
        _kv(
            "a.tok_arr",
            9,
            struct.pack("<I", 8)  # element type = STRING
            + struct.pack("<Q", 2)  # 2 elements
            + struct.pack("<Q", 2) + b"hi"
            + struct.pack("<Q", 5) + b"llama",
        ),
        # f32 array of 3
        _kv("a.f32_arr", 9, struct.pack("<I", 6) + struct.pack("<Q", 3) + struct.pack("<3f", 1.0, 2.0, 3.0)),
    ]
    return build_gguf_v3(pairs)


def test_header_types_roundtrip(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(make_header_bytes())
    md = read_gguf_metadata(str(p))
    assert md["general.architecture"] == "qwen35"
    assert md["a.u8"] == 7
    assert md["a.i16"] == -1234
    assert md["a.u32"] == 40
    assert md["a.i64"] == -9876543210
    assert abs(md["a.f32"] - 0.9) < 1e-6
    assert abs(md["a.f64"] - 2.0157) < 1e-9
    assert md["a.bool"] is True
    assert md["a.tok_arr__array_len"] == 2
    assert md["a.f32_arr__array_len"] == 3


def test_bad_magic_rejected(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"XXXX" + b"\0" * 64)
    try:
        read_gguf_metadata(str(p))
    except GgufError:
        return
    raise AssertionError("bad magic should raise GgufError")


def test_quant_from_filename():
    assert _quant_from_filename("Qwen3.8-27B-UD-IQ4_XS") == "IQ4_XS"
    assert _quant_from_filename("Model-Q8_0") == "Q8_0"
    assert _quant_from_filename("mmproj-F16") == "F16"


def test_scan_picks_up_metadata(tmp_path):
    # A real model's header shape: arch + per-arch ctx/embd + ftype
    pairs = [
        _kv("general.architecture", 8, struct.pack("<Q", 6) + b"qwen35"),
        _kv("general.file_type", 4, struct.pack("<I", 22)),  # GGML_FTYPE_MOSTLY_IQ4_XS
        _kv("qwen35.context_length", 4, struct.pack("<I", 262144)),
        _kv("qwen35.embedding_length", 4, struct.pack("<I", 5120)),
    ]
    p = tmp_path / "MyModel-IQ4_XS.gguf"
    p.write_bytes(build_gguf_v3(pairs))
    infos = scan_models([str(tmp_path)])
    assert len(infos) == 1
    i = infos[0]
    assert isinstance(i, ModelInfo)
    assert i.arch == "qwen35"
    assert i.native_ctx == 262144
    assert i.embedding_length == 5120
    assert i.quant == "IQ4_XS"  # filename wins over ftype table
    assert "Qwen35" not in i.metadata  # no stray keys
