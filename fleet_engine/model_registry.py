"""Model registry: scan local GGUF folders, parse quant/size/ctx metadata.

Phase 4 of PLAN I. Replaces the hand-picked MODEL var from config.env with a
folder scan. Metadata comes from a minimal, bounded GGUF header reader (we
never allocate the huge tokenizer arrays - we skip them).
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass
from typing import Optional

GGUF_MAGIC = b"GGUF"

# scalar value type -> struct format (per ggml/include/gguf.h GGUF_TYPE_*)
# 0 UINT8, 1 INT8, 2 UINT16, 3 INT16, 4 UINT32, 5 INT32, 6 FLOAT32,
# 7 BOOL (u32 on disk), 8 STRING, 9 ARRAY (element type nested), 10 UINT64, 11 INT64, 12 FLOAT64
_SCALAR_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
    10: "<Q", 11: "<q", 12: "<d",
}
# element byte size (arrays skip count * elem_size)
_ELEM_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8}


class GgufError(Exception):
    pass


@dataclass
class ModelInfo:
    path: str
    name: str
    size_bytes: int
    quant: str = "unknown"
    arch: str = "unknown"
    native_ctx: int = 0
    embedding_length: int = 0
    file_type: int = 0  # ggml file_type u32
    metadata: dict = None  # scalar kv pairs (arrays excluded)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def _quant_from_file_type(file_type: int) -> str:
    """Approximate quant label from ggml_ftype (general.file_type).

    NOTE: ggml_ftype is the 'mostly_*' enum from ggml.h (NOT tensor types).
    It caps at 28 in the pinned build; newer 'UD' (unsloth dynamic) codes like
    30 have no canonical label - callers must fall back to the filename.
    """
    table = {
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_1_SOME_F16",
        7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K",
        12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "IQ2_XXS", 16: "IQ2_XS",
        17: "IQ3_XXS", 18: "IQ1_S", 19: "IQ4_NL", 20: "IQ3_S", 21: "IQ2_S",
        22: "IQ4_XS", 23: "IQ1_M", 24: "BF16", 25: "MXFP4", 26: "NVFP4",
        27: "Q1_0", 28: "Q2_0",
    }
    return table.get(file_type, f"type-{file_type}")


def _quant_from_filename(stem: str) -> str:
    """Pull a quant token out of the filename, e.g. '...-UD-IQ4_XS' -> IQ4_XS,
    '...-Q8_0' -> Q8_0, 'mmproj-F16' -> F16. Matches the trailing uppercase
    quant label (letters/digits around an underscore, or a bare F16/BF16)."""
    m = re.search(r"([A-Z][A-Z0-9]{0,2}_[A-Z0-9]{1,3}(?:_[A-Z0-9]{1,3})?)$", stem)
    if m:
        return m.group(1)
    m = re.search(r"([A-Z]{1,2}16)$", stem)  # F16 / BF16 mmproj style
    return m.group(1) if m else "unknown"


def read_gguf_metadata(path: str, max_kv: int = 64) -> dict:
    """Read the GGUF key/value header, skipping (not allocating) array values.

    Returns a dict of scalar values. Raises GgufError on a bad header.
    Header layout (little-endian): magic(4) version(4) n_kv n_tensors
    [kv pairs: keylen(u64) key type(u8) payload] [tensor info].
    Counts are u32 in v1/v2, u64 in v3.
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise GgufError(f"not a GGUF file (magic={magic!r}): {path}")
        version = struct.unpack("<I", f.read(4))[0]
        if version == 0 or version > 3:
            raise GgufError(f"unsupported GGUF version {version}: {path}")
        # v1/v2 used u32 counts; v3 (all real models) uses u64
        cnt_fmt = "<Q" if version >= 3 else "<I"
        n_kv = struct.unpack(cnt_fmt, f.read(8 if version >= 3 else 4))[0]
        n_tensors = struct.unpack(cnt_fmt, f.read(8 if version >= 3 else 4))[0]
        if n_kv > 100000:
            raise GgufError(f"implausible n_kv={n_kv}: {path}")
        out: dict = {"__n_tensors__": n_tensors, "__version__": version}
        for _ in range(min(n_kv, max_kv)):
            klen = struct.unpack("<Q", f.read(8))[0]
            if klen > 1024:
                break  # mis-parse guard
            key = f.read(klen).decode("utf-8", "replace")
            vtype = struct.unpack("<I", f.read(4))[0]
            if vtype in _SCALAR_FMT:
                fmt = _SCALAR_FMT[vtype]
                out[key] = struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
            elif vtype == 8:  # string
                slen = struct.unpack("<Q", f.read(8))[0]
                if slen > 1 << 20:
                    break
                out[key] = f.read(slen).decode("utf-8", "replace")
            elif vtype == 7:  # bool
                out[key] = bool(struct.unpack("<I", f.read(4))[0])
            elif vtype == 9:  # array: element type (u32) + count (u64), skip payload
                etype = struct.unpack("<I", f.read(4))[0]
                cnt = struct.unpack("<Q", f.read(8))[0]
                if cnt > (1 << 22):
                    break  # mis-parse guard
                if etype in _ELEM_SIZE:  # fixed-size elements: seek past them
                    f.seek(cnt * _ELEM_SIZE[etype], 1)
                elif etype == 8:  # string array: read lengths, skip payloads
                    for _ in range(cnt):
                        slen = struct.unpack("<Q", f.read(8))[0]
                        if slen > (1 << 24):
                            raise GgufError(f"implausible string len {slen} in {key!r}")
                        f.seek(slen, 1)
                else:
                    raise GgufError(f"unknown GGUF array element type {etype} for {key!r}")
                out.setdefault(f"{key}__array_len", cnt)
            else:
                raise GgufError(f"unknown GGUF value type {vtype} for {key!r}")
        # stop reading kv (we only need the header, not tensor blobs)
    return out


def scan_models(dirs: list[str], max_files: int = 500) -> list[ModelInfo]:
    """Scan directories (recursively) for .gguf files and read their metadata."""
    found: list[ModelInfo] = []
    seen: set[str] = set()
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, _subdirs, files in os.walk(d):
            for fn in files:
                if not fn.lower().endswith(".gguf"):
                    continue
                full = os.path.join(root, fn)
                real = os.path.realpath(full)
                if real in seen:
                    continue
                seen.add(real)
                info = _info_for(full)
                found.append(info)
                if len(found) >= max_files:
                    return found
    return found


def _info_for(path: str) -> ModelInfo:
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    try:
        md = read_gguf_metadata(path)
    except (GgufError, OSError, struct.error):
        return ModelInfo(path=path, name=stem, size_bytes=size)
    arch = str(md.get("general.architecture", "unknown"))
    ctx = int(md.get(f"{arch}.context_length", 0) or 0)
    embd = int(md.get(f"{arch}.embedding_length", 0) or 0)
    ftype = int(md.get("general.file_type", 0) or 0)
    # Filename wins: unsloth 'UD' dynamic quants use ftype codes (e.g. 30) that
    # have no canonical label in the pinned ggml_ftype enum.
    quant = _quant_from_filename(stem)
    if quant == "unknown":
        quant = _quant_from_file_type(ftype)
    return ModelInfo(
        path=path,
        name=stem,
        size_bytes=size,
        quant=quant,
        arch=str(arch),
        native_ctx=ctx,
        embedding_length=embd,
        file_type=ftype,
        metadata={k: v for k, v in md.items() if not k.startswith("__") and k.endswith("__array_len") is False},
    )


def find_model(dirs: list[str], needle: str) -> Optional[ModelInfo]:
    """Find a model by a substring of its name or path (case-insensitive)."""
    n = needle.lower()
    for info in scan_models(dirs):
        if n in info.name.lower() or n in info.path.lower():
            return info
    return None
