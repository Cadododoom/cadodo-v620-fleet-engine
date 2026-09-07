"""Benchmark report generation: JSON + Markdown from slot results.

Pure functions (no I/O except writing the output files), so the report
layout is unit-testable. Spec-acceptance stats are pulled from the
slot's llama-server log file when a path is supplied (metrics.parse).
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from .client import SlotBenchResult, result_to_dict
from .prompts import PROMPT_SET_VERSION


def spec_stats_from_log(log_path: str) -> dict:
    """Latest + aggregate draft-acceptance stats from a llama-server log.

    Returns {} when the log is missing or has no spec lines.
    """
    from fleet_engine.metrics import RE_SPEC

    try:
        text = open(log_path, "r", encoding="utf-8", errors="replace").read()
    except OSError:
        return {}
    accs, accepted, generated, mlens = [], 0, 0, []
    for line in text.splitlines():
        m = RE_SPEC.search(line)
        if not m:
            continue
        accs.append(float(m.group("acc")))
        accepted += int(m.group("accepted"))
        generated += int(m.group("generated"))
        if m.group("mlen"):
            mlens.append(float(m.group("mlen")))
    if not accs:
        return {}
    return {
        "latest": accs[-1],
        "mean": sum(accs) / len(accs),
        "min": min(accs),
        "max": max(accs),
        "accepted_total": accepted,
        "generated_total": generated,
        "mean_len": sum(mlens) / len(mlens) if mlens else None,
        "samples": len(accs),
    }


def build_report(
    slot_results: list[SlotBenchResult],
    slot_log_paths: Optional[dict[int, str]] = None,
    meta: Optional[dict] = None,
) -> dict:
    """Aggregate per-slot results into one report dict (JSON-serializable)."""
    slot_log_paths = slot_log_paths or {}
    slots = []
    for res in slot_results:
        entry = result_to_dict(res)
        entry["ok_count"] = res.ok_count
        entry["failed_count"] = res.failed_count
        entry["decode_tps_mean"] = res.decode_tps_mean()
        entry["prefill_tps_mean"] = res.prefill_tps_mean()
        logp = slot_log_paths.get(res.port)
        entry["spec"] = spec_stats_from_log(logp) if logp else {}
        slots.append(entry)
    return {
        "report_version": "1.0",
        "prompt_set": PROMPT_SET_VERSION,
        "generated_ts": meta.get("generated_ts", time.time()) if meta else time.time(),
        "meta": meta or {},
        "slots": slots,
    }


def _fmt(v, spec: str = "") -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return format(v, spec) + " " if not spec else format(v, spec)
    return str(v)


def render_markdown(report: dict) -> str:
    """Render the report dict to Markdown (the BENCHMARKS.md body)."""
    L: list[str] = []
    meta = report.get("meta", {})
    L.append("# V620 Fleet Benchmark Report")
    L.append("")
    L.append(f"- prompt set: **{report.get('prompt_set')}**")
    if meta:
        for k in ("host", "model", "engine", "notes"):
            if meta.get(k):
                L.append(f"- {k}: {meta[k]}")
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append("| slot | endpoint | c | ok/total | decode t/s (mean) | prefill t/s (mean) | spec acc (mean) | spec mean len |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in report["slots"]:
        spec = s.get("spec") or {}
        L.append(
            f"| {s['port']} | {s['endpoint']} | {s['concurrency']} "
            f"| {s['ok_count']}/{len(s['results'])} "
            f"| {_fmt(s.get('decode_tps_mean'), '.2f')} "
            f"| {_fmt(s.get('prefill_tps_mean'), '.1f')} "
            f"| {_fmt(spec.get('mean'), '.3f')} "
            f"| {_fmt(spec.get('mean_len'), '.2f')} |"
        )
    L.append("")
    L.append("Per-slot detail")
    for s in report["slots"]:
        L.append("")
        L.append(f"### slot {s['port']} (c={s['concurrency']})")
        L.append("")
        L.append("| prompt | p_tok | gen | total ms | rate t/s | cached |")
        L.append("|---|---|---|---|---|---|")
        for r in s["results"]:
            L.append(
                f"| {r['prompt']} | {r['prompt_tokens']} | {r['completion_tokens']} "
                f"| {r['total_ms']:.0f} | {r['total_tps']:.2f} | {r['cached_tokens']} |"
            )
    L.append("")
    L.append("> decode t/s in per-run rows is the client-observed total rate (non-streaming);")
    L.append("> the spec column and server-side rates come from the slot log (draft acceptance).")
    L.append("")
    return "\n".join(L)


def write_report(report: dict, out_dir: str, stem: str = "bench") -> tuple[str, str]:
    """Write <stem>.json + <stem>.md to out_dir. Returns (json_path, md_path)."""
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, f"{stem}.json")
    mpath = os.path.join(out_dir, f"{stem}.md")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))
    return jpath, mpath
