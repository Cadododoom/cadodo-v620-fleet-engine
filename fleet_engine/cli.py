"""CLI entry point: python -m fleet_engine <command>

Commands (Phase 2):
  detect [--llama-bin PATH] [--roc-vendor PATH] [--slots-json PATH] [--no-probe]
      Scan for V620 GPUs, print a table, and (with --slots-json) merge the
      results into slots.json.

Later phases add: start/stop/status/restart (runtime), models (registry),
bench, connect (connector), link (harness).
"""

from __future__ import annotations

import argparse
import sys

from .config_store import ConfigStore
from .detector import detect_v620_slots


def cmd_detect(args: argparse.Namespace) -> int:
    detection = detect_v620_slots(
        llama_bin=None if args.no_probe else args.llama_bin,
        roc_vendor=args.roc_vendor,
    )
    print(f"{'slot':<5} {'name':<10} {'hip':<5} {'pci':<14} {'vram':>9} {'used':>9}")
    for s in detection.slots:
        hip = str(s.hip_index) if s.hip_index is not None else "?"
        total_gib = s.vram_total_bytes / (1024**3)
        used_gib = s.vram_used_bytes / (1024**3)
        print(f"{s.slot:<5} {s.name:<10} {hip:<5} {s.pci_addr:<14} {total_gib:>8.1f}G {used_gib:>8.1f}G")
    if not detection.probe_ok:
        print(f"probe: {detection.probe_error}", file=sys.stderr)
    if args.slots_json:
        store = ConfigStore(args.slots_json)
        data = store.merge_detection(detection)
        store.save(data)
        print(f"merged {len(detection.slots)} slots into {args.slots_json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fleet_engine", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="scan for V620 GPUs")
    d.add_argument("--llama-bin", default=None, help="path to llama-server (for HIP probe)")
    d.add_argument("--roc-vendor", default=None, help="ROCm vendor lib dir for the probe")
    d.add_argument("--slots-json", default=None, help="merge results into this slots.json")
    d.add_argument("--no-probe", action="store_true", help="sysfs scan only, no engine probe")
    d.set_defaults(func=cmd_detect)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
