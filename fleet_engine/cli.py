"""CLI entry point: python -m fleet_engine <command>

Commands:
  detect [--llama-bin PATH] [--roc-vendor PATH] [--slots-json PATH] [--no-probe]
      Scan for V620 GPUs, print a table, and (with --slots-json) merge the
      results into slots.json.

  start   --state-dir DIR [--slot N] [--llama-bin PATH] [--roc-vendor PATH] [--no-wait]
      Spawn one slot (or all slots in DIR/slots.json) under a supervisor.
  stop    --state-dir DIR [--slot N] [--llama-bin PATH] [--roc-vendor PATH]
      Stop one slot (or all).
  status  --state-dir DIR [--llama-bin PATH]
      Table of slots: pid, supervisor, port, /health.
  restart --state-dir DIR [--slot N] ...
      stop + start.

  runtime-supervisor ...   (internal: detached per-slot supervisor process)

The state dir holds slots.json, pids/, and logs/. Campaign rule: dev slots
use ports 45700+ and their own state dir; production lanes are untouched.
"""

from __future__ import annotations

import argparse
import sys
import time

from .config_store import ConfigStore
from .detector import detect_v620_slots
from .runtime import Runtime, ServerState, run_supervisor


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


def _runtime(args: argparse.Namespace) -> Runtime:
    return Runtime(
        state_dir=args.state_dir,
        llama_bin=getattr(args, "llama_bin", None) or "",
        roc_vendor=getattr(args, "roc_vendor", None),
    )


def _slot_ids(rt: Runtime, slot: int | None) -> list[int]:
    if slot is not None:
        return [slot]
    data = rt.store.load()
    return [int(k) for k in sorted(data.get("slots", {}), key=lambda x: int(x))]


def _print_state(states: list[ServerState]) -> None:
    print(f"{'slot':<6} {'running':<8} {'pid':<8} {'sup':<8} {'port':<7} {'health'}")
    for st in states:
        pid = str(st.pid) if st.pid else "-"
        sup = str(st.supervisor_pid) if st.supervisor_pid else "-"
        health = st.health if st.health is not None else "-"
        print(f"{st.name:<6} {str(st.running).lower():<8} {pid:<8} {sup:<8} {st.port:<7} {health}")


def cmd_start(args: argparse.Namespace) -> int:
    rt = _runtime(args)
    rc = 0
    for sid in _slot_ids(rt, args.slot):
        cfg = rt.store.get_slot(sid)
        t0 = time.time()
        try:
            pid = rt.start(cfg, wait_ready=not args.no_wait, log_fn=print)
            print(f"started slot {cfg.slot} {cfg.name} pid {pid} in {time.time() - t0:.0f}s")
        except (RuntimeError, FileNotFoundError, PermissionError, TimeoutError) as e:
            print(f"FAILED slot {cfg.slot}: {e}", file=sys.stderr)
            rc = 1
    return rc


def cmd_stop(args: argparse.Namespace) -> int:
    rt = _runtime(args)
    rc = 0
    for sid in _slot_ids(rt, args.slot):
        cfg = rt.store.get_slot(sid)
        try:
            rt.stop(cfg, log_fn=print)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"stop failed slot {cfg.slot}: {e}", file=sys.stderr)
            rc = 1
    return rc


def cmd_status(args: argparse.Namespace) -> int:
    rt = _runtime(args)
    states = [rt.status(rt.store.get_slot(sid)) for sid in _slot_ids(rt, args.slot)]
    _print_state(states)
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    rt = _runtime(args)
    rc = 0
    for sid in _slot_ids(rt, args.slot):
        cfg = rt.store.get_slot(sid)
        rt.stop(cfg, log_fn=print)
        try:
            pid = rt.start(cfg, wait_ready=not args.no_wait, log_fn=print)
            print(f"restarted slot {cfg.slot} {cfg.name} pid {pid}")
        except (RuntimeError, FileNotFoundError, PermissionError, TimeoutError) as e:
            print(f"FAILED restart slot {cfg.slot}: {e}", file=sys.stderr)
            rc = 1
    return rc


def cmd_supervisor(args: argparse.Namespace) -> int:
    return run_supervisor(
        state_dir=args.state_dir,
        slot=args.slot,
        llama_bin=args.llama_bin,
        roc_vendor=args.roc_vendor,
    )


def _add_runtime_opts(p: argparse.ArgumentParser, need_llama: bool) -> None:
    p.add_argument("--state-dir", required=True, help="runtime state dir (slots.json, pids/, logs/)")
    p.add_argument("--slot", type=int, default=None, help="slot number (default: all in slots.json)")
    if need_llama:
        p.add_argument("--llama-bin", default=None, help="path to llama-server")
        p.add_argument("--roc-vendor", default=None, help="ROCm vendor lib dir")
    p.add_argument("--no-wait", action="store_true", help="do not wait for /health readiness")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fleet_engine", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="scan for V620 GPUs")
    d.add_argument("--llama-bin", default=None, help="path to llama-server (for HIP probe)")
    d.add_argument("--roc-vendor", default=None, help="ROCm vendor lib dir for the probe")
    d.add_argument("--slots-json", default=None, help="merge results into this slots.json")
    d.add_argument("--no-probe", action="store_true", help="sysfs scan only, no engine probe")
    d.set_defaults(func=cmd_detect)

    for name, func, help_ in (
        ("start", cmd_start, "spawn slot(s) under a supervisor"),
        ("stop", cmd_stop, "stop slot(s)"),
        ("restart", cmd_restart, "restart slot(s)"),
    ):
        sp = sub.add_parser(name, help=help_)
        _add_runtime_opts(sp, need_llama=True)
        sp.set_defaults(func=func)

    st = sub.add_parser("status", help="status table")
    _add_runtime_opts(st, need_llama=False)
    st.set_defaults(func=cmd_status)

    spv = sub.add_parser("runtime-supervisor", help=argparse.SUPPRESS)
    spv.add_argument("--state-dir", required=True)
    spv.add_argument("--slot", type=int, required=True)
    spv.add_argument("--llama-bin", required=True)
    spv.add_argument("--roc-vendor", default=None)
    spv.set_defaults(func=cmd_supervisor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
