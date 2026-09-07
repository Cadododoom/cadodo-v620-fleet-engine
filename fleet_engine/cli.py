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
import json
import os
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


def cmd_models(args: argparse.Namespace) -> int:
    from .model_registry import scan_models

    infos = scan_models(args.dirs, max_files=args.max_files)
    if not infos:
        print("no .gguf files found", file=sys.stderr)
        return 1
    print(f"{'name':<46} {'quant':<11} {'arch':<12} {'ctx':>7} {'embd':>6} {'size':>8}")
    for m in sorted(infos, key=lambda x: x.name.lower()):
        print(
            f"{m.name:<46} {m.quant:<11} {m.arch:<12} {m.native_ctx:>7} "
            f"{m.embedding_length:>6} {m.size_bytes / 1e9:>7.2f}G"
        )
        if not args.compact:
            print(f"  {m.path}")
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


def cmd_ui(args: argparse.Namespace) -> int:
    """Launch the control UI (tkinter). --screenshot dumps one rendered
    frame to a file and exits (headless verification)."""
    from .app import Panel
    from .runtime import Runtime
    from .ui import FleetModel

    try:
        import tkinter as tk
    except ImportError:
        print("tkinter not available on this platform", file=sys.stderr)
        return 1

    model = FleetModel()
    runtime = None
    if args.state_dir:
        dev = FleetModel.for_dev(args.state_dir)
        model.dev_slots = dev.dev_slots
        model.dev_state_dir = args.state_dir
        runtime = Runtime(state_dir=args.state_dir, llama_bin=args.llama_bin or "")
    if args.fleet_dir:
        model.prod_lanes = FleetModel.for_prod(args.fleet_dir).prod_lanes
    model.refresh()
    model.update_logs()
    model.update_health()

    root = tk.Tk()
    Panel(root, model, runtime, state_dir=args.state_dir or "",
          hermes_config=getattr(args, "hermes_config", None) or "")
    if args.screenshot:
        root.update_idletasks()
        root.update()
        desc = _grab_screenshot(root, args.screenshot)
        print(f"screenshot: {args.screenshot} ({desc})")
        root.destroy()
    else:
        root.mainloop()
    return 0


def _grab_screenshot(root, path: str, settle_s: float = 1.0) -> str:
    """Capture the live X window to `path` (PNG) via PIL.ImageGrab (X11
    backend) and crop to the window's geometry. Returns a short description
    (size/mode). Robust across X servers/visuals — no hand-rolled XWD binary
    parsing (xwd emits v0/v1 depending on the build; ImageGrab handles both).

    Before grabbing we pump the Tk event loop for `settle_s` seconds in small
    steps: a bare sleep() does NOT process X events, so the window's Expose
    would go unhandled and Xvfb returns a blank framebuffer on the first grab.
    """
    import os
    import time

    from PIL import ImageGrab

    # Pump the event loop so the freshly-mapped window actually paints
    # (Expose handled) before we read the framebuffer.
    steps = max(1, int(settle_s * 20))
    for _ in range(steps):
        root.update_idletasks()
        root.update()
        time.sleep(settle_s / steps)
    root.update_idletasks()
    root.update()
    display = os.environ.get("DISPLAY", ":0")
    screen = ImageGrab.grab(xdisplay=display)
    x, y = root.winfo_rootx(), root.winfo_rooty()
    w, h = root.winfo_width(), root.winfo_height()
    cropped = crop_to_window(screen, x, y, w, h)
    cropped.save(path)
    return f"{cropped.size[0]}x{cropped.size[1]} {cropped.mode}"


def crop_to_window(screen, x: int, y: int, w: int, h: int):
    """Crop `screen` (a PIL image) to the window at (x, y) of size (w, h),
    clamped to the screen bounds. Pure — unit-testable without a display."""
    x0 = max(0, min(x, screen.width - 1))
    y0 = max(0, min(y, screen.height - 1))
    x1 = min(screen.width, x0 + max(0, w))
    y1 = min(screen.height, y0 + max(0, h))
    if x1 <= x0 or y1 <= y0:
        return screen  # degenerate geometry: fall back to full screen
    return screen.crop((x0, y0, x1, y1))


def _conn_slots(args: argparse.Namespace) -> list:
    """Load SlotConfigs for --slot N (default: all) from --state-dir/slots.json."""
    from .config_store import ConfigStore, slot_from_dict
    store = ConfigStore(os.path.join(args.state_dir, "slots.json"))
    data = store.load()
    ids = sorted(int(k) for k in data.get("slots", {}))
    if args.slot is not None:
        ids = [i for i in ids if i == args.slot]
    return [slot_from_dict(data["slots"][str(i)], i) for i in ids]


def _conn_model_names(slots: list) -> dict[int, str]:
    """Model string per slot: the GGUF path from slots.json (prod convention)."""
    return {s.slot: s.model for s in slots}


def cmd_conn_preview(args: argparse.Namespace) -> int:
    from .connector import diff_preview
    slots = _conn_slots(args)
    if not slots:
        print("no slots selected", file=sys.stderr)
        return 1
    print(diff_preview(args.config, slots, _conn_model_names(slots), host=args.host))
    return 0


def cmd_conn_apply(args: argparse.Namespace) -> int:
    from .connector import apply as conn_apply
    slots = _conn_slots(args)
    if not slots:
        print("no slots selected", file=sys.stderr)
        return 1
    changed = conn_apply(
        args.config, slots, _conn_model_names(slots), host=args.host,
        dry_run=args.dry_run, backup=not args.no_backup,
    )
    mode = "DRY-RUN" if args.dry_run else "APPLIED"
    if changed:
        print(f"[{mode}] changed: {', '.join(changed)}")
    else:
        print(f"[{mode}] no changes needed")
    return 0


def cmd_conn_drift(args: argparse.Namespace) -> int:
    from .connector import check_drift
    slots = _conn_slots(args)
    drifts = []
    for s in slots:
        drifts.extend(check_drift(args.config, s, s.model, host=args.host))
    if drifts:
        for d in drifts:
            print(f"DRIFT {d}")
        return 2
    print(f"no drift across {len(slots)} slot(s)")
    return 0


def cmd_conn_verify(args: argparse.Namespace) -> int:
    from .connector import verify_reconnect
    slots = _conn_slots(args)
    rc = 0
    for s in slots:
        ok, detail = verify_reconnect(s, host=args.host)
        print(f"[{'ok' if ok else 'DOWN'}] {s.name} :{s.port} — {detail}")
        if not ok:
            rc = 1
    return rc


def cmd_opc_preview(args: argparse.Namespace) -> int:
    from .opencode import provider_entry, provider_key
    slots = _conn_slots(args)
    if not slots:
        print("no slots selected", file=sys.stderr)
        return 1
    for s in slots:
        print(f"--- {provider_key(s)} ---")
        print(json.dumps(provider_entry(s, host=args.host), indent=2))
    return 0


def cmd_opc_apply(args: argparse.Namespace) -> int:
    from .opencode import apply as opc_apply
    slots = _conn_slots(args)
    if not slots:
        print("no slots selected", file=sys.stderr)
        return 1
    changed = opc_apply(args.config, slots, host=args.host, dry_run=args.dry_run,
                        backup=not args.no_backup, make_default=args.make_default)
    mode = "DRY-RUN" if args.dry_run else "APPLIED"
    print(f"[{mode}] changed: {', '.join(changed) if changed else 'none'}")
    return 0


def cmd_opc_drift(args: argparse.Namespace) -> int:
    from .opencode import check_drift as opc_drift
    slots = _conn_slots(args)
    drifts = []
    for s in slots:
        drifts.extend(opc_drift(args.config, s, host=args.host))
    if drifts:
        for d in drifts:
            print(f"DRIFT {d}")
        return 2
    print(f"no drift across {len(slots)} slot(s)")
    return 0


def cmd_opc_turn(args: argparse.Namespace) -> int:
    from .opencode import chat_probe, model_id, opencode_turn, provider_key
    slots = _conn_slots(args)
    rc = 0
    for s in slots:
        ok, detail = chat_probe(s, host=args.host)
        print(f"[{'ok' if ok else 'FAIL'}] {s.name} :{s.port} — {detail[:160]!r}")
        if not ok:
            rc = 1
        if ok and args.opencode_bin:
            ref = f"{provider_key(s)}/{model_id(s)}"
            ok2, out = opencode_turn(args.opencode_bin, ref, args.prompt)
            print(f"[{'ok' if ok2 else 'FAIL'}] opencode turn {ref} — {out[:160]!r}")
            if not ok2:
                rc = 1
    return rc


def cmd_bench(args: argparse.Namespace) -> int:
    """Run the standardized benchmark suite against one or more endpoints.

    --endpoint takes a comma list of base URLs or ports (bare ports mean
    127.0.0.1). --all-dev / --all-prod expand to the dev (45798-45801) or
    prod (45600-45603) endpoint sets on --host.
    """
    import sys

    from bench.client import run_suite
    from bench.report import build_report, write_report

    host = args.host
    endpoints = []
    if args.endpoint:
        for tok in args.endpoint.split(","):
            tok = tok.strip()
            if not tok:
                continue
            endpoints.append(tok if tok.startswith("http") else f"http://{host}:{tok}/v1")
    elif args.all_dev:
        endpoints = [f"http://{host}:{p}/v1" for p in (45798, 45799, 45800, 45801)]
    elif args.all_prod:
        endpoints = [f"http://{host}:{p}/v1" for p in (45600, 45601, 45602, 45603)]
    else:
        print("no endpoints given (use --endpoint, --all-dev, or --all-prod)", file=sys.stderr)
        return 1
    prompts = [x.strip() for x in args.prompts.split(",") if x.strip()] or None
    results = []
    rc = 0
    for ep in endpoints:
        try:
            import json as _json
            from urllib.request import urlopen as _uop
            with _uop(ep.rstrip("/") + "/models", timeout=5) as r:
                names = [m.get("name", m.get("id", "")) for m in _json.load(r).get("models", [])]
            model = args.model or (names[0] if names else "unknown")
        except Exception:
            print(f"[FAIL] {ep} unreachable for /models", file=sys.stderr)
            rc = 1
            continue
        res = run_suite(ep, model, prompts=prompts, concurrency=args.concurrency,
                        timeout=args.timeout, log=print)
        results.append(res)
        if res.ok_count == 0:
            rc = 1
    if results:
        log_paths = {
            r.port: lp
            for r, lp in zip(results, [x for x in args.logs.split(",") if x], strict=False)
            if lp
        }
        report = build_report(
            results,
            slot_log_paths=log_paths,
            meta={
                "host": host,
                "model": args.model or "(per-endpoint)",
                "concurrency": args.concurrency,
                "prompt_set": prompts or "default(3)",
            },
        )
        jpath, mpath = write_report(report, args.out_dir, stem=args.stem)
        print(f"report: {jpath}")
        print(f"        {mpath}")
    return rc


def _add_conn_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--state-dir", required=True, help="runtime state dir with slots.json")
    p.add_argument("--config", required=True, help="Hermes config.yaml path (managed blocks live here)")
    p.add_argument("--slot", type=int, default=None, help="slot number (default: all)")
    p.add_argument("--host", default="127.0.0.1", help="host used in base_url / probes")


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

    m = sub.add_parser("models", help="scan GGUF folders, print metadata table")
    m.add_argument("dirs", nargs="+", help="GGUF folder(s) to scan")
    m.add_argument("--max-files", type=int, default=500)
    m.add_argument("--compact", action="store_true", help="hide full paths")
    m.set_defaults(func=cmd_models)

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

    u = sub.add_parser("ui", help="launch the control UI (16-slot grid)")
    u.add_argument("--state-dir", default=None, help="dev state dir with slots.json")
    u.add_argument("--fleet-dir", default=None,
                   help="production fleet dir (read-only prod view)")
    u.add_argument("--llama-bin", default=None, help="path to llama-server")
    u.add_argument("--hermes-config", default=None,
                   help="Hermes config.yaml to sync on slot start/restart (dev slots only)")
    u.add_argument("--screenshot", default=None,
                   help="capture one frame to this PNG path and exit")
    u.set_defaults(func=cmd_ui)

    for name, func, help_ in (
        ("conn-preview", cmd_conn_preview, "preview managed provider blocks (no writes)"),
        ("conn-drift", cmd_conn_drift, "read-only drift check: config vs slots.json"),
        ("conn-verify", cmd_conn_verify, "reconnect probe: /v1/models per slot"),
    ):
        cp = sub.add_parser(name, help=help_)
        _add_conn_opts(cp)
        cp.set_defaults(func=func)

    ca = sub.add_parser("conn-apply", help="write managed provider blocks into config")
    _add_conn_opts(ca)
    ca.add_argument("--dry-run", action="store_true", help="report changes without writing")
    ca.add_argument("--no-backup", action="store_true", help="skip the .bak timestamped copy")
    ca.set_defaults(func=cmd_conn_apply)

    for name, func, help_ in (
        ("opc-preview", cmd_opc_preview, "preview OpenCode provider entries (no writes)"),
        ("opc-drift", cmd_opc_drift, "read-only OpenCode drift check"),
        ("opc-turn", cmd_opc_turn, "chat probe via slot (optional real opencode turn)"),
    ):
        cp = sub.add_parser(name, help=help_)
        _add_conn_opts(cp)
        if name == "opc-turn":
            cp.add_argument("--opencode-bin", default=None,
                            help="path to opencode binary for a real end-to-end turn")
            cp.add_argument("--prompt", default="Reply with the single word: ok")
        cp.set_defaults(func=func)

    b = sub.add_parser("bench", help="run the standardized benchmark suite")
    b.add_argument("--endpoint", default=None,
                   help="comma list of ports or base URLs (bare port -> --host)")
    b.add_argument("--all-dev", action="store_true", help="dev set 45798-45801 on --host")
    b.add_argument("--all-prod", action="store_true", help="prod set 45600-45603 on --host (read-only)")
    b.add_argument("--host", default="127.0.0.1")
    b.add_argument("--model", default=None, help="model id (default: first /v1/models entry)")
    b.add_argument("--prompts", default=None,
                   help="comma list of prompt names (default: all 3)")
    b.add_argument("--concurrency", type=int, default=1)
    b.add_argument("--timeout", type=float, default=600.0, help="per-request timeout s")
    b.add_argument("--logs", default="",
                   help="comma list of llama-server log paths (one per endpoint) for spec stats")
    b.add_argument("--out-dir", default="bench/results")
    b.add_argument("--stem", default="bench")
    b.set_defaults(func=cmd_bench)

    oa = sub.add_parser("opc-apply", help="write OpenCode provider entries")
    _add_conn_opts(oa)
    oa.add_argument("--dry-run", action="store_true", help="report changes without writing")
    oa.add_argument("--no-backup", action="store_true", help="skip the .bak timestamped copy")
    oa.add_argument("--make-default", action="store_true",
                    help="also set the top-level model to the first slot's provider/model")
    oa.set_defaults(func=cmd_opc_apply)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
