"""tkinter control UI: 16-slot grid + live rates + bars + log viewer.

Phase 5 of PLAN I. Front end only — all data comes from FleetModel
(ui.FleetModel), all control goes through fleet_engine.runtime.Runtime.
Stdlib only; runs under a normal X session or Xvfb (see cli `ui` command
with --screenshot for headless verification).
"""

from __future__ import annotations

import re
import sys
import threading
import time
from typing import Optional

from .ui import (
    BG,
    BORDER,
    CARD,
    COLS,
    CYAN,
    DIM,
    FG,
    GOLD,
    GREEN,
    MAGENTA,
    MAX_SLOTS,
    MONO,
    RED,
    REFRESH_MS,
    FleetModel,
    SlotView,
    bar_fraction,
    fmt_gib,
    fmt_rate,
)

# log viewer filter: prefill/decode/spec lines highlighted
RE_HL = re.compile(
    r"eval time|prompt eval|prompt processing|tg_3s|draft acceptance|mean len"
)


class Panel:
    """The whole window. Construct, call run() (blocking) or
    one_shot() for headless screenshot mode."""

    def __init__(self, root, model: FleetModel, runtime=None,
                 state_dir: str = "") -> None:
        self.root = root
        self.model = model
        self.runtime = runtime  # Runtime or None (prod-only mode)
        self.state_dir = state_dir
        self._busy = False
        self._log_focus: Optional[int] = None  # slot shown in log viewer
        self._widgets: dict[int, dict] = {}

        root.title("V620 FLEET ENGINE")
        root.configure(bg=BG)
        root.geometry("1280x860")
        self._build_grid()
        self._build_log_viewer()
        self.refresh()

    # ------------------------------------------------------------------ UI

    def _build_grid(self) -> None:
        import tkinter as tk

        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(top, text="V620 FLEET ENGINE", font=("Helvetica", 15, "bold"),
                 bg=BG, fg=CYAN).pack(side="left")
        n_up = sum(1 for v in self.model._views if v.running)
        self._hdr = tk.Label(top, text=f"{n_up}/{len(self.model._views)} UP",
                             font=(MONO, 10), bg=BG, fg=DIM)
        self._hdr.pack(side="right")

        grid = tk.Frame(self.root, bg=BG)
        grid.pack(fill="both", expand=True, padx=10, pady=4)

        for v in self.model._views:
            frame = tk.Frame(grid, bg=CARD, highlightbackground=BORDER,
                             highlightthickness=1)
            frame.grid(row=(v.slot - 1) // COLS, column=(v.slot - 1) % COLS,
                       sticky="nsew", padx=4, pady=4)
            for _ in range(COLS):
                grid.columnconfigure(_, weight=1)
            for _ in range(MAX_SLOTS // COLS):
                grid.rowconfigure(_, weight=1)

            dot = tk.Canvas(frame, width=14, height=14, bg=CARD,
                            highlightthickness=0)
            dot.grid(row=0, column=0, rowspan=2, sticky="w", padx=(8, 4), pady=8)

            name = tk.Label(frame, text=v.name, font=(MONO, 10, "bold"),
                            bg=CARD, fg=FG)
            name.grid(row=0, column=1, columnspan=2, sticky="w")

            sub = tk.Label(frame, text="", font=(MONO, 8), bg=CARD, fg=DIM)
            sub.grid(row=1, column=1, columnspan=2, sticky="w")

            rate = tk.Label(frame, text="", font=(MONO, 9), bg=CARD, fg=CYAN)
            rate.grid(row=2, column=0, columnspan=3, sticky="w", padx=8)

            spec = tk.Label(frame, text="", font=(MONO, 8), bg=CARD, fg=MAGENTA)
            spec.grid(row=3, column=0, columnspan=3, sticky="w", padx=8)

            vbar = tk.Canvas(frame, width=100, height=8, bg=CARD,
                             highlightthickness=0)
            vbar.grid(row=4, column=0, columnspan=3, sticky="w", padx=8, pady=(2, 0))
            pbar = tk.Canvas(frame, width=100, height=8, bg=CARD,
                             highlightthickness=0)
            pbar.grid(row=5, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4))

            btns = []
            if v.kind == "dev" and self.runtime is not None:
                for txt, cmd in (
                    ("ON", lambda v=v: self._ctrl("start", v)),
                    ("OFF", lambda v=v: self._ctrl("stop", v)),
                    ("RESTART", lambda v=v: self._ctrl("restart", v)),
                ):
                    b = tk.Button(frame, text=txt, width=7, font=(MONO, 8),
                                  command=cmd)
                    b.grid(row=6, column=len(btns), sticky="w", padx=(8, 2), pady=2)
                    btns.append(b)

            if v.kind != "empty":
                frame.bind("<Button-1>", lambda e, v=v: self._focus_log(v.slot))

            self._widgets[v.slot] = {
                "dot": dot, "name": name, "sub": sub, "rate": rate,
                "spec": spec, "vbar": vbar, "pbar": pbar,
            }

    def _build_log_viewer(self) -> None:
        import tkinter as tk

        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=10, pady=(2, 0))
        tk.Label(bar, text="LOG", font=(MONO, 9, "bold"), bg=BG, fg=GOLD
                 ).pack(side="left")
        self._log_title = tk.Label(bar, text="(select a slot)", font=(MONO, 8),
                                   bg=BG, fg=DIM)
        self._log_title.pack(side="left", padx=8)

        self._log = tk.Text(self.root, height=10, bg="#0c0f12", fg=FG,
                            font=(MONO, 8), state="disabled", wrap="none",
                            relief="flat")
        self._log.pack(fill="x", padx=10, pady=(0, 10))
        self._log.tag_configure("hl", foreground=CYAN)

    # ------------------------------------------------------------ refresh

    def refresh(self) -> None:
        now = time.time()
        self.model.update_logs(now=now)
        self.model.refresh()
        self.model.update_health()
        self._render(now)

    def _render(self, now: float) -> None:
        n_up = 0
        for v in self.model._views:
            n_up += 1 if v.running else 0
            self._render_cell(v, now)
        self._hdr.config(text=f"{n_up}/{len(self.model._views)} UP")
        self._render_log(v_focus=self._log_focus)

    def _render_cell(self, v: SlotView, now: float) -> None:
        w = self._widgets.get(v.slot)
        if not w:
            return
        if v.kind == "empty":
            w["dot"].delete("all")
            w["name"].config(text=f"slot {v.slot}", fg="#2c353d")
            w["sub"].config(text="—")
            w["rate"].config(text="")
            w["spec"].config(text="")
            return

        color = GREEN if v.running else RED
        if v.running:
            w["dot"].create_oval(2, 2, 12, 12, fill=color, outline="")
        else:
            w["dot"].delete("all")
            w["dot"].create_rectangle(2, 2, 12, 12, outline=color, width=2)

        hip = f"HIP {v.hip}" if v.hip is not None else "HIP ?"
        kind = "PROD" if v.kind == "prod" else "DEV"
        w["sub"].config(text=f"{kind} · {hip} · :{v.port}")

        dec = v.stats.header_decode(5.0, now) if v.stats else None
        pre = v.stats.header_prefill(5.0, now) if v.stats else None
        w["rate"].config(text=f"{fmt_rate(dec)}  ·  prefill {fmt_rate(pre)}",
                         fg=CYAN if v.running else DIM)
        if v.stats and v.stats.spec_accept is not None:
            ml = v.stats.spec_mean_len
            mlt = f"  ml={ml:.2f}" if ml is not None else ""
            w["spec"].config(
                text=f"spec {v.stats.spec_accept:.3f} "
                     f"({v.stats.spec_accepted}/{v.stats.spec_generated}){mlt}"
            )
        else:
            w["spec"].config(text="")

        self._draw_bar(w["vbar"], bar_fraction(v.vram_used, v.vram_total),
                       CYAN, f"{fmt_gib(v.vram_used)}/{fmt_gib(v.vram_total)}")
        cap = v.power_cap or 250
        pw = v.power_w
        frac = (pw / cap) if (pw is not None and cap) else 0.0
        self._draw_bar(w["pbar"], frac, GOLD,
                       f"{pw:.0f}W" if pw is not None else "—")

    def _draw_bar(self, canvas, frac: float, color: str, label: str) -> None:
        canvas.delete("all")
        w = 100
        canvas.create_rectangle(0, 0, w, 8, outline=BORDER, width=1)
        canvas.create_rectangle(1, 1, max(1, int(w * frac)), 7, fill=color,
                                outline="")
        canvas.create_text(w + 4, 4, anchor="w", text=label,
                           font=(MONO, 7), fill=DIM)

    def _render_log(self, v_focus: Optional[int]) -> None:
        if not v_focus:
            return
        v = self.model._views[v_focus - 1] if v_focus <= MAX_SLOTS else None
        if not v or not v.log:
            return
        self._log_title.config(text=f"{v.name} :{v.port}  ·  {v.log}")
        try:
            with open(v.log, "rb") as f:
                f.seek(max(0, f.seek(0, 2) - 262144))
                lines = f.read().decode("utf-8", "replace").splitlines()
        except OSError:
            lines = []
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        for line in lines[-400:]:
            self._log.insert("end", line + "\n", "hl" if RE_HL.search(line) else ())
        self._log.config(state="disabled")
        self._log.see("end")

    # ------------------------------------------------------------- control

    def _focus_log(self, slot: int) -> None:
        self._log_focus = slot
        self._render_log(slot)

    def _ctrl(self, op: str, v: SlotView) -> None:
        rt = self.runtime
        if self._busy or rt is None:
            return
        self._busy = True

        def work() -> None:
            from .config_store import ConfigStore

            try:
                store = ConfigStore(rt.paths.slots_json())
                cfg = store.get_slot(v.slot)
                if op == "start":
                    rt.start(cfg, wait_ready=True)
                elif op == "stop":
                    rt.stop(cfg)
                else:
                    rt.stop(cfg)
                    rt.start(cfg, wait_ready=True)
            except Exception as e:  # noqa: BLE001 - surfaced in stderr
                print(f"control {op} slot {v.slot}: {e}", file=sys.stderr)
            finally:
                self._busy = False
                self.root.after(0, self.refresh)

        threading.Thread(target=work, daemon=True).start()

    # -------------------------------------------------------------- run

    def run(self) -> None:
        self._tick()

    def _tick(self) -> None:
        try:
            self.refresh()
        except Exception as e:  # noqa: BLE001 - UI must not die
            print(f"refresh error: {e}", file=sys.stderr)
        self.root.after(REFRESH_MS, self._tick)


def main(argv: Optional[list[str]] = None) -> int:
    """Standalone entry: python -m fleet_engine.ui [--state-dir DIR]
    [--fleet-dir DIR]."""
    args = argv if argv is not None else sys.argv[1:]
    state_dir = ""
    fleet_dir = ""
    i = 0
    while i < len(args):
        if args[i] == "--state-dir":
            state_dir = args[i + 1]
            i += 2
        elif args[i] == "--fleet-dir":
            fleet_dir = args[i + 1]
            i += 2
        else:
            i += 1
    try:
        import tkinter as tk
    except ImportError:
        print("tkinter not available on this platform", file=sys.stderr)
        return 1

    model = FleetModel()
    runtime = None
    if state_dir:
        dev = FleetModel.for_dev(state_dir)
        model.dev_slots = dev.dev_slots
        model.dev_state_dir = state_dir
        from .runtime import Runtime

        runtime = Runtime(state_dir=state_dir, llama_bin="")
    if fleet_dir:
        model.prod_lanes = FleetModel.for_prod(fleet_dir).prod_lanes
    model.refresh()
    model.update_logs()

    root = tk.Tk()
    panel = Panel(root, model, runtime, state_dir)
    panel.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
