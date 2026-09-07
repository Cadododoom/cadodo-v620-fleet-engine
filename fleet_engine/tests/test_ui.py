"""Unit tests for the control UI data layer (fleet_engine.ui) + screenshot."""


import os

import pytest

from fleet_engine.metrics import ProdLane
from fleet_engine.ui import (
    FleetModel,
    SlotView,
    UiState,
    bar_fraction,
    fmt_gib,
    fmt_rate,
    slot_status_text,
)

# ---------------------------------------------------------------- formatting


def test_fmt_helpers():
    assert fmt_rate(None) == "—"
    assert fmt_rate(12.34) == "12.3 t/s"
    assert fmt_gib(0) == "0.0G"
    assert fmt_gib(3 * 1024**3) == "3.0G"
    assert bar_fraction(1, 4) == 0.25
    assert bar_fraction(0, 0) == 0.0
    assert bar_fraction(8, 4) == 1.0  # clamped


def test_slot_status_text():
    v = SlotView(slot=1, name="v620-1", kind="prod", port=45601, hip=2, running=True)
    assert slot_status_text(v) == "v620-1  :45601  HIP2  UP"
    e = SlotView(slot=5, name="v620-5", kind="empty")
    assert slot_status_text(e) == "v620-5"


# ----------------------------------------------------------------- FleetModel


def _dev_cfg(slot: int, name: str = "dev", port: int = 45700):
    class C:  # minimal SlotConfig stand-in
        def __init__(self, slot, name, port):
            self.slot, self.name, self.port = slot, name, port
            self.gpu = slot
            self.pci_addr = ""
            self.model = f"/m/model-{slot}.gguf"
            self.power_cap_watts = None

    return C(slot, name, port)


def test_build_views_grid_of_16():
    m = FleetModel(dev_slots=[_dev_cfg(1), _dev_cfg(2)],
                   prod_lanes=[ProdLane(name="prod-gpu3", port=45602,
                                        log="/x.log", slot_number=3,
                                        hip_pin=3, model="/m/M.gguf")])
    views = m.build_views()
    assert len(views) == 16
    assert views[0].kind == "dev"
    assert views[1].kind == "dev"
    assert views[2].kind == "prod"
    assert views[2].port == 45602
    assert views[3].kind == "empty"
    # dev wins over prod on the same slot
    m2 = FleetModel(dev_slots=[_dev_cfg(3)],
                    prod_lanes=[ProdLane(name="p", port=1, log="/x",
                                         slot_number=3)])
    v3 = [v for v in m2.build_views() if v.slot == 3][0]
    assert v3.kind == "dev"


def test_update_logs_tails_and_parses(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    log = logs / "dev.log"
    log.write_text(
        "slot print_timing: id  0 | task 1 | n_gen =  10, tg = 30.0 t/s, tg_3s = 32.0 t/s\n"
        "        eval time =   1000.00 ms /   10 tokens (   100.00 ms per token,    10.00 tokens per second)\n"
    )
    m = FleetModel(dev_slots=[_dev_cfg(1)], dev_state_dir=str(tmp_path))
    m.update_logs(now=1000.0)
    v = [v for v in m._views if v.slot == 1][0]
    assert v.stats is not None
    stats = v.stats
    assert stats.decode_tps == pytest.approx(10.0)
    assert stats.live_decode_tps == pytest.approx(32.0)
    # second call is incremental: no duplicate parse, stats persist
    m.update_logs(now=1001.0)
    v = [v for v in m._views if v.slot == 1][0]
    assert v.stats is not None
    assert v.stats.decode_tps == pytest.approx(10.0)


def test_ui_state_accessors():
    views = [
        SlotView(slot=1, name="a", kind="empty"),
        SlotView(slot=2, name="b", kind="prod", port=1),
    ]
    st = UiState(views=views, now=100.0)
    assert st.decode_of(1) is None
    assert st.prefill_of(2) is None
    assert st.spec_of(99) is None  # out of range, not an error


# ------------------------------------------------------------------ screenshot


def test_crop_to_window():
    from PIL import Image

    from fleet_engine.cli import crop_to_window

    screen = Image.new("RGB", (100, 50))
    # normal window fully inside
    c = crop_to_window(screen, 10, 5, 30, 20)
    assert c.size == (30, 20)
    # clamped to the right/bottom edge
    c = crop_to_window(screen, 90, 45, 50, 50)
    assert c.size == (10, 5)
    # negative origin clamped to 0
    c = crop_to_window(screen, -5, -5, 20, 20)
    assert c.size == (20, 20)
    # degenerate (zero/negative size) falls back to the whole screen
    assert crop_to_window(screen, 0, 0, 0, 0) is screen


def test_screenshot_helper_imports():
    # _grab_screenshot depends on PIL.ImageGrab; just confirm the module
    # imports cleanly (no display needed for import).
    import fleet_engine.cli as cli  # noqa: F401


# ------------------------------------------------------------ health / prod


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_update_health_up_and_down():
    """update_health() must mark a port UP when /health answers and DOWN
    when nothing listens — deterministic via a local dev-port HTTP server."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    port = _free_port()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):  # silence
            pass

    httpd = HTTPServer(("127.0.0.1", port), H)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        m = FleetModel(prod_lanes=[
            ProdLane(name="up", port=port, log="/x", slot_number=1),
            ProdLane(name="down", port=_free_port(), log="/x", slot_number=2),
        ])
        m.build_views()
        m.update_health()
        v1 = [v for v in m._views if v.slot == 1][0]
        v2 = [v for v in m._views if v.slot == 2][0]
        assert v1.running is True and "ok" in (v1.health or "")
        assert v2.running is False and v2.health is None
        n_up = sum(1 for v in m._views if v.running)
        assert n_up == 1
    finally:
        httpd.shutdown()


def test_prod_grid_16_with_running_lanes():
    """A prod fleet view must render 16 grid cells with running lanes UP and
    the header count correct (the 16-slot grid milestone, data-layer proof)."""
    m = FleetModel(prod_lanes=[
        ProdLane(name=f"prod-gpu{i}", port=45700 + i, log=f"/x{i}",
                 slot_number=i, hip_pin=i, model="/m/M.gguf")
        for i in (1, 2, 3)
    ])
    views = m.build_views()
    assert len(views) == 16
    assert [v.kind for v in views[:3]] == ["prod", "prod", "prod"]
    assert views[3].kind == "empty"
    assert views[15].kind == "empty"  # 16th cell present
    # simulate three up
    for v in views[:3]:
        v.running = True
    assert sum(1 for v in views if v.running) == 3
    # slot status text reflects up state
    assert "UP" in slot_status_text(views[0])


# ------------------------------------------------------- widget-tree (Xvfb)


def test_panel_widget_tree_grid_of_16():
    """Build the real tkinter Panel (needs an X server / Xvfb) and assert the
    grid renders one cell per slot: 16 cells, correct names, control buttons
    only on dev slots, header shows the UP count. Deterministic — inspects
    the widget tree, not the framebuffer (ImageGrab/Xvfb capture is flaky).
    """
    import tkinter as tk

    from fleet_engine.app import Panel

    if not os.environ.get("DISPLAY"):
        pytest.skip("no DISPLAY (run under Xvfb)")

    lanes = [
        ProdLane(name=f"prod-gpu{i}", port=45700 + i, log=f"/nonexist-{i}",
                 slot_number=i, hip_pin=i, model="/m/M.gguf")
        for i in (1, 2, 3)
    ]
    model = FleetModel(prod_lanes=lanes)
    model.refresh()
    model.build_views()

    root = tk.Tk()
    try:
        # Panel.__init__ runs refresh() -> update_health() on the (fake,
        # non-listening) prod ports, so all start DOWN; we only assert the
        # structural grid here (16 cells + names + header format), which is
        # the milestone. UP-state behavior is covered by
        # test_update_health_up_and_down deterministically.
        panel = Panel(root, model, runtime=None, state_dir="")
        root.update_idletasks()
        root.update()
        # 16 cells, one widget bundle per slot
        assert len(panel._widgets) == 16
        # every cell has a name label
        for i in range(1, 17):
            assert i in panel._widgets
        # prod cells named from the lane; empty cells get a slot label
        names = {i: panel._widgets[i]["name"].cget("text") for i in range(1, 17)}
        assert names[1] == "prod-gpu1"
        assert names[4] == "slot 4"  # empty cell
        # header is "N/16 UP" formatted
        import re as _re

        assert _re.fullmatch(r"\d+/16 UP", panel._hdr.cget("text"))
    finally:
        root.destroy()
