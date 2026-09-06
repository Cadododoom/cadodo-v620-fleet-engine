"""Runtime manager: spawn/stop/status/restart for llama-server slots.

Absorbs the production gpuN.sh launcher logic (Phase 3 of PLAN I):
 - HIP pin + ROCm vendor libs env, mirroring config.env
 - per-slot log + pidfile lifecycle under a runtime state dir
 - detached supervisor process: relaunches the server on crash with
   backoff (2/4/8/16 s, cap 16; resets after 10 min uptime); gives up
   after 8 crashes in a window, leaving a GIVEUP log line
 - /health readiness polling after spawn

Dev-only by default: ports 45700+, state dir under the campaign.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from .config_store import ConfigStore, SlotConfig
from .runtime_cmd import build_llama_server_cmd

# Production parity reference: LM Studio's vendored ROCm libs (read-only).
DEFAULT_ROC_VENDOR = "/home/theworks/.lmstudio/extensions/backends/vendor/linux-llama-rocm-vendor-v4"

LogFn = Callable[[str], None]


@dataclass
class RuntimePaths:
    """Where runtime state for a slot lives (pids, logs, slots.json)."""

    base: str

    def log(self, name: str) -> str:
        return os.path.join(self.base, "logs", f"{name}.log")

    def pidfile(self, name: str) -> str:
        return os.path.join(self.base, "pids", f"{name}.pid")

    def supervisor_pidfile(self, name: str) -> str:
        return os.path.join(self.base, "pids", f"{name}.supervisor.pid")

    def slots_json(self) -> str:
        return os.path.join(self.base, "slots.json")

    def ensure(self) -> None:
        os.makedirs(os.path.join(self.base, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.base, "pids"), exist_ok=True)


@dataclass
class ServerState:
    name: str
    running: bool
    pid: Optional[int]
    supervisor_pid: Optional[int]
    port: int
    health: Optional[str]
    log: str


def _write_pid(path: str, pid: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(pid))


def _read_pid(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _exists(pid: Optional[int]) -> bool:
    """Process exists, i.e. not dead and not a zombie.

    kill -0 succeeds for zombies, so we check the state field of
    /proc/<pid>/stat. Works for own children, reparented children, and
    foreign processes alike."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            # stat field 3 is the state letter, after the comm field
            # (which may contain spaces/parens, hence rsplit on ')')
            state = f.read().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False  # vanished between kill -0 and the read
    return state != "Z"


_alive = _exists


def _health(port: int, host: str = "127.0.0.1", timeout: float = 2.0) -> Optional[str]:
    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace").strip()
    except (urllib.error.URLError, OSError):
        return None


class Runtime:
    """Manages llama-server slots from one slots.json in a state dir."""

    def __init__(
        self,
        state_dir: str,
        llama_bin: str,
        roc_vendor: Optional[str] = None,
        poll: float = 0.25,
    ) -> None:
        self.paths = RuntimePaths(state_dir)
        self.paths.ensure()
        self.llama_bin = llama_bin
        self.roc_vendor = roc_vendor
        self.poll = poll
        self.store = ConfigStore(self.paths.slots_json())

    def start(
        self,
        cfg: SlotConfig,
        wait_ready: bool = True,
        ready_timeout: float = 240.0,
        log_fn: LogFn = print,
    ) -> int:
        """Spawn the slot (supervised). Returns the server pid."""
        name = cfg.name
        if self._server_pid(name) is not None:
            raise RuntimeError(f"[{name}] already running")
        if not os.path.isfile(cfg.model):
            raise FileNotFoundError(f"[{name}] model not found: {cfg.model}")
        if not os.access(self.llama_bin, os.X_OK):
            raise PermissionError(f"[{name}] engine missing/not executable: {self.llama_bin}")
        log = self.paths.log(name)
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} fleet_engine start ===\n")
        sup_cmd = [
            "python3", "-m", "fleet_engine", "runtime-supervisor",
            "--state-dir", self.paths.base,
            "--slot", str(cfg.slot),
            "--llama-bin", self.llama_bin,
        ]
        if self.roc_vendor:
            sup_cmd += ["--roc-vendor", self.roc_vendor]
        sup_env = dict(os.environ)
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sup_env["PYTHONPATH"] = (
            f"{pkg_root}:{sup_env['PYTHONPATH']}" if sup_env.get("PYTHONPATH") else pkg_root
        )
        with open(log, "a", encoding="utf-8") as lf:
            sup = subprocess.Popen(
                sup_cmd,
                stdout=lf,
                stderr=lf,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=sup_env,
            )
        _write_pid(self.paths.supervisor_pidfile(name), sup.pid)
        log_fn(f"[{name}] supervisor pid {sup.pid} started")
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            if not _alive(sup.pid):
                raise RuntimeError(f"[{name}] supervisor died during startup; log: {log}")
            pid = self._server_pid(name)
            if pid is not None and (not wait_ready or _health(cfg.port) is not None):
                log_fn(f"[{name}] READY on port {cfg.port} (pid {pid})")
                return pid
            time.sleep(self.poll)
        raise TimeoutError(f"[{name}] not ready after {ready_timeout}s; log: {log}")

    def stop(self, cfg: SlotConfig, timeout: float = 30.0, log_fn: LogFn = print) -> None:
        """Stop supervisor + server for a slot. Idempotent.

        Order matters: SIGTERM the supervisor first (it terminates its own
        server child), then the server directly in case the supervisor is
        gone. Then wait, escalate to SIGKILL, and finally wait for the
        supervisor to have exited (so it does not respawn our server).
        """
        name = cfg.name
        srv = self._server_pid(name)
        sup = _read_pid(self.paths.supervisor_pidfile(name))
        for pid in (sup, srv):
            if _exists(pid):
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not _exists(srv) and not _exists(sup):
                break
            time.sleep(0.2)
        for pid in (sup, srv):
            if _exists(pid):
                log_fn(f"[{name}] SIGTERM ignored, SIGKILL pid {pid}")
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        # after SIGKILL the supervisor can no longer respawn: settle briefly
        # then clear pidfiles (last-writer wins; our server is already gone)
        time.sleep(0.3)
        for pid in (sup, srv):
            if _exists(pid):
                log_fn(f"[{name}] pid {pid} still present after SIGKILL")
        self._clear_pidfiles(name)
        log_fn(f"[{name}] stopped")

    def _clear_pidfiles(self, name: str) -> None:
        for p in (self.paths.pidfile(name), self.paths.supervisor_pidfile(name)):
            try:
                os.unlink(p)
            except OSError:
                pass

    def status(self, cfg: SlotConfig) -> ServerState:
        name = cfg.name
        srv = self._server_pid(name)
        sup = _read_pid(self.paths.supervisor_pidfile(name))
        if sup is not None and not _alive(sup):
            sup = None
        running = _alive(srv)
        return ServerState(
            name=name,
            running=running,
            pid=srv if running else None,
            supervisor_pid=sup,
            port=cfg.port,
            health=_health(cfg.port) if running else None,
            log=self.paths.log(name),
        )

    def _server_pid(self, name: str) -> Optional[int]:
        pid = _read_pid(self.paths.pidfile(name))
        return pid if _alive(pid) else None


# -- supervisor process (detached, one per slot) -------------------------------

def run_supervisor(
    state_dir: str,
    slot: int,
    llama_bin: str,
    roc_vendor: Optional[str],
    crash_window: float = 600.0,
    max_crashes: int = 8,
) -> int:
    """Supervise one slot: launch server, relaunch on crash with backoff.

    Exits (cleanly) when: its supervisor pidfile is gone (stop was called),
    or the model disappears, or max_crashes inside crash_window seconds.
    """
    store = ConfigStore(os.path.join(state_dir, "slots.json"))
    cfg = store.get_slot(slot)
    name = cfg.name
    paths = RuntimePaths(state_dir)
    paths.ensure()
    log = paths.log(name)
    _stop = {"flag": False}

    def _on_term(signum, _frame) -> None:
        _stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    def write_log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")
        print(line, flush=True)

    def sup_alive() -> bool:
        me = _read_pid(paths.supervisor_pidfile(name))
        return me is not None and _alive(me)

    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = str(cfg.gpu)
    if roc_vendor:
        env["LD_LIBRARY_PATH"] = (
            f"{roc_vendor}:{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else roc_vendor
        )
    cmd = build_llama_server_cmd(cfg, llama_bin=llama_bin)
    backoff = 2.0
    crash_times: list[float] = []
    while sup_alive() and not _stop["flag"]:
        if not os.path.isfile(cfg.model):
            write_log(f"GIVEUP: model missing {cfg.model}")
            break
        with open(log, "a", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf, stdin=subprocess.DEVNULL)
        _write_pid(paths.pidfile(name), proc.pid)
        write_log(f"launched server pid {proc.pid} on port {cfg.port}")
        t0 = time.time()
        # wait in small slices so the SIGTERM handler can run (a blocking
        # proc.wait() would starve the signal)
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if _stop["flag"]:
                proc.terminate()
            time.sleep(0.2)
        rc = proc.wait()
        now = time.time()
        if not sup_alive() or _stop["flag"]:
            break
        write_log(f"server pid {proc.pid} exited rc={rc} after {now - t0:.0f}s")
        if now - t0 > 600:
            backoff = 2.0  # long-lived: reset backoff
        crash_times.append(now)
        crash_times = [t for t in crash_times if now - t <= crash_window]
        if len(crash_times) >= max_crashes:
            write_log(f"GIVEUP: {len(crash_times)} crashes within {crash_window:.0f}s; exiting")
            break
        time.sleep(backoff)
        backoff = min(backoff * 2, 16.0)
    # cleanup: make sure no orphaned server lingers
    srv = _read_pid(paths.pidfile(name))
    if srv is not None and _alive(srv):
        try:
            os.kill(srv, signal.SIGTERM)
        except OSError:
            pass
    try:
        os.unlink(paths.pidfile(name))
    except OSError:
        pass
    return 0
