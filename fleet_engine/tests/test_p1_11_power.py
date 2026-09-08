"""P1-11 tests: power-cap control (rocm-smi floor detection + apply).

All fake-rocm-smi: the module shells out to a `rocm-smi`-lookalike script
that records every call and returns canned range/set output. No GPU touched.
"""
from __future__ import annotations

import json
import stat

from fleet_engine.power import PowerRange, power_range, set_power  # noqa: E402

FAKE_BODY = """\
import sys, json
args = sys.argv[1:]
log_path = __LOG__
calls = json.load(open(log_path))
calls.append(args)
json.dump(calls, open(log_path, "w"))
if "--showminpower" in args:
    print("GPU[0]\\t\\t: Min Graphics Package Power (W): __MIN__")
elif "--showmaxpower" in args:
    print("GPU[0]\\t\\t: Max Graphics Package Power (W): __MAX__")
elif "--setpower" in args:
    watts = float(args[args.index("--setpower") + 1])
    if watts < __MINF__:
        print("ERROR: GPU[0]: Unable to set Power OverDrive")
        print("ERROR: GPU[0]\\t\\t: Value cannot be less than: __MINI__W")
    else:
        print("GPU[0]: Power OverDrive set to " + str(int(watts)) + "W")
"""


def make_fake_rocm(tmp_path, min_w: float, max_w: float) -> tuple[str, str]:
    log = tmp_path / "calls.json"
    log.write_text("[]")
    script = tmp_path / "rocm-smi"
    body = (
        FAKE_BODY
        .replace("__LOG__", repr(str(log)))
        .replace("__MIN__", f"{min_w:.1f}")
        .replace("__MAX__", f"{max_w:.1f}")
        .replace("__MINF__", repr(min_w))
        .replace("__MINI__", f"{min_w:.0f}")
    )
    script.write_text("#!/usr/bin/env python3\n" + body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script), str(log)


def test_power_range_reads_min_max(tmp_path):
    script, _log = make_fake_rocm(tmp_path, 250.0, 250.0)
    rng = power_range(script, gpu=0)
    assert isinstance(rng, PowerRange)
    assert rng.settable
    assert rng.min_w == 250.0
    assert rng.max_w == 250.0


def test_set_power_below_floor_refused_cleanly(tmp_path):
    script, log = make_fake_rocm(tmp_path, 250.0, 250.0)
    res = set_power(gpu=0, watts=140, rocm_smi=script)
    assert res["ok"] is False
    assert res["applied"] is False
    assert "floor 250" in res["reason"]
    assert res["range"] == {"min": 250.0, "max": 250.0}
    calls = json.load(open(log))
    # range was read, but the SET must NOT have been attempted
    assert not any("--setpower" in c for c in calls)


def test_set_power_above_max_refused(tmp_path):
    script, _ = make_fake_rocm(tmp_path, 250.0, 250.0)
    res = set_power(gpu=0, watts=300, rocm_smi=script)
    assert res["applied"] is False
    assert "above hardware max 250W" in res["reason"]


def test_set_power_at_floor_applies(tmp_path):
    script, log = make_fake_rocm(tmp_path, 250.0, 250.0)
    res = set_power(gpu=0, watts=250, rocm_smi=script)
    assert res["ok"] is True
    assert res["applied"] is True
    calls = json.load(open(log))
    assert any("--setpower" in c and "250" in c for c in calls)


def test_set_power_in_range_on_wider_card(tmp_path):
    script, log = make_fake_rocm(tmp_path, 50.0, 250.0)
    res = set_power(gpu=0, watts=160, rocm_smi=script)
    assert res["applied"] is True
    calls = json.load(open(log))
    assert any("--setpower" in c and "160" in c for c in calls)


def test_missing_binary_reports_error(tmp_path):
    res = set_power(gpu=0, watts=250, rocm_smi=str(tmp_path / "nope"))
    assert res["ok"] is False
    assert res["applied"] is False
    assert "No such file" in res["reason"] or "rocm-smi error" in res["reason"]
