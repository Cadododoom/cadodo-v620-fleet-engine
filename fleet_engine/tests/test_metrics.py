"""Unit tests for metrics: llama-server log parsing, power, prod lanes."""

import subprocess
import textwrap

import pytest

from fleet_engine.metrics import (
    LogTailer,
    PowerTable,
    SlotLogStats,
    parse_log_chunk,
    parse_prod_lanes,
    rocm_smi_power,
    sysfs_vram_bytes,
)

# Real line shapes captured from the production fleet (2026-09-06)
LIVE_TG3S = ("1294.03.183.902 I slot print_timing: id  0 | task 71563 | "
             "n_gen =    467, tg =  23.61 t/s, tg_3s =  21.10 t/s")
EVAL_DONE = ("1294.09.020.576 I slot print_timing: id  0 | task 71563 | "
             "        eval time =   25577.06 ms /   583 tokens "
             "(   43.95 ms per token,    22.75 tokens per second)")
PROMPT_DONE = ("1294.09.020.571 I slot print_timing: id  0 | task 71563 | "
               "prompt eval time =  233288.82 ms / 71457 tokens "
               "(    3.26 ms per token,   306.30 tokens per second)")
PROMPT_LIVE = ("1294.09.020.500 I slot print_timing: id  0 | task 71563 | "
               "prompt processing, n_tokens =  2048, progress =  0.35, "
               "t =  0.12 s / 16824.51 tokens per second")
SPEC = ("1294.09.020.586 I slot print_timing: id  0 | task 71563 | "
        "draft acceptance = 0.46502 (  339 accepted /   729 generated), "
        "mean len =  2.40")


def test_parse_real_log_lines():
    st = SlotLogStats(path="t")
    now = 1000.0
    parse_log_chunk(LIVE_TG3S, st, now=now)
    assert st.live_decode_tps == pytest.approx(21.10)
    parse_log_chunk(EVAL_DONE, st, now=now)
    assert st.decode_tps == pytest.approx(22.75)
    assert st.decode_n == 583
    parse_log_chunk(PROMPT_DONE, st, now=now)
    assert st.prefill_tps == pytest.approx(306.30)
    assert st.prefill_n == 71457
    assert st.live_prefill_tps is None  # prompt finished clears in-flight
    parse_log_chunk(PROMPT_LIVE, st, now=now)
    assert st.live_prefill_tps == pytest.approx(16824.51)
    parse_log_chunk(SPEC, st, now=now)
    assert st.spec_accept == pytest.approx(0.46502)
    assert st.spec_accepted == 339
    assert st.spec_generated == 729
    assert st.spec_mean_len == pytest.approx(2.40)


def test_header_freshness_window():
    st = SlotLogStats()
    st.live_decode_tps = 99.0
    st.live_decode_ts = 1000.0
    st.decode_tps = 10.0
    # within window -> live wins
    assert st.header_decode(now=1004.0) == pytest.approx(99.0)
    # stale -> falls back to last completed
    assert st.header_decode(now=1006.0) == pytest.approx(10.0)
    # no live at all
    st2 = SlotLogStats(decode_tps=5.0)
    assert st2.header_decode(now=0.0) == pytest.approx(5.0)
    assert st2.header_prefill(now=0.0) is None


def test_log_tailer_incremental_and_rotation(tmp_path):
    p = tmp_path / "s.log"
    p.write_text("line1\n")
    t = LogTailer(str(p))
    assert t.poll() == "line1\n"
    assert t.poll() is None
    p.write_text("line1\nline2\n")
    assert t.poll() == "line2\n"
    p.write_text("fresh\n")  # truncated -> restart from 0
    assert t.poll() == "fresh\n"
    (tmp_path / "s.log").unlink()
    assert t.poll() is None


@pytest.fixture
def fake_drm(tmp_path):
    drm = tmp_path / "drm"
    drm.mkdir()
    # card0: V620 (73A1), card1: RX 5700 (731F)
    for name, dev in (("card0", "73a1"), ("card1", "731f")):
        d = drm / name / "device"
        d.mkdir(parents=True)
        (d / "uevent").write_text(
            f"DRIVER=amdgpu\nPCI_SLOT_NAME=0000:{name[4]}0:00.0\n"
        )
        (d / "vendor").write_text("0x1002\n")
        (d / "device").write_text(f"0x{dev}\n")
        (d / "mem_info_vram_total").write_text("32195477504\n")
        (d / "mem_info_vram_used").write_text("1000000000\n")
    return drm


def test_sysfs_vram_bytes(fake_drm):
    total, used = sysfs_vram_bytes("0000:00:00.0", str(fake_drm))
    assert total == 32195477504
    assert used == 1000000000
    assert sysfs_vram_bytes("0000:ff:00.0", str(fake_drm)) == (0, 0)


SAMPLE_POWER = """
============================ ROCm System Management Interface ============================
=================================== Power Consumption ====================================
GPU[0]\t\t: Average Graphics Package Power (W): 8.0
GPU[2]\t\t: Average Graphics Package Power (W): 98.0
==========================================================================================
================================== End of ROCm SMI Log ===================================
============================ ROCm System Management Interface ============================
================================== Memory Usage (Bytes) ==================================
GPU[0]\t\t: VRAM Total Memory (B): 32195477504
GPU[0]\t\t: VRAM Total Used Memory (B): 30414307328
==========================================================================================
================================== End of ROCm SMI Log ===================================
"""


def test_rocm_smi_power_parse(monkeypatch):
    def fake_run(*a, **k):
        class R:
            stdout = SAMPLE_POWER

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    t = rocm_smi_power()
    assert isinstance(t, PowerTable)
    assert t.power_w[2] == pytest.approx(98.0)
    assert t.vram_total[0] == 32195477504
    assert t.vram_used[0] == 30414307328


def test_rocm_smi_power_missing_bin(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("rocm-smi")

    monkeypatch.setattr(subprocess, "run", boom)
    t = rocm_smi_power()
    assert t.power_w == {}


def test_parse_prod_lanes(tmp_path):
    fleet = tmp_path / "fleet"
    (fleet / "logs").mkdir(parents=True)
    (fleet / "config.env").write_text('MODEL="/x/models/Big.gguf"\nPORT_BASE=45600\n')
    for i in (1, 2):
        (fleet / f"gpu{i}.sh").write_text(textwrap.dedent(f"""
            NAME="gpu{i}"
            GPU_INDEX={i}
            PORT=4560{i}
        """))
    lanes = parse_prod_lanes(str(fleet))
    assert len(lanes) == 2
    assert lanes[0].name == "prod-gpu1"
    assert lanes[0].port == 45601
    assert lanes[0].hip_pin == 1
    assert lanes[0].slot_number == 1
    assert lanes[0].model == "/x/models/Big.gguf"
    assert lanes[0].log.endswith("logs/gpu1.log")
    assert lanes[1].port == 45602


def test_parse_prod_lanes_missing_dir(tmp_path):
    assert parse_prod_lanes(str(tmp_path / "nope")) == []
