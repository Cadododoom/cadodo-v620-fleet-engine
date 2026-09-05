"""Runtime manager (skeleton): llama-server command construction.

Phase 3 absorbs the production gpuN.sh launchers (spawn/stop/health/restart).
The command builder is the unit-testable core of that work, so it ships now.
Flags mirror the verified production launcher (gpu1.sh + config.env).
"""

from __future__ import annotations

from .config_store import SlotConfig


def _fmt_num(x: float) -> str:
    """Render a float without trailing .0 when it is integral."""
    return str(int(x)) if float(x).is_integer() else str(x)


def build_llama_server_cmd(cfg: SlotConfig, llama_bin: str = "llama-server") -> list[str]:
    """Build the full llama-server argv for one slot.

    Phase 3 diffs this output against the live gpuN.sh launchers to prove
    parity before any dev-port cutover.
    """
    cmd: list[str] = [
        llama_bin,
        "--model", cfg.model,
        "--host", cfg.host,
        "--port", str(cfg.port),
        "--n-gpu-layers", str(cfg.num_gpu_layers),
        "--threads", str(cfg.threads),
        "--ctx-size", str(cfg.ctx),
        "--rope-scaling", "yarn",
        "--rope-scale", _fmt_num(cfg.rope_scale),
        "--yarn-orig-ctx", str(cfg.yarn_orig_ctx),
        "--cache-type-k", cfg.kv_type,
        "--cache-type-v", cfg.kv_type,
    ]
    if cfg.mmproj:
        cmd += ["--mmproj", cfg.mmproj]
    cmd += ["--parallel", str(cfg.concurrency)]
    if cfg.kv_unified:
        cmd.append("--kv-unified")
    spec = cfg.spec
    if spec.spec_type != "none":
        spec_args = [
            "--spec-type", spec.spec_type,
        ]
        if "mtp" in spec.spec_type:
            spec_args += [
                "--spec-draft-n-max", str(spec.mtp_n_max),
                "--spec-draft-n-min", str(spec.mtp_n_min),
            ]
        if "ngram-mod" in spec.spec_type:
            spec_args += [
                "--spec-ngram-mod-n-min", str(spec.ngram_n_min),
                "--spec-ngram-mod-n-max", str(spec.ngram_n_max),
                "--spec-ngram-mod-n-match", str(spec.ngram_match),
            ]
        if spec.draft_model:
            spec_args += ["--draft-model", spec.draft_model]
        spec_args += ["--threads-draft", str(spec.draft_threads)]
        cmd += spec_args
    return cmd
