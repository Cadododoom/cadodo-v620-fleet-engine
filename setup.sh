#!/usr/bin/env bash
# setup.sh - pre-flight check for the V620 Fleet Engine.
# Checks (does NOT build or install anything): ROCm devices, llama-server,
# ROCm vendor libs, python, GGUF models. Prints the exact commands for
# docs/INSTALL.md steps 2-5 with your paths filled in.
set -uo pipefail

fail=0
note() { printf '  %-14s %s\n' "$1" "$2"; }
bad()  { printf '  %-14s %s\n' "$1" "$2"; fail=1; }

# Resolve a "real" home: cron/container profiles may run with a synthetic
# HOME (e.g. /home/<user>/.hermes/home). The repo itself lives under
# <real_home>/AI_Workstation_Work, so a pwd walk up to AI_Workstation_Work
# is the strongest signal; a HOME walk with AI_Workstation_Work-only
# matching is the fallback (a stray models/ dir is not dispositive).
_real_home_from_dir() {  # $1 = dir; prints ancestor owning AI_Workstation_Work
  local d
  d="$1"
  for _ in 1 2 3 4 5; do
    [ -d "$d" ] || return 1
    [ -d "$d/AI_Workstation_Work" ] && { echo "$d"; return 0; }
    [ "$d" = "/" ] && return 1
    d=$(cd "$d/.." 2>/dev/null && pwd || true)
    [ -n "$d" ] || return 1
  done
  return 1
}
REAL_HOME=""
REAL_HOME=$(_real_home_from_dir "$(pwd)" 2>/dev/null || true)
[ -n "$REAL_HOME" ] || REAL_HOME=$(_real_home_from_dir "$HOME" 2>/dev/null || true)
[ -n "${FLEET_HOME:-}" ] && [ -d "${FLEET_HOME:-}" ] && REAL_HOME="$FLEET_HOME"
REAL_HOME="${REAL_HOME:-$HOME}"

echo "== V620 Fleet Engine pre-flight =="
echo "   (home used: $REAL_HOME)"

# 1. GPUs (read-only). Real card nodes only: /sys/class/drm/cardN/device
#    (display connectors live at cardN-<name> and are not iterated).
echo "-- GPUs --"
n=0; v620=0
for i in $(seq 0 15); do
  dev="/sys/class/drm/card$i/device/uevent"
  [ -e "$dev" ] || continue
  d=$(grep -h '^PCI_ID=' "$dev" 2>/dev/null | cut -d= -f2)
  v=$(grep -h '^PCI_SLOT_NAME=' "$dev" 2>/dev/null | cut -d= -f2)
  n=$((n + 1))
  if [ "$d" = "1002:73A1" ]; then v620=$((v620 + 1)); note "card$i" "V620 (PCI $d at $v)"; else note "card$i" "other GPU (PCI $d at $v)"; fi
done
if [ "$n" -ge 1 ]; then note "v620_count" "$v620 of $n"; else bad "sysfs" "no /sys/class/drm/cardN/device found"; fi
[ "$v620" -ge 1 ] || bad "v620" "no 1002:73A1 (V620) PCI device found"

# 2. llama-server
echo "-- llama-server --"
LLAMA_BIN="${LLAMA_BIN:-}"
if [ -z "$LLAMA_BIN" ]; then
  for c in "$REAL_HOME/AI_Workstation_Work/llama.cpp/build-rocm/bin/llama-server" \
           /usr/local/bin/llama-server /usr/bin/llama-server; do
    [ -x "$c" ] && { LLAMA_BIN="$c"; break; }
  done
fi
if [ -n "$LLAMA_BIN" ] && [ -x "$LLAMA_BIN" ]; then
  note "binary" "$LLAMA_BIN"
else
  bad "binary" "not found (build llama.cpp with -DGGML_ROCM=ON; YARN support required for > native ctx)"
fi

# 3. ROCm vendor libs
echo "-- ROCm libs --"
ROC_VENDOR="${ROC_VENDOR:-}"
if [ -z "$ROC_VENDOR" ]; then
  for c in "$REAL_HOME/.lmstudio/extensions/backends/vendor/linux-llama-rocm-vendor-v4" \
           /opt/rocm/lib; do
    [ -d "$c" ] && { ROC_VENDOR="$c"; break; }
  done
fi
[ -n "$ROC_VENDOR" ] && [ -d "$ROC_VENDOR" ] && note "vendor" "$ROC_VENDOR" || bad "vendor" "not found (set ROC_VENDOR=/path/to/lib)"

# 4. python
echo "-- python --"
PY="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
[ -n "$PY" ] || PY=$(command -v python3 || true)
if [ -n "$PY" ]; then
  note "python" "$($PY --version 2>&1)"
  [ -d .venv ] && [ -x .venv/bin/python ] && PY=".venv/bin/python"
else
  bad "python" "python3 not found"
fi

# 5. GGUF models
echo "-- GGUF models --"
MODELS_DIR="${MODELS_DIR:-}"
found=0
for c in "$MODELS_DIR" "$REAL_HOME/models" "$REAL_HOME/.lmstudio/models"; do
  [ -n "$c" ] && [ -d "$c" ] || continue
  hits=$(find "$c" -name '*.gguf' 2>/dev/null | head -3)
  if [ -n "$hits" ]; then
    note "models" "$(echo "$hits" | head -1 | xargs basename) (+$(find "$c" -name '*.gguf' 2>/dev/null | wc -l) total under $c)"
    found=1
  fi
done
[ "$found" -eq 1 ] || bad "models" "no .gguf found (set MODELS_DIR=/path/to/models)"

echo
if [ "$fail" -eq 0 ]; then
  echo "== OK - next steps (paths filled) =="
  cat <<EOF
  $PY -m fleet_engine detect --llama-bin "$LLAMA_BIN" --roc-vendor "$ROC_VENDOR" --slots-json /opt/fleet/slots.json
  # then: edit slots.json (docs/INSTALL.md step 3) and
  $PY -m fleet_engine start --state-dir /opt/fleet --llama-bin "$LLAMA_BIN" --roc-vendor "$ROC_VENDOR"
  # baseline check:
  $PY -m fleet_engine bench --endpoint 45800 --out-dir bench/results
EOF
  exit 0
else
  echo "== FAILURES above - fix and re-run =="
  exit 1
fi
