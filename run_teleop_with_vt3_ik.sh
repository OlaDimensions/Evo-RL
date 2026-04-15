#!/usr/bin/env bash
set -euo pipefail

# Run lerobot teleoperation with vt3 IK stack (pinocchio/casadi/eigenpy/hppfcl)
# injected only for this process.
#
# Usage:
#   conda activate evo-rl-test
#   ./run_teleop_with_vt3_ik.sh --robot.type=... --teleop.type=quest3_vr
#
# Optional:
#   ./run_teleop_with_vt3_ik.sh --check-only

VT3_ROOT="${VT3_ROOT:-/home/ola/miniforge3/envs/vt3}"
VT3_SITE="${VT3_SITE:-$VT3_ROOT/lib/python3.10/site-packages}"
EVO_PY="${EVO_PY:-python}"

if [[ ! -d "$VT3_SITE" ]]; then
  echo "[vt3-ik] ERROR: VT3 site-packages not found: $VT3_SITE" >&2
  exit 1
fi

_check_stack() {
  PYTHONPATH="$VT3_SITE:${PYTHONPATH:-}" \
  LD_LIBRARY_PATH="$VT3_ROOT/lib:${LD_LIBRARY_PATH:-}" \
  "$EVO_PY" - <<'PY'
import pinocchio, casadi, eigenpy, hppfcl
print("[vt3-ik] pinocchio", pinocchio.__version__, pinocchio.__file__)
print("[vt3-ik] casadi   ", casadi.__version__, casadi.__file__)
print("[vt3-ik] eigenpy  ", eigenpy.__version__, eigenpy.__file__)
print("[vt3-ik] hppfcl   ", hppfcl.__version__, hppfcl.__file__)
from pinocchio import casadi as _cpin  # noqa: F401
print("[vt3-ik] pinocchio.casadi OK")
PY
}

if [[ "${1:-}" == "--check-only" ]]; then
  _check_stack
  exit 0
fi

_check_stack

exec env \
  PYTHONPATH="$VT3_SITE:${PYTHONPATH:-}" \
  LD_LIBRARY_PATH="$VT3_ROOT/lib:${LD_LIBRARY_PATH:-}" \
  # "$EVO_PY" -m lerobot.scripts.lerobot_teleoperate "$@"
  lerobot-teleoperate     --robot.type=piper_follower     --robot.port=can_left     --teleop.type=quest3_vr  --fps=30
