#!/usr/bin/env bash
# Source this file before launching lerobot-teleoperate to prefer a Pinocchio build
# that exposes pinocchio.casadi over the ROS Humble package.
#
# Usage:
#   source /home/ola/code/Evo-RL/fix_pinocchio_import.sh
#   lerobot-teleoperate ...

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This script is meant to be sourced, not executed. Use: source ${BASH_SOURCE[0]}"
    return 1 2>/dev/null || exit 1
fi

# Prefer the evo-rl-test environment when present, since it is the conda
# environment used by the teleoperation stack and is expected to contain the
# Pinocchio build with CasADi.
EVO_RL_ROOT="/home/ola/miniforge3/envs/evo-rl-test"
EVO_RL_PYTHON="${EVO_RL_ROOT}/bin/python3"
EVO_RL_SITE=""

if [[ -x "$EVO_RL_PYTHON" ]]; then
    EVO_RL_SITE="$($EVO_RL_PYTHON - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
    export PATH="${EVO_RL_ROOT}/bin:${PATH}"
fi

# If evo-rl-test is unavailable, fall back to the active Python interpreter.
if [[ -z "$EVO_RL_SITE" ]]; then
    EVO_RL_SITE="$(python3 - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else site.getusersitepackages())
PY
)"
fi

export PYTHONPATH="${EVO_RL_SITE}:${PYTHONPATH:-}"

CHECK_PYTHON="python3"
if [[ -x "$EVO_RL_PYTHON" ]]; then
    CHECK_PYTHON="$EVO_RL_PYTHON"
fi

if "$CHECK_PYTHON" - <<'PY' >/dev/null 2>&1
try:
    from pinocchio import casadi as cpin  # noqa: F401
except Exception:
    raise SystemExit(1)
PY
then
    echo "[fix_pinocchio_import] PATH updated to prefer: ${EVO_RL_ROOT}/bin"
    echo "[fix_pinocchio_import] PYTHONPATH updated to prefer: ${EVO_RL_SITE}"
    echo "[fix_pinocchio_import] pinocchio.casadi import check: OK"
else
    echo "[fix_pinocchio_import] WARNING: pinocchio.casadi still cannot be imported."
    echo "[fix_pinocchio_import] Current PATH: ${PATH}"
    echo "[fix_pinocchio_import] Current PYTHONPATH: ${PYTHONPATH}"
    echo "[fix_pinocchio_import] You may still be picking up ROS Humble's pinocchio or evo-rl-test lacks the casadi-enabled build."
fi
