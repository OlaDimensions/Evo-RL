#!/usr/bin/env bash
set -euo pipefail

EVO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VT3_ROOT="${VT3_ROOT:-/home/ola/miniforge3/envs/vt3}"
VT3_SITE="${VT3_SITE:-$VT3_ROOT/lib/python3.10/site-packages}"
VT3_PY="${VT3_PY:-$VT3_ROOT/bin/python}"
EVO_UV_PYTHON="${EVO_UV_PYTHON:-/home/ola/miniforge3/envs/evo-rl/bin/python}"
EVO_UV_PROJECT_ENVIRONMENT="${EVO_UV_PROJECT_ENVIRONMENT:-$EVO_ROOT/.venv-gr00t-310}"
EVO_UV_EXTRA_ARGS=(${EVO_UV_EXTRA_ARGS:---extra intelrealsense --with pyzmq --with msgpack})

GR00T_ROOT="${GR00T_ROOT:-/home/ola/code/Isaac-GR00T}"
GR00T_HOST="${GR00T_HOST:-127.0.0.1}"
GR00T_PORT="${GR00T_PORT:-5555}"
GR00T_TIMEOUT_MS="${GR00T_TIMEOUT_MS:-15000}"
GR00T_ACTION_STEPS="${GR00T_ACTION_STEPS:-8}"
GR00T_CHUNK_SIZE="${GR00T_CHUNK_SIZE:-16}"
POLICY_ACTION_SCHEMA="${POLICY_ACTION_SCHEMA:-bimanual_ee_rpy}"
DATASET_REPO_ID="${DATASET_REPO_ID:-ruanafan/eval_evo-rl-data-pnp-vr-ee-pose-gr00t-${POLICY_ACTION_SCHEMA#bimanual_ee_}-$(date +%m%d-%H%M%S)}"

if [[ ! -d "$VT3_SITE" ]]; then
  echo "[vt3-ik] ERROR: VT3 site-packages not found: $VT3_SITE" >&2
  exit 1
fi

if [[ ! -d "$GR00T_ROOT" ]]; then
  echo "[gr00t] ERROR: Isaac-GR00T root not found: $GR00T_ROOT" >&2
  exit 1
fi

_check_stack() {
  PYTHONPATH="$VT3_SITE:${PYTHONPATH:-}" \
  LD_LIBRARY_PATH="$VT3_ROOT/lib:${LD_LIBRARY_PATH:-}" \
  "$VT3_PY" - <<'PY'
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

_gr00t_env_has_runtime_deps() {
  local python_bin="$1"
  "$python_bin" - <<'PY' >/dev/null 2>&1
import msgpack  # noqa: F401
import pyrealsense2  # noqa: F401
import zmq  # noqa: F401
PY
}

if [[ -n "${EVO_RECORD_CMD:-}" ]]; then
  read -r -a LEROBOT_RECORD_CMD <<< "$EVO_RECORD_CMD"
elif command -v lerobot-human-inloop-record >/dev/null 2>&1; then
  LEROBOT_RECORD_CMD=(lerobot-human-inloop-record)
elif [[ -x "$EVO_UV_PROJECT_ENVIRONMENT/bin/lerobot-human-inloop-record" ]] \
  && _gr00t_env_has_runtime_deps "$EVO_UV_PROJECT_ENVIRONMENT/bin/python"; then
  LEROBOT_RECORD_CMD=("$EVO_UV_PROJECT_ENVIRONMENT/bin/lerobot-human-inloop-record")
elif command -v uv >/dev/null 2>&1; then
  if [[ -x "$EVO_UV_PROJECT_ENVIRONMENT/bin/lerobot-human-inloop-record" ]]; then
    echo "[evo-rl] existing uv env is missing pyrealsense2/pyzmq/msgpack; syncing extras."
  fi
  echo "[evo-rl] no installed lerobot-human-inloop-record found; uv may sync deps on first run."
  echo "[evo-rl] uv python: $EVO_UV_PYTHON"
  echo "[evo-rl] uv env:    $EVO_UV_PROJECT_ENVIRONMENT"
  echo "[evo-rl] uv args:   ${EVO_UV_EXTRA_ARGS[*]}"
  LEROBOT_RECORD_CMD=(uv run --python "$EVO_UV_PYTHON" "${EVO_UV_EXTRA_ARGS[@]}" lerobot-human-inloop-record)
else
  LEROBOT_RECORD_CMD=(python -m lerobot.scripts.lerobot_human_inloop_record)
fi
echo "[evo-rl] launching: ${LEROBOT_RECORD_CMD[*]}"

exec env \
  UV_PROJECT_ENVIRONMENT="$EVO_UV_PROJECT_ENVIRONMENT" \
  PYTHONPATH="$EVO_ROOT/src:$VT3_SITE:$GR00T_ROOT:${PYTHONPATH:-}" \
  LD_LIBRARY_PATH="$VT3_ROOT/lib:${LD_LIBRARY_PATH:-}" \
  "${LEROBOT_RECORD_CMD[@]}" \
  --robot.type=bi_piper_follower \
  --robot.id=my_bi_piper_follower \
  --robot.left_arm_config.port=can_left \
  --robot.right_arm_config.port=can_right \
  --robot.left_arm_config.require_calibration=false \
  --robot.right_arm_config.require_calibration=false \
  --robot.left_arm_config.cameras='{"wrist": {"type": "intelrealsense", "serial_number_or_name": "152122072280", "width": 640, "height":480, "fps": 30, "warmup_s": 2}}' \
  --robot.right_arm_config.cameras='{"wrist": {"type": "intelrealsense", "serial_number_or_name": "008222070618", "width": 640, "height":480, "fps": 30, "warmup_s": 2}, "front": {"type": "intelrealsense", "serial_number_or_name": "213622074413", "width": 640, "height":480, "fps": 30, "warmup_s": 2}}' \
  --teleop.type=bi_quest3_vr \
  --teleop.id=my_bi_vr_leader \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.single_task="Locate and pull open the air fryer drawer, pick up the sweet potato and place it steadily into the basket, then push the drawer back." \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=150 \
  --dataset.reset_time_s=20 \
  --dataset.push_to_hub=false \
  --display_data=false \
  --teleop.ik_pose_error_mode=reject \
  --teleop.ik_max_position_error_m=0.08 \
  --teleop.ik_max_orientation_error_deg=60 \
  --policy_tighten_closed_gripper=true \
  --policy_gripper_tighten_enter_threshold=50 \
  --policy_gripper_tighten_release_threshold=65 \
  --record_ee_pose=true \
  --policy_action_schema="$POLICY_ACTION_SCHEMA" \
  --policy_sync_to_teleop=true \
  --policy_sync_parallel=true \
  --policy.type=gr00t_remote \
  --policy.host="$GR00T_HOST" \
  --policy.port="$GR00T_PORT" \
  --policy.timeout_ms="$GR00T_TIMEOUT_MS" \
  --policy.device=cpu \
  --policy.gr00t_root="$GR00T_ROOT" \
  --policy.chunk_size="$GR00T_CHUNK_SIZE" \
  --policy.n_action_steps="$GR00T_ACTION_STEPS"
