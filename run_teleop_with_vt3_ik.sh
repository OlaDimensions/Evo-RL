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

# rm -rf /home/ola/.cache/huggingface/lerobot/ruanafan/evo-rl-data-pnp-vr-test-ee-pose

# exec env \
#   PYTHONPATH="$VT3_SITE:${PYTHONPATH:-}" \
#   LD_LIBRARY_PATH="$VT3_ROOT/lib:${LD_LIBRARY_PATH:-}" \
#   lerobot-human-inloop-record \
#   --robot.type=bi_piper_follower \
#   --robot.id=my_bi_piper_follower \
#   --robot.left_arm_config.port=can_left \
#   --robot.right_arm_config.port=can_right \
#   --robot.left_arm_config.require_calibration=false \
#   --robot.right_arm_config.require_calibration=false \
#   --robot.left_arm_config.cameras='{"wrist": {"type": "intelrealsense", "serial_number_or_name": "152122072280", "width": 640, "height":480, "fps": 30, "warmup_s": 2}}' \
#   --robot.right_arm_config.cameras='{"wrist": {"type": "intelrealsense", "serial_number_or_name": "008222070618", "width": 640, "height":480, "fps": 30, "warmup_s": 2}, "front": {"type": "intelrealsense", "serial_number_or_name": "213622074413", "width": 640, "height":480, "fps": 30, "warmup_s": 2}}' \
#   --teleop.type=bi_quest3_vr \
#   --teleop.id=my_bi_vr_leader \
#   --dataset.repo_id=ruanafan/evo-rl-data-pnp-vr-test-ee-pose \
#   --dataset.single_task="Pick up the white bottle and insert it into the middle of the tape roll" \
#   --dataset.num_episodes=2 \
#   --dataset.episode_time_s=60 \
#   --dataset.reset_time_s=8 \
#   --dataset.push_to_hub=false \
#   --display_data=false \
#   --teleop.ik_pose_error_mode=reject \
#   --teleop.ik_max_position_error_m=0.08 \
#   --teleop.ik_max_orientation_error_deg=60 \
#   --record_ee_pose=true

exec env \
  PYTHONPATH="$VT3_SITE:${PYTHONPATH:-}" \
  LD_LIBRARY_PATH="$VT3_ROOT/lib:${LD_LIBRARY_PATH:-}" \
  lerobot-human-inloop-record \
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
  --dataset.repo_id=ruanafan/evo-rl-data-pnp-vr-ee-pose-round0-0417-1 \
  --dataset.single_task="Pick up the white bottle and insert it into the middle of the tape roll" \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=8 \
  --dataset.push_to_hub=false \
  --display_data=false \
  --teleop.ik_pose_error_mode=reject \
  --teleop.ik_max_position_error_m=0.08 \
  --teleop.ik_max_orientation_error_deg=60 \
  --record_ee_pose=true
  # > teleop_with_vt3_ik_ee_pose.log 2>&1
  
  # lerobot-human-inloop-record  \  #single
  # --robot.type=piper_follower     \
  # --robot.id=my_piper_follower     \
  # --robot.port=can_left     \
  # --robot.require_calibration=false     \
  # --robot.cameras='{"wrist": {"type": "intelrealsense", "serial_number_or_name": "008222070618", "width": 640, "height": 480, "fps": 30, "warmup_s": 2}, "front": {"type": "intelrealsense", "serial_number_or_name": "213622074413", "width": 640, "height": 480, "fps": 30, "warmup_s": 2}}'     \
  # --teleop.type=quest3_vr     \
  # --teleop.id=my_vr_leader     \
  # --dataset.repo_id=ruanafan/evo-rl-data-pnp-vr-test     \
  # --dataset.single_task="Pick up the white bottle and insert it into the middle of the tape roll"     \
  # --dataset.num_episodes=2     \
  # --dataset.episode_time_s=30     \
  # --dataset.reset_time_s=5     \
  # --dataset.push_to_hub=false     \
  # --display_data=false

  # lerobot-teleoperate     --robot.type=piper_follower     --robot.port=can_left     --teleop.type=quest3_vr  --fps=30


  # "$EVO_PY" -m lerobot.scripts.lerobot_teleoperate "$@"
  # lerobot-teleoperate     --robot.type=piper_follower     --robot.port=can_left     --teleop.type=quest3_vr  --fps=30
