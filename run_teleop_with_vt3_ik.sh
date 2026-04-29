#!/usr/bin/env bash
set -euo pipefail

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
#   --dataset.repo_id=ruanafan/eval_evo-rl-data-pnp-vr-ee-pose-round1-0421-all-rpy-infer-5 \
#   --dataset.single_task="Locate and pull open the air fryer drawer, pick up the sweet potato and place it steadily into the basket, then push the drawer back." \
#   --dataset.num_episodes=10 \
#   --dataset.episode_time_s=100 \
#   --dataset.reset_time_s=30 \
#   --dataset.push_to_hub=false \
#   --display_data=false \
#   --teleop.ik_pose_error_mode=reject \
#   --teleop.ik_max_position_error_m=0.08 \
#   --teleop.ik_max_orientation_error_deg=60 \
#   --record_ee_pose=true \
#   --policy_action_schema=bimanual_ee_rpy \
#   --policy.path=outputs/train/pi05-vr-ee-pose-round0-0421-all-rpy/checkpoints/030000/pretrained_model/ \
#   --policy_sync_to_teleop=true \
#   --policy_sync_parallel=true


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
  --robot.left_arm_config.cameras='{}' \
  --robot.right_arm_config.cameras='{}' \
  --teleop.type=bi_quest3_vr \
  --teleop.id=my_bi_vr_leader \
  --dataset.repo_id=ruanafan/evo-rl-data-pnp-vr-ee-pose-round1-sweet-potato-0429-testumi \
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
  --policy_action_schema=bimanual_ee_rpy
  #--policy_sync_to_teleop=true \
  #--policy_sync_parallel=true \
  #--policy.path=outputs/train/pi05-vr-ee-pose-round0-0421-all-rpy/checkpoints/030000/pretrained_model/
  # --policy.type=openpi_remote \
  # --policy.host=127.0.0.1 \
  # --policy.port=8000 \
  # --acp_inference.enable=true
  # # --acp_inference.use_cfg=true \
  # --acp_inference.cfg_beta=1.0
  #> run_teleop_with_vt3_ik.log 2>&1

  # --policy.path=outputs/train/pi05-vr-ee-pose-round0-0421-all-rpy/checkpoints/030000/pretrained_model/


  # --robot.left_arm_config.cameras='{"wrist": {"type": "intelrealsense", "serial_number_or_name": "152122072280", "width": 640, "height":480, "fps": 30, "warmup_s": 2}}' \
  # --robot.right_arm_config.cameras='{"wrist": {"type": "intelrealsense", "serial_number_or_name": "008222070618", "width": 640, "height":480, "fps": 30, "warmup_s": 2}, "front": {"type": "intelrealsense", "serial_number_or_name": "213622074413", "width": 640, "height":480, "fps": 30, "warmup_s": 2}}' \
  
