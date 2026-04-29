#!/usr/bin/env bash
set -euo pipefail


ROBOT_TYPE="${ROBOT_TYPE:-bi_piper_follower}"
ROBOT_ID="${ROBOT_ID:-my_bi_piper_follower}"
ROBOT_LEFT_PORT="${ROBOT_LEFT_PORT:-can_left}"
ROBOT_RIGHT_PORT="${ROBOT_RIGHT_PORT:-can_right}"
ROBOT_LEFT_REQUIRE_CALIBRATION="${ROBOT_LEFT_REQUIRE_CALIBRATION:-false}"
ROBOT_RIGHT_REQUIRE_CALIBRATION="${ROBOT_RIGHT_REQUIRE_CALIBRATION:-false}"
ROBOT_LEFT_CAMERAS=${ROBOT_LEFT_CAMERAS:-'{}'}
ROBOT_RIGHT_CAMERAS=${ROBOT_RIGHT_CAMERAS:-'{}'}

# ROBOT_LEFT_CAMERAS=${ROBOT_LEFT_CAMERAS:-'{"wrist": {"type": "intelrealsense", "serial_number_or_name": "152122072280", "width": 640, "height":480, "fps": 30, "warmup_s": 2}}'}
# ROBOT_RIGHT_CAMERAS=${ROBOT_RIGHT_CAMERAS:-'{"wrist": {"type": "intelrealsense", "serial_number_or_name": "008222070618", "width": 640, "height":480, "fps": 30, "warmup_s": 2}, "front": {"type": "intelrealsense", "serial_number_or_name": "213622074413", "width": 640, "height":480, "fps": 30, "warmup_s": 2}}'}

TELEOP_TYPE="${TELEOP_TYPE:-bi_quest3_vr}"
TELEOP_ID="${TELEOP_ID:-my_bi_vr_leader}"

DATASET_REPO_ID="${DATASET_REPO_ID:-ruanafan/eval_evo-rl-data-pnp-vr-ee-pose-round1-sweet-potato-0426-all-drop-last-frame-infer-1}"
DATASET_SINGLE_TASK="${DATASET_SINGLE_TASK:-Locate and pull open the air fryer drawer, pick up the sweet potato and place it steadily into the basket, then push the drawer back.}"
DATASET_NUM_EPISODES="${DATASET_NUM_EPISODES:-10}"
DATASET_EPISODE_TIME_S="${DATASET_EPISODE_TIME_S:-150}"
DATASET_RESET_TIME_S="${DATASET_RESET_TIME_S:-20}"
DATASET_PUSH_TO_HUB="${DATASET_PUSH_TO_HUB:-false}"
RESUME="${RESUME:-false}"

DISPLAY_DATA="${DISPLAY_DATA:-false}"

TELEOP_IK_POSE_ERROR_MODE="${TELEOP_IK_POSE_ERROR_MODE:-reject}"
TELEOP_IK_MAX_POSITION_ERROR_M="${TELEOP_IK_MAX_POSITION_ERROR_M:-0.08}"
TELEOP_IK_MAX_ORIENTATION_ERROR_DEG="${TELEOP_IK_MAX_ORIENTATION_ERROR_DEG:-60}"

RECORD_EE_POSE="${RECORD_EE_POSE:-true}"

POLICY_TIGHTEN_CLOSED_GRIPPER="${POLICY_TIGHTEN_CLOSED_GRIPPER:-true}"
POLICY_GRIPPER_TIGHTEN_ENTER_THRESHOLD="${POLICY_GRIPPER_TIGHTEN_ENTER_THRESHOLD:-50}"
POLICY_GRIPPER_TIGHTEN_RELEASE_THRESHOLD="${POLICY_GRIPPER_TIGHTEN_RELEASE_THRESHOLD:-65}"
POLICY_ACTION_SCHEMA="${POLICY_ACTION_SCHEMA:-bimanual_ee_rpy}"
POLICY_SYNC_TO_TELEOP="${POLICY_SYNC_TO_TELEOP:-true}"
POLICY_SYNC_PARALLEL="${POLICY_SYNC_PARALLEL:-true}"
POLICY_MODE="${POLICY_MODE:-openpi_remote}"
POLICY_TYPE="${POLICY_TYPE:-openpi_remote}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-8000}"
POLICY_PATH="${POLICY_PATH:-}"

ACP_INFERENCE_ENABLE="${ACP_INFERENCE_ENABLE:-true}"


RECORD_ARGS=(
  "--resume=$RESUME"
  "--robot.type=$ROBOT_TYPE"
  "--robot.id=$ROBOT_ID"
  "--robot.left_arm_config.port=$ROBOT_LEFT_PORT"
  "--robot.right_arm_config.port=$ROBOT_RIGHT_PORT"
  "--robot.left_arm_config.require_calibration=$ROBOT_LEFT_REQUIRE_CALIBRATION"
  "--robot.right_arm_config.require_calibration=$ROBOT_RIGHT_REQUIRE_CALIBRATION"
  "--robot.left_arm_config.cameras=$ROBOT_LEFT_CAMERAS"
  "--robot.right_arm_config.cameras=$ROBOT_RIGHT_CAMERAS"
  "--teleop.type=$TELEOP_TYPE"
  "--teleop.id=$TELEOP_ID"
  "--dataset.repo_id=$DATASET_REPO_ID"
  "--dataset.single_task=$DATASET_SINGLE_TASK"
  "--dataset.num_episodes=$DATASET_NUM_EPISODES"
  "--dataset.episode_time_s=$DATASET_EPISODE_TIME_S"
  "--dataset.reset_time_s=$DATASET_RESET_TIME_S"
  "--dataset.push_to_hub=$DATASET_PUSH_TO_HUB"
  "--display_data=$DISPLAY_DATA"
  "--teleop.ik_pose_error_mode=$TELEOP_IK_POSE_ERROR_MODE"
  "--teleop.ik_max_position_error_m=$TELEOP_IK_MAX_POSITION_ERROR_M"
  "--teleop.ik_max_orientation_error_deg=$TELEOP_IK_MAX_ORIENTATION_ERROR_DEG"
  "--record_ee_pose=$RECORD_EE_POSE"
  "--policy_action_schema=$POLICY_ACTION_SCHEMA"
)

POLICY_AUX_ARGS=(
  "--policy_tighten_closed_gripper=$POLICY_TIGHTEN_CLOSED_GRIPPER"
  "--policy_gripper_tighten_enter_threshold=$POLICY_GRIPPER_TIGHTEN_ENTER_THRESHOLD"
  "--policy_gripper_tighten_release_threshold=$POLICY_GRIPPER_TIGHTEN_RELEASE_THRESHOLD"
  "--policy_sync_to_teleop=$POLICY_SYNC_TO_TELEOP"
  "--policy_sync_parallel=$POLICY_SYNC_PARALLEL"
  "--acp_inference.enable=$ACP_INFERENCE_ENABLE"
)

case "$POLICY_MODE" in
  teleop_only)
    ;;
  openpi_remote)
    RECORD_ARGS+=(
      "${POLICY_AUX_ARGS[@]}"
      "--policy.type=$POLICY_TYPE"
      "--policy.host=$POLICY_HOST"
      "--policy.port=$POLICY_PORT"
    )
    ;;
  local_path)
    if [[ -z "$POLICY_PATH" ]]; then
      echo "[evo-rl] ERROR: POLICY_PATH must be set when POLICY_MODE=local_path." >&2
      exit 1
    fi
    RECORD_ARGS+=(
      "${POLICY_AUX_ARGS[@]}"
      "--policy.path=$POLICY_PATH"
    )
    ;;
  *)
    echo "[evo-rl] ERROR: unknown POLICY_MODE '$POLICY_MODE'. Use teleop_only, openpi_remote, or local_path." >&2
    exit 1
    ;;
esac

exec env \
  PYTHONPATH="$VT3_SITE:${PYTHONPATH:-}" \
  LD_LIBRARY_PATH="$VT3_ROOT/lib:${LD_LIBRARY_PATH:-}" \
  lerobot-human-inloop-record \
  "${RECORD_ARGS[@]}"
