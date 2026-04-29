# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core recording loop used by `lerobot_record.py`."""

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import numpy as np

from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.robots import Robot
from lerobot.scripts.recording_hil import (
    INTERVENTION_STATE_ACTIVE,
    INTERVENTION_STATE_POLICY,
    INTERVENTION_STATE_RELEASE,
    ACPInferenceConfig,
    PolicySyncDualArmExecutor,
    _capture_policy_runtime_state,
    _predict_policy_action_with_acp_inference,
)
from lerobot.scripts.ee_pose_action_utils import (
    detect_bimanual_ee_schema,
    tighten_closed_policy_grippers,
    with_bimanual_ee_enabled_flags,
)
from lerobot.teleoperators import Teleoperator, koch_leader, omx_leader, so_leader
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.recording_annotations import resolve_collector_policy_id
from lerobot.utils.piper_sdk import PIPER_JOINT_ACTION_KEYS
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.visualization_utils import log_rerun_data

T = TypeVar("T")


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     .-----( ACTION LOGIC )------------------.
     V                                       V
     [ From Teleoperator ]                   [ From Policy ]
     |                                       |
     |  [teleop.get_action] -> raw_action    |   [predict_action]
     |          |                            |          |
     |          V                            |          V
     | [teleop_action_processor]             |          |
     |          |                            |          |
     '---> processed_teleop_action           '---> processed_policy_action
     |                                       |
     '-------------------------.-------------'
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


def _complete_action_values_for_dataset(
    ds_features: dict[str, Any],
    values: RobotAction,
    observation: RobotObservation,
    previous_values: RobotAction | None = None,
) -> RobotAction:
    """Fill action fields required by the dataset without changing robot commands.

    Some teleop processors intentionally emit partial actions on idle ticks so
    robot.send_action() does not resend stale joint targets. Dataset frames still
    need every action feature. Prefer the current observation for missing action
    values, then fall back to the previous stored action if observation lacks it.
    """
    completed = dict(values)
    for key, ft in ds_features.items():
        if not key.startswith(ACTION) or ft.get("dtype") != "float32" or len(ft.get("shape", ())) != 1:
            continue
        for name in ft.get("names", ()):
            if name in completed:
                continue
            if name in observation:
                completed[name] = observation[name]
            elif previous_values is not None and name in previous_values:
                completed[name] = previous_values[name]
    return completed


def _zero_values_for_feature(ds_features: dict[str, Any], feature_key: str) -> RobotAction:
    feature = ds_features.get(feature_key)
    if not feature or feature.get("dtype") != "float32" or len(feature.get("shape", ())) != 1:
        return {}
    return dict.fromkeys(feature.get("names", ()), 0.0)


def _round_handoff_value(value: Any) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(number):
        return number
    return round(number, 4)


def _handoff_arm_prefixes(*actions: Mapping[str, Any] | None) -> tuple[str, ...]:
    keys: set[str] = set()
    for action in actions:
        if isinstance(action, Mapping):
            keys.update(str(key) for key in action)
    if any(key.startswith("left_") or key.startswith("right_") for key in keys):
        return ("left_", "right_")
    return ("",)


def _summarize_handoff_action(action: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(action, Mapping):
        return None

    summary: dict[str, Any] = {}
    prefixes = _handoff_arm_prefixes(action)
    for prefix in prefixes:
        label = prefix.removesuffix("_") or "single"
        arm: dict[str, Any] = {}

        for key in ("enabled", "reset", "gripper.pos"):
            full_key = f"{prefix}{key}"
            if full_key in action:
                arm[key] = _round_handoff_value(action[full_key])

        absolute_key = f"{prefix}__absolute_joint_targets__"
        if absolute_key in action:
            arm["absolute_joint_targets"] = bool(action[absolute_key])

        joint_values = [action.get(f"{prefix}{key}") for key in PIPER_JOINT_ACTION_KEYS]
        if any(value is not None for value in joint_values):
            arm["joints_deg"] = [_round_handoff_value(value) for value in joint_values]

        target_keys = (
            "ee.target_x",
            "ee.target_y",
            "ee.target_z",
            "ee.target_rx",
            "ee.target_ry",
            "ee.target_rz",
        )
        target_values = [action.get(f"{prefix}{key}") for key in target_keys]
        if all(value is not None for value in target_values):
            arm["ee_target"] = [_round_handoff_value(value) for value in target_values]

        quat_keys = ("ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw")
        quat_values = [action.get(f"{prefix}{key}") for key in quat_keys]
        if all(value is not None for value in quat_values):
            arm["ee_quat"] = [_round_handoff_value(value) for value in quat_values]

        rpy_keys = ("ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw")
        rpy_values = [action.get(f"{prefix}{key}") for key in rpy_keys]
        if all(value is not None for value in rpy_values):
            arm["ee_rpy"] = [_round_handoff_value(value) for value in rpy_values]

        delta_keys = (
            "ee.delta_x",
            "ee.delta_y",
            "ee.delta_z",
            "ee.delta_rx",
            "ee.delta_ry",
            "ee.delta_rz",
        )
        delta_values = [action.get(f"{prefix}{key}") for key in delta_keys]
        if all(value is not None for value in delta_values):
            arm["ee_delta"] = [_round_handoff_value(value) for value in delta_values]

        if arm:
            summary[label] = arm

    if not summary:
        summary["keys"] = sorted(str(key) for key in action)[:16]
    return summary


def _handoff_joint_delta_summary(
    command: Mapping[str, Any] | None,
    observation: Mapping[str, Any] | None,
) -> dict[str, float]:
    if not isinstance(command, Mapping) or not isinstance(observation, Mapping):
        return {}

    diffs: dict[str, float] = {}
    for prefix in _handoff_arm_prefixes(command, observation):
        deltas: list[float] = []
        for key in PIPER_JOINT_ACTION_KEYS:
            command_value = command.get(f"{prefix}{key}")
            observation_value = observation.get(f"{prefix}{key}")
            if command_value is None or observation_value is None:
                continue
            try:
                deltas.append(abs(float(command_value) - float(observation_value)))
            except (TypeError, ValueError):
                continue
        if deltas:
            diffs[prefix.removesuffix("_") or "single"] = round(max(deltas), 4)
    return diffs


def _call_processor_handoff_hook(processor: Any, hook_name: str, *args: Any) -> tuple[bool, bool]:
    called = False
    updated = False
    seen: set[int] = set()

    def _visit(obj: Any) -> None:
        nonlocal called, updated
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        hook = getattr(obj, hook_name, None)
        if callable(hook):
            called = True
            updated = bool(hook(*args)) or updated

        for step in getattr(obj, "steps", ()) or ():
            _visit(step)

    _visit(processor)
    return called, updated


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    policy_action_processor: (
        RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction] | None
    ) = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    policy_sync_executor: PolicySyncDualArmExecutor | None = None,
    intervention_state_machine_enabled: bool = True,
    collector_policy_id_policy: str = "policy",
    collector_policy_id_human: str = "human",
    acp_inference: ACPInferenceConfig | None = None,
    communication_retry_timeout_s: float = 2.0,
    communication_retry_interval_s: float = 0.1,
    control_features: dict[str, Any] | None = None,
    policy_features: dict[str, Any] | None = None,
    policy_observation_transform: Callable[[RobotObservation], RobotObservation] | None = None,
    ee_pose_storage: Any | None = None,
    policy_stationary_arm_delta_threshold: float = 1e-5,
    policy_tighten_closed_gripper: bool = False,
    policy_gripper_tighten_enter_threshold: float = 40.0,
    policy_gripper_tighten_release_threshold: float = 55.0,
    policy_gripper_tighten_value: float = 0.0,
    handoff_debug_enabled: bool = True,
    handoff_debug_log_frames: int = 8,
):
    if acp_inference is None:
        acp_inference = ACPInferenceConfig()

    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    if dataset is None and policy is not None:
        raise ValueError("Policy-driven recording requires a dataset for feature mapping.")

    if dataset is not None:
        features_for_policy = policy_features
        if features_for_policy is None:
            features_for_policy = control_features if control_features is not None else dataset.features
        features_for_storage_completion = control_features if control_features is not None else dataset.features
    else:
        features_for_policy = None
        features_for_storage_completion = None
    action_feature_names = features_for_policy[ACTION]["names"] if features_for_policy is not None else None
    if action_feature_names is None:
        if hasattr(robot.action_features, "keys"):
            action_feature_names = list(robot.action_features.keys())
        else:
            action_feature_names = list(robot.action_features)
    zero_policy_action = dict.fromkeys(action_feature_names, 0.0)
    has_teleop = isinstance(teleop, (Teleoperator, list))
    intervention_enabled = intervention_state_machine_enabled and policy is not None and has_teleop
    intervention_state = INTERVENTION_STATE_POLICY
    last_teleop_action: RobotAction | None = None
    last_action_values_for_storage: RobotAction | None = None
    last_policy_action_for_enabled: RobotAction | None = None
    teleop_fallback_warned = False
    pending_teleop_pose_sync = False
    pending_policy_release_baseline = False
    policy_gripper_tightened_keys: set[str] = set()
    handoff_debug_frames_remaining = 0
    handoff_debug_event_idx = 0

    teleop_arm_for_mode_switch: Any | None = None
    if isinstance(teleop, Teleoperator):
        teleop_arm_for_mode_switch = teleop
    elif isinstance(teleop, list):
        teleop_arm_for_mode_switch = teleop_arm

    def set_teleop_manual_control(enabled: bool) -> None:
        if teleop_arm_for_mode_switch is None:
            return
        if not hasattr(teleop_arm_for_mode_switch, "set_manual_control"):
            return
        try:
            teleop_arm_for_mode_switch.set_manual_control(enabled)
        except Exception:
            logging.exception("Failed to switch teleop manual-control mode to %s", enabled)

    if policy is None:
        # During reset/teleop-only loops keep leader backdrivable for manual dragging.
        set_teleop_manual_control(True)

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        if policy_action_processor is not None:
            policy_action_processor.reset()
            last_policy_action_for_enabled = None

    cond_policy_runtime_state: dict[str, Any] | None = None
    uncond_policy_runtime_state: dict[str, Any] | None = None
    if policy is not None and acp_inference.enable and acp_inference.use_cfg:
        cond_policy_runtime_state = _capture_policy_runtime_state(policy)
        uncond_policy_runtime_state = _capture_policy_runtime_state(policy)

    if ee_pose_storage is not None and hasattr(ee_pose_storage, "reset"):
        ee_pose_storage.reset()

    if intervention_enabled:
        # Start in S0: policy drives both arms, teleop arm should accept feedback commands.
        set_teleop_manual_control(False)

    def run_with_connection_retry(action_name: str, fn: Callable[[], T]) -> T:
        timeout_s = max(communication_retry_timeout_s, 0.0)
        interval_s = max(communication_retry_interval_s, 0.0)
        deadline_t = time.perf_counter() + timeout_s
        attempts = 0
        first_error: ConnectionError | None = None

        while True:
            attempts += 1
            try:
                result = fn()
                if attempts > 1:
                    elapsed_s = timeout_s - max(deadline_t - time.perf_counter(), 0.0)
                    logging.warning(
                        "%s recovered after %d retries in %.2fs.",
                        action_name,
                        attempts - 1,
                        elapsed_s,
                    )
                return result
            except ConnectionError as error:
                if first_error is None:
                    first_error = error
                    logging.warning(
                        "%s failed with transient communication error; retrying for up to %.2fs (%s)",
                        action_name,
                        timeout_s,
                        error,
                    )

                if timeout_s <= 0.0:
                    raise

                remaining_s = deadline_t - time.perf_counter()
                if remaining_s <= 0.0:
                    raise

                sleep_s = interval_s if interval_s > 0.0 else remaining_s
                time.sleep(min(sleep_s, remaining_s))

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        if events.get("toggle_intervention", False):
            events["toggle_intervention"] = False
            if handoff_debug_enabled and handoff_debug_log_frames > 0:
                handoff_debug_event_idx += 1
                handoff_debug_frames_remaining = int(handoff_debug_log_frames)
            if intervention_enabled:
                if intervention_state == INTERVENTION_STATE_POLICY:
                    intervention_state = INTERVENTION_STATE_ACTIVE
                    pending_teleop_pose_sync = True
                    set_teleop_manual_control(True)
                    logging.info("Intervention enabled (S1): teleop actions now override policy execution.")
                else:
                    intervention_state = INTERVENTION_STATE_RELEASE
                    set_teleop_manual_control(False)
                    if policy is not None and preprocessor is not None and postprocessor is not None:
                        policy.reset()
                        preprocessor.reset()
                        postprocessor.reset()
                        if policy_action_processor is not None:
                            policy_action_processor.reset()
                            last_policy_action_for_enabled = None
                            pending_policy_release_baseline = True
                        if acp_inference.enable and acp_inference.use_cfg:
                            cond_policy_runtime_state = _capture_policy_runtime_state(policy)
                            uncond_policy_runtime_state = _capture_policy_runtime_state(policy)
                    if policy is not None and preprocessor is not None and postprocessor is not None:
                        logging.info("Policy cache reset on release: next policy action is recomputed.")
                    logging.info("Intervention release requested (S2): returning control to policy.")
            else:
                logging.info("Intervention toggle ignored because policy+teleop are not both active.")

        # Get robot observation
        obs = robot.get_observation()

        if pending_teleop_pose_sync:
            pending_teleop_pose_sync = False
            sync_from_observation = getattr(teleop_arm_for_mode_switch, "sync_from_observation", None)
            if callable(sync_from_observation):
                try:
                    synced = bool(sync_from_observation(obs))
                    if synced:
                        logging.info("Teleop VR anchor synced from current robot observation.")
                    else:
                        logging.warning("Teleop VR anchor sync was requested but no arm pose was updated.")
                except Exception:
                    logging.exception("Failed to sync teleop VR anchor from current robot observation.")
            try:
                called, synced = _call_processor_handoff_hook(teleop_action_processor, "sync_from_observation", obs)
                if called and synced:
                    logging.info("Teleop IK state synced from current robot observation.")
                elif called:
                    logging.warning("Teleop IK state sync was requested but no arm state was updated.")
            except Exception:
                logging.exception("Failed to sync teleop IK state from current robot observation.")

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            ee_state_before_action = (
                run_with_connection_retry("ee_pose_storage.read_state", ee_pose_storage.read)
                if ee_pose_storage is not None
                else None
            )
            observation_values_for_policy = (
                {**obs_processed, **ee_state_before_action}
                if ee_state_before_action is not None
                else obs_processed
            )
            if policy_observation_transform is not None:
                observation_values_for_policy = policy_observation_transform(observation_values_for_policy)
            policy_observation_frame = build_dataset_frame(
                features_for_policy, observation_values_for_policy, prefix=OBS_STR
            )

        # Get action from policy and/or teleop
        act_processed_policy: RobotAction | None = None
        act_processed_teleop: RobotAction | None = None
        raw_teleop_action: RobotAction | None = None
        raw_policy_action_for_storage: RobotAction | None = None
        policy_action_for_execution: RobotAction | None = None
        if (
            policy is not None
            and preprocessor is not None
            and postprocessor is not None
            and not (intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE)
        ):
            policy_action = _predict_policy_action_with_acp_inference(
                observation_frame=policy_observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
                acp_inference=acp_inference,
                cond_runtime_state=cond_policy_runtime_state,
                uncond_runtime_state=uncond_policy_runtime_state,
            )
            raw_policy_action_for_storage = make_robot_action(policy_action, features_for_policy)
            if policy_tighten_closed_gripper:
                raw_policy_action_for_storage = tighten_closed_policy_grippers(
                    raw_policy_action_for_storage,
                    policy_gripper_tightened_keys,
                    enter_threshold=policy_gripper_tighten_enter_threshold,
                    release_threshold=policy_gripper_tighten_release_threshold,
                    tighten_value=policy_gripper_tighten_value,
                )
            if pending_policy_release_baseline and detect_bimanual_ee_schema(raw_policy_action_for_storage):
                try:
                    called, anchored = _call_processor_handoff_hook(
                        policy_action_processor,
                        "anchor_absolute_target_from_observation",
                        raw_policy_action_for_storage,
                        obs,
                    )
                    if called and anchored:
                        logging.info("Policy IK absolute EE anchor synced from current robot observation.")
                    elif called:
                        logging.warning("Policy IK absolute EE anchor sync was requested but no arm state was updated.")
                except Exception:
                    logging.exception("Failed to sync policy IK absolute EE anchor from current robot observation.")
                policy_action_for_execution = {
                    **raw_policy_action_for_storage,
                    "left_enabled": False,
                    "right_enabled": False,
                }
                pending_policy_release_baseline = False
                logging.info("Policy release baseline captured; suppressing bimanual EE motion for one tick.")
            else:
                policy_action_for_execution = with_bimanual_ee_enabled_flags(
                    raw_policy_action_for_storage,
                    last_policy_action_for_enabled,
                    policy_stationary_arm_delta_threshold,
                )
                pending_policy_release_baseline = False
            last_policy_action_for_enabled = raw_policy_action_for_storage
            act_processed_policy = (
                policy_action_processor((policy_action_for_execution, obs))
                if policy_action_processor is not None
                else raw_policy_action_for_storage
            )

        if isinstance(teleop, Teleoperator):
            act = run_with_connection_retry("teleop.get_action", teleop.get_action)
            raw_teleop_action = act

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))

        elif isinstance(teleop, list):
            arm_action = run_with_connection_retry("teleop_arm.get_action", teleop_arm.get_action)
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            raw_teleop_action = act
            act_processed_teleop = teleop_action_processor((act, obs))

        if act_processed_policy is None and act_processed_teleop is None:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue

        if act_processed_teleop is not None:
            last_teleop_action = act_processed_teleop
            teleop_fallback_warned = False

        policy_action_for_storage = raw_policy_action_for_storage or zero_policy_action

        is_intervention = 0.0
        if intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE:
            is_intervention = 1.0
            if act_processed_teleop is not None:
                action_values = act_processed_teleop
            elif last_teleop_action is not None:
                action_values = last_teleop_action
                if not teleop_fallback_warned:
                    logging.warning(
                        "Intervention is active but no fresh teleop action is available; reusing last teleop action."
                    )
                    teleop_fallback_warned = True
            elif act_processed_policy is not None:
                action_values = act_processed_policy
                if not teleop_fallback_warned:
                    logging.warning(
                        "Intervention is active but teleop action is unavailable; falling back to policy action."
                    )
                    teleop_fallback_warned = True
            else:
                action_values = zero_policy_action
                if not teleop_fallback_warned:
                    logging.warning(
                        "Intervention is active but no teleop/policy action is available; sending zero action."
                    )
                    teleop_fallback_warned = True
        else:
            action_values = act_processed_policy if act_processed_policy is not None else act_processed_teleop

        # Applies a pipeline to the action, default is IdentityProcessor
        robot_action_to_send = robot_action_processor((action_values, obs))

        # Send action to robot
        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset. action = postprocessor.process(action)
        # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
        selected_from_policy = act_processed_policy is not None and action_values is act_processed_policy
        if policy_sync_executor is not None and selected_from_policy:
            _sent_action = run_with_connection_retry(
                "policy_sync_executor.send_action",
                lambda robot_action_to_send=robot_action_to_send: policy_sync_executor.send_action(
                    robot_action_to_send
                ),
            )
        else:
            _sent_action = run_with_connection_retry(
                "robot.send_action",
                lambda robot_action_to_send=robot_action_to_send: robot.send_action(robot_action_to_send),
            )

        if handoff_debug_enabled and handoff_debug_frames_remaining > 0:
            selected_source = "policy" if selected_from_policy else "teleop"
            logging.info(
                "[HIL_HANDOFF] event=%d remaining=%d state=%.1f source=%s "
                "obs=%s policy_raw=%s policy_exec=%s policy_ik=%s "
                "teleop_raw=%s teleop_ik=%s final=%s sent=%s max_joint_delta_deg=%s",
                handoff_debug_event_idx,
                handoff_debug_frames_remaining,
                intervention_state,
                selected_source,
                _summarize_handoff_action(obs),
                _summarize_handoff_action(raw_policy_action_for_storage),
                _summarize_handoff_action(policy_action_for_execution),
                _summarize_handoff_action(act_processed_policy),
                _summarize_handoff_action(raw_teleop_action),
                _summarize_handoff_action(act_processed_teleop),
                _summarize_handoff_action(robot_action_to_send),
                _summarize_handoff_action(_sent_action),
                _handoff_joint_delta_summary(robot_action_to_send, obs),
            )
            handoff_debug_frames_remaining -= 1

        ee_action_after_action = (
            run_with_connection_retry("ee_pose_storage.read_action", ee_pose_storage.read)
            if dataset is not None and ee_pose_storage is not None
            else None
        )

        # Write to dataset
        if dataset is not None:
            action_values_for_storage = _complete_action_values_for_dataset(
                features_for_storage_completion,
                action_values,
                obs_processed,
                last_action_values_for_storage,
            )
            last_action_values_for_storage = action_values_for_storage

            if ee_pose_storage is not None:
                if ee_state_before_action is None or ee_action_after_action is None:
                    raise RuntimeError("EE pose storage is enabled but SDK pose feedback was not captured.")
                observation_values_for_storage = {**obs_processed, **ee_state_before_action}
                action_values_for_frame = ee_action_after_action
                if selected_from_policy and policy_action_processor is not None:
                    policy_action_values_for_frame = policy_action_for_storage
                elif selected_from_policy:
                    policy_action_values_for_frame = ee_action_after_action
                else:
                    policy_action_values_for_frame = _zero_values_for_feature(
                        dataset.features, "complementary_info.policy_action"
                    )
            else:
                observation_values_for_storage = obs_processed
                action_values_for_frame = action_values_for_storage
                policy_action_values_for_frame = policy_action_for_storage

            observation_frame = build_dataset_frame(
                dataset.features, observation_values_for_storage, prefix=OBS_STR
            )
            action_frame = build_dataset_frame(dataset.features, action_values_for_frame, prefix=ACTION)
            policy_action_frame = build_dataset_frame(
                dataset.features,
                policy_action_values_for_frame,
                prefix="complementary_info.policy_action",
            )
            frame = {**observation_frame, **action_frame, **policy_action_frame, "task": single_task}

            if "complementary_info.is_intervention" in dataset.features:
                frame["complementary_info.is_intervention"] = np.array([is_intervention], dtype=np.float32)
            if "complementary_info.state" in dataset.features:
                frame["complementary_info.state"] = np.array([intervention_state], dtype=np.float32)
            if "complementary_info.collector_policy_id" in dataset.features:
                frame["complementary_info.collector_policy_id"] = resolve_collector_policy_id(
                    intervention_enabled=intervention_enabled,
                    is_intervention=bool(is_intervention),
                    selected_from_policy=selected_from_policy,
                    policy_id=collector_policy_id_policy,
                    human_id=collector_policy_id_human,
                )
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(
                observation=obs_processed, action=action_values, compress_images=display_compressed_images
            )

        if intervention_state == INTERVENTION_STATE_RELEASE:
            intervention_state = INTERVENTION_STATE_POLICY

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(1 / fps - dt_s, 0.0))

        timestamp = time.perf_counter() - start_episode_t
