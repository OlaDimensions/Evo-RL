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
from collections.abc import Callable
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
from lerobot.scripts.ee_pose_action_utils import with_bimanual_ee_enabled_flags
from lerobot.teleoperators import Teleoperator, koch_leader, omx_leader, so_leader
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.recording_annotations import resolve_collector_policy_id
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
        raw_policy_action_for_storage: RobotAction | None = None
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
            policy_action_for_execution = with_bimanual_ee_enabled_flags(
                raw_policy_action_for_storage,
                last_policy_action_for_enabled,
                policy_stationary_arm_delta_threshold,
            )
            last_policy_action_for_enabled = raw_policy_action_for_storage
            act_processed_policy = (
                policy_action_processor((policy_action_for_execution, obs))
                if policy_action_processor is not None
                else raw_policy_action_for_storage
            )

        if isinstance(teleop, Teleoperator):
            act = run_with_connection_retry("teleop.get_action", teleop.get_action)

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))

        elif isinstance(teleop, list):
            arm_action = run_with_connection_retry("teleop_arm.get_action", teleop_arm.get_action)
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
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
                    policy_action_values_for_frame = ee_pose_storage.zero()
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
