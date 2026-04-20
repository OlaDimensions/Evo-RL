# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Replays the actions of an episode from a dataset on a robot.

Examples:

```shell
lerobot-replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=aliberts/record-test \
    --dataset.episode=0
```

Example replay with bimanual so100:
```shell
lerobot-replay \
  --robot.type=bi_so_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --dataset.repo_id=${HF_USER}/bimanual-so100-handover-cube \
  --dataset.episode=0
```

"""

import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import numpy as np

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import (
    RobotAction,
    RobotProcessorPipeline,
    make_default_robot_action_processor,
)
from lerobot.scripts.ee_pose_action_utils import (
    BIMANUAL_EE_RPY_NAMES,
    BIMANUAL_EE_RXRYRZ_NAMES,
    SDK_EE_OFFSET_XYZRPY,
    infer_bimanual_ee_arm_enabled,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_piper_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    piper_follower,
    reachy2,
    so_follower,
    unitree_g1,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)

ACTION_MODE_AUTO = "auto"
ACTION_MODE_JOINT = "joint"
ACTION_MODE_BIMANUAL_EE_RPY = "bimanual_ee_rpy"
ACTION_MODE_BIMANUAL_EE_RXRYRZ = "bimanual_ee_rxryrz"

@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # How to interpret dataset action vectors before sending them to the robot.
    action_mode: str = ACTION_MODE_AUTO
    # Run action decoding and processing without sending commands to the robot.
    dry_run: bool = False
    # Optional cap on replayed frames, useful for hardware bring-up.
    max_frames: int | None = None
    # Treat a bimanual EE-pose arm as stationary when its per-frame action delta is at or below this.
    stationary_arm_delta_threshold: float = 1e-5

    def __post_init__(self):
        valid_modes = {ACTION_MODE_AUTO, ACTION_MODE_JOINT, ACTION_MODE_BIMANUAL_EE_RPY, ACTION_MODE_BIMANUAL_EE_RXRYRZ}
        if self.action_mode not in valid_modes:
            raise ValueError(f"`action_mode` must be one of {sorted(valid_modes)}, got {self.action_mode!r}.")
        if self.max_frames is not None and self.max_frames < 0:
            raise ValueError("`max_frames` must be >= 0 when set.")
        if self.stationary_arm_delta_threshold < 0:
            raise ValueError("`stationary_arm_delta_threshold` must be >= 0.")


def is_bimanual_ee_rpy_action_schema(names: list[str] | tuple[str, ...]) -> bool:
    return tuple(names) == BIMANUAL_EE_RPY_NAMES


def is_bimanual_ee_rxryrz_action_schema(names: list[str] | tuple[str, ...]) -> bool:
    return tuple(names) == BIMANUAL_EE_RXRYRZ_NAMES


def resolve_action_mode(action_mode: str, action_names: list[str] | tuple[str, ...]) -> str:
    if action_mode != ACTION_MODE_AUTO:
        return action_mode
    if is_bimanual_ee_rpy_action_schema(action_names):
        return ACTION_MODE_BIMANUAL_EE_RPY
    if is_bimanual_ee_rxryrz_action_schema(action_names):
        return ACTION_MODE_BIMANUAL_EE_RXRYRZ
    return ACTION_MODE_JOINT


def validate_action_schema_for_mode(action_mode: str, action_names: list[str] | tuple[str, ...]) -> None:
    if action_mode == ACTION_MODE_BIMANUAL_EE_RPY:
        required_names = BIMANUAL_EE_RPY_NAMES
    elif action_mode == ACTION_MODE_BIMANUAL_EE_RXRYRZ:
        required_names = BIMANUAL_EE_RXRYRZ_NAMES
    else:
        return
    missing = [name for name in required_names if name not in action_names]
    if missing:
        raise ValueError(
            f"`{action_mode}` replay requires action names {list(required_names)}, missing {missing}."
        )


def adapt_bimanual_ee_rpy_action(
    action: RobotAction, *, left_enabled: bool = True, right_enabled: bool = True
) -> RobotAction:
    """Map dataset `left/right_ee.xyz+roll/pitch/yaw` fields to the bimanual EE IK target format."""
    return {
        "left_enabled": left_enabled,
        "left_reset": False,
        "left_ee.target_x": float(action["left_ee.x"]),
        "left_ee.target_y": float(action["left_ee.y"]),
        "left_ee.target_z": float(action["left_ee.z"]),
        "left_ee.target_rx": float(action["left_ee.roll"]),
        "left_ee.target_ry": float(action["left_ee.pitch"]),
        "left_ee.target_rz": float(action["left_ee.yaw"]),
        "left_gripper.pos": float(action["left_gripper.pos"]),
        "right_enabled": right_enabled,
        "right_reset": False,
        "right_ee.target_x": float(action["right_ee.x"]),
        "right_ee.target_y": float(action["right_ee.y"]),
        "right_ee.target_z": float(action["right_ee.z"]),
        "right_ee.target_rx": float(action["right_ee.roll"]),
        "right_ee.target_ry": float(action["right_ee.pitch"]),
        "right_ee.target_rz": float(action["right_ee.yaw"]),
        "right_gripper.pos": float(action["right_gripper.pos"]),
    }


def rxryrz_to_rpy(rx: float, ry: float, rz: float) -> tuple[float, float, float]:
    """Convert XYZ Euler `R = Rx(rx) * Ry(ry) * Rz(rz)` to IK RPY."""
    ca, sa = math.cos(rx), math.sin(rx)
    cb, sb = math.cos(ry), math.sin(ry)
    cc, sc = math.cos(rz), math.sin(rz)
    matrix = np.array(
        [
            [cb * cc, -cb * sc, sb],
            [sa * sb * cc + ca * sc, -sa * sb * sc + ca * cc, -sa * cb],
            [-ca * sb * cc + sa * sc, ca * sb * sc + sa * cc, ca * cb],
        ],
        dtype=np.float64,
    )
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return roll, pitch, yaw


def adapt_bimanual_ee_rxryrz_action(
    action: RobotAction, *, left_enabled: bool = True, right_enabled: bool = True
) -> RobotAction:
    """Map dataset XYZ Euler fields to the RPY fields expected by the existing IK processor."""
    left_roll, left_pitch, left_yaw = rxryrz_to_rpy(
        float(action["left_ee.rx"]),
        float(action["left_ee.ry"]),
        float(action["left_ee.rz"]),
    )
    right_roll, right_pitch, right_yaw = rxryrz_to_rpy(
        float(action["right_ee.rx"]),
        float(action["right_ee.ry"]),
        float(action["right_ee.rz"]),
    )
    return {
        "left_enabled": left_enabled,
        "left_reset": False,
        "left_ee.target_x": float(action["left_ee.x"]),
        "left_ee.target_y": float(action["left_ee.y"]),
        "left_ee.target_z": float(action["left_ee.z"]),
        "left_ee.target_rx": left_roll,
        "left_ee.target_ry": left_pitch,
        "left_ee.target_rz": left_yaw,
        "left_gripper.pos": float(action["left_gripper.pos"]),
        "right_enabled": right_enabled,
        "right_reset": False,
        "right_ee.target_x": float(action["right_ee.x"]),
        "right_ee.target_y": float(action["right_ee.y"]),
        "right_ee.target_z": float(action["right_ee.z"]),
        "right_ee.target_rx": right_roll,
        "right_ee.target_ry": right_pitch,
        "right_ee.target_rz": right_yaw,
        "right_gripper.pos": float(action["right_gripper.pos"]),
    }


def make_bimanual_ee_rpy_action_processor() -> RobotProcessorPipeline:
    # Import lazily so joint-space replay does not require the Piper IK stack.
    from lerobot.teleoperators.bi_quest3_vr.config_bi_quest3_vr import BiQuest3VRTeleopConfig
    from lerobot.teleoperators.quest3_vr.processors import make_bi_quest3_vr_robot_action_processor_from_config

    return make_bi_quest3_vr_robot_action_processor_from_config(
        BiQuest3VRTeleopConfig(piper_ee_offset_xyzrpy=SDK_EE_OFFSET_XYZRPY)
    )


def _dataset_action_to_dict(action_array, action_names: list[str]) -> RobotAction:
    return {name: action_array[i] for i, name in enumerate(action_names)}


def _bimanual_ee_arm_enabled(
    action: RobotAction,
    previous_action: RobotAction | None,
    action_mode: str,
    stationary_arm_delta_threshold: float,
) -> tuple[bool, bool]:
    if action_mode == ACTION_MODE_BIMANUAL_EE_RPY:
        names = BIMANUAL_EE_RPY_NAMES
    elif action_mode == ACTION_MODE_BIMANUAL_EE_RXRYRZ:
        names = BIMANUAL_EE_RXRYRZ_NAMES
    else:
        return True, True
    return infer_bimanual_ee_arm_enabled(action, previous_action, names, stationary_arm_delta_threshold)


def _prepare_action_for_mode(
    action: RobotAction,
    action_mode: str,
    previous_action: RobotAction | None = None,
    stationary_arm_delta_threshold: float = 1e-5,
) -> RobotAction:
    left_enabled, right_enabled = _bimanual_ee_arm_enabled(
        action, previous_action, action_mode, stationary_arm_delta_threshold
    )
    if action_mode == ACTION_MODE_BIMANUAL_EE_RPY:
        return adapt_bimanual_ee_rpy_action(
            action, left_enabled=left_enabled, right_enabled=right_enabled
        )
    if action_mode == ACTION_MODE_BIMANUAL_EE_RXRYRZ:
        return adapt_bimanual_ee_rxryrz_action(
            action, left_enabled=left_enabled, right_enabled=right_enabled
        )
    return action


def _make_action_processor_for_mode(action_mode: str) -> RobotProcessorPipeline:
    if action_mode in {ACTION_MODE_BIMANUAL_EE_RPY, ACTION_MODE_BIMANUAL_EE_RXRYRZ}:
        return make_bimanual_ee_rpy_action_processor()
    return make_default_robot_action_processor()


@parser.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])
    action_names = list(dataset.features[ACTION]["names"])
    action_mode = resolve_action_mode(cfg.action_mode, action_names)
    validate_action_schema_for_mode(action_mode, action_names)
    robot_action_processor = _make_action_processor_for_mode(action_mode)
    logging.info("Resolved replay action mode: %s", action_mode)

    # Filter dataset to only include frames from the specified episode since episodes are chunked in dataset V3.0
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.dataset.episode)
    actions = episode_frames.select_columns(ACTION)
    num_frames = len(episode_frames) if cfg.max_frames is None else min(len(episode_frames), cfg.max_frames)

    robot.connect()

    try:
        event = "Dry-running replay episode" if cfg.dry_run else "Replaying episode"
        log_say(event, cfg.play_sounds, blocking=True)
        previous_action = None
        for idx in range(num_frames):
            start_episode_t = time.perf_counter()

            action_array = actions[idx][ACTION]
            action = _dataset_action_to_dict(action_array, action_names)
            prepared_action = _prepare_action_for_mode(
                action,
                action_mode,
                previous_action=previous_action,
                stationary_arm_delta_threshold=cfg.stationary_arm_delta_threshold,
            )
            previous_action = action

            robot_obs = robot.get_observation()

            processed_action = robot_action_processor((prepared_action, robot_obs))

            if cfg.dry_run:
                logging.info("Dry-run frame %d processed action: %s", idx, pformat(processed_action))
            else:
                _ = robot.send_action(processed_action)

            dt_s = time.perf_counter() - start_episode_t
            precise_sleep(max(1 / dataset.fps - dt_s, 0.0))
    finally:
        robot.disconnect()


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
