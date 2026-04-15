#!/usr/bin/env python

from __future__ import annotations

from functools import cached_property

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from lerobot.teleoperators.quest3_vr.quest3_vr import ArmRuntime, Quest3VRTeleop

from .config_bi_quest3_vr import BiQuest3VRTeleopConfig


class BiQuest3VRTeleop(Quest3VRTeleop):
    """Bimanual Quest3 VR teleoperator.

    Keeps reader/frame math in single-arm module and only adds dual registration
    and prefixed action multiplexing.
    """

    config_class = BiQuest3VRTeleopConfig
    name = "bi_quest3_vr"

    def __init__(self, config: BiQuest3VRTeleopConfig):
        super().__init__(config)
        self.config = config
        self._left = ArmRuntime()
        self._left.arm_T = self._arm_init_T.copy()
        self._left.last_T = self._arm_init_T.copy()

    @cached_property
    def action_features(self) -> dict[str, type]:
        base = super().action_features
        return {
            **{f"left_{k}": v for k, v in base.items()},
            **{f"right_{k}": v for k, v in base.items()},
        }

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        transforms, buttons = self._reader.get_transformations_and_buttons()
        right_T = self._extract_transform(transforms, "r")
        left_T = self._extract_transform(transforms, "l")

        right_action = self._arm_action(
            controller_T=right_T,
            buttons=buttons,
            state=self._right,
            enable_button=self.config.enable_button,
            reset_button=self.config.reset_button,
            gripper_button=self.config.gripper_button,
        )
        left_action = self._arm_action(
            controller_T=left_T,
            buttons=buttons,
            state=self._left,
            enable_button=self.config.left_enable_button,
            reset_button=self.config.left_reset_button,
            gripper_button=self.config.left_gripper_button,
        )
        return {
            **{f"left_{k}": v for k, v in left_action.items()},
            **{f"right_{k}": v for k, v in right_action.items()},
        }
