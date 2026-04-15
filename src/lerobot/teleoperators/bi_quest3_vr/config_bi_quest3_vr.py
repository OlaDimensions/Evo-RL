#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.teleoperators.quest3_vr.config_quest3_vr import Quest3VRTeleopConfig

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_quest3_vr")
@dataclass
class BiQuest3VRTeleopConfig(Quest3VRTeleopConfig):
    """Configuration for bimanual Quest3 VR teleoperation."""

    left_enable_button: str = "Y"
    left_reset_button: str = "X"
    left_gripper_button: str = "leftTrig"
