#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@PreTrainedConfig.register_subclass("openpi_remote")
@dataclass
class OpenPIRemoteConfig(PreTrainedConfig):
    """Configuration for calling an OpenPI WebSocket policy server from LeRobot."""

    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None

    chunk_size: int = 50
    n_action_steps: int = 50
    action_dim: int = 14

    image_size: int = 224
    resize_images: bool = True
    action_key: str = "actions"
    state_key: str = "observation/state"
    prompt_key: str = "prompt"

    # Maps Evo-RL/LeRobot batch keys to OpenPI observation keys.
    #input_map: dict[str, str] = field(
    #    default_factory=lambda: {
    #        OBS_STATE: "observation/state",
    #        "task": "prompt",
    #        f"{OBS_IMAGES}.right_front": "observation/image",
    #        f"{OBS_IMAGES}.right_wrist": "observation/wrist_image",
    #        f"{OBS_IMAGES}.left_wrist": "observation/left_wrist_image",
    #    }
    #)

    input_map: dict[str, str] = field(
        default_factory=lambda: {
            OBS_STATE: "state",
            "task": "prompt",
            f"{OBS_IMAGES}.right_front": "front_camera",
            f"{OBS_IMAGES}.right_wrist": "right_wrist_camera",
            f"{OBS_IMAGES}.left_wrist": "left_wrist_camera",
        }
    )


    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"`port` must be between 1 and 65535, got {self.port}.")
        if self.chunk_size <= 0:
            raise ValueError(f"`chunk_size` must be positive, got {self.chunk_size}.")
        if self.n_action_steps <= 0 or self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` must be in [1, chunk_size], got {self.n_action_steps}."
            )
        if self.action_dim <= 0:
            raise ValueError(f"`action_dim` must be positive, got {self.action_dim}.")
        if self.image_size <= 0:
            raise ValueError(f"`image_size` must be positive, got {self.image_size}.")
        self.validate_features()

    def validate_features(self) -> None:
        if self.output_features is None:
            self.output_features = {}
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.action_dim,),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig()

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
