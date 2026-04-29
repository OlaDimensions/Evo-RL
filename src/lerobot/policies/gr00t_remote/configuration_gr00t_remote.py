#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@PreTrainedConfig.register_subclass("gr00t_remote")
@dataclass
class Gr00TRemoteConfig(PreTrainedConfig):
    """Configuration for calling an Isaac-GR00T ZMQ policy server from LeRobot."""

    host: str = "127.0.0.1"
    port: int = 5555
    timeout_ms: int = 15000
    api_token: str | None = None
    gr00t_root: str | None = "/home/ola/code/Isaac-GR00T"

    chunk_size: int = 16
    n_action_steps: int = 8
    action_dim: int = 14

    # GR00T policies expose action streams by modality key. If empty, the wrapper
    # uses the server's action modality order, falling back to the response order.
    action_keys: list[str] = field(default_factory=list)

    # Maps LeRobot image batch keys to GR00T video modality keys. The wrapper also
    # has conservative fallbacks for front/right_wrist/left_wrist names.
    video_input_map: dict[str, str] = field(
        default_factory=lambda: {
            f"{OBS_IMAGES}.right_front": "front",
            f"{OBS_IMAGES}.right_wrist": "right_wrist",
            f"{OBS_IMAGES}.left_wrist": "left_wrist",
        }
    )

    # Maps LeRobot low-dimensional state keys to GR00T state modality keys when
    # the GR00T checkpoint expects a single state stream.
    state_input_map: dict[str, str] = field(default_factory=lambda: {OBS_STATE: "state"})

    # Optional slices for splitting observation.state into GR00T state streams.
    # Format is {gr00t_state_key: [start, end]}. If empty, common bimanual EE-RPY
    # stream names are inferred automatically when the server reports them.
    state_slices: dict[str, list[int]] = field(default_factory=dict)

    # Maps the LeRobot task field to the GR00T language modality key. If empty or
    # missing the server key, "task" is used as the source for all language keys.
    language_input_map: dict[str, str] = field(default_factory=lambda: {"task": "annotation.human.task_description"})

    image_size: int = 224
    resize_images: bool = False

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
        if self.timeout_ms <= 0:
            raise ValueError(f"`timeout_ms` must be positive, got {self.timeout_ms}.")
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
