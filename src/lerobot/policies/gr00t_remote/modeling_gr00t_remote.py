#!/usr/bin/env python

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import io
from types import SimpleNamespace
from typing import Any

import msgpack
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import zmq
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.gr00t_remote.configuration_gr00t_remote import Gr00TRemoteConfig
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


_BIMANUAL_EE_RPY_SLICES = {
    "left_ee": (0, 6),
    "left_eef": (0, 6),
    "left_arm": (0, 6),
    "left_gripper": (6, 7),
    "right_ee": (7, 13),
    "right_eef": (7, 13),
    "right_arm": (7, 13),
    "right_gripper": (13, 14),
}

_VIDEO_SOURCE_ALIASES = {
    "front": (
        f"{OBS_IMAGES}.right_front",
        f"{OBS_IMAGES}.front",
        f"{OBS_IMAGES}.image",
    ),
    "front_camera": (
        f"{OBS_IMAGES}.right_front",
        f"{OBS_IMAGES}.front",
        f"{OBS_IMAGES}.image",
    ),
    "image": (
        f"{OBS_IMAGES}.right_front",
        f"{OBS_IMAGES}.front",
        f"{OBS_IMAGES}.image",
    ),
    "right_wrist": (f"{OBS_IMAGES}.right_wrist",),
    "right_wrist_camera": (f"{OBS_IMAGES}.right_wrist",),
    "wrist": (
        f"{OBS_IMAGES}.right_wrist",
        f"{OBS_IMAGES}.wrist",
        f"{OBS_IMAGES}.left_wrist",
    ),
    "wrist_image": (
        f"{OBS_IMAGES}.right_wrist",
        f"{OBS_IMAGES}.wrist_image",
        f"{OBS_IMAGES}.wrist",
    ),
    "left_wrist": (f"{OBS_IMAGES}.left_wrist",),
    "left_wrist_camera": (f"{OBS_IMAGES}.left_wrist",),
}


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


class _Gr00TMsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=_Gr00TMsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=_Gr00TMsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj: Any) -> Any:
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            return _to_namespace(obj["as_json"])
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


class _Gr00TPolicyClient:
    """Small GR00T ZMQ client that avoids importing the full Isaac-GR00T policy package."""

    def __init__(self, host: str, port: int, timeout_ms: int, api_token: str | None):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.context = zmq.Context()
        self._init_socket()

    def _init_socket(self) -> None:
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(
        self, endpoint: str, data: dict[str, Any] | None = None, requires_input: bool = True
    ) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        try:
            self.socket.send(_Gr00TMsgSerializer.to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again:
            self.socket.close(linger=0)
            self._init_socket()
            raise

        response = _Gr00TMsgSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint("get_action", {"observation": observation, "options": None})
        return tuple(response)

    def get_modality_config(self) -> dict[str, Any]:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def reset(self) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": None})


class Gr00TRemotePolicy(PreTrainedPolicy):
    """Thin LeRobot policy wrapper around an Isaac-GR00T ZMQ policy server."""

    config_class = Gr00TRemoteConfig
    name = "gr00t_remote"

    def __init__(self, config: Gr00TRemoteConfig, **kwargs):
        super().__init__(config)
        self.config = config
        self._client = None
        self._modality_config: dict[str, Any] | None = None
        self._video_history: dict[str, deque[np.ndarray]] = {}
        self._state_history: dict[str, deque[np.ndarray]] = {}
        self.reset()

    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_name_or_path: str,
        *,
        config: PreTrainedConfig | None = None,
        **kwargs,
    ) -> T:
        if config is None:
            raise ValueError("`gr00t_remote` does not load local weights; pass a config instead.")
        return cls(config=config, **kwargs)

    def _get_client(self):
        if self._client is not None:
            return self._client

        self._client = _Gr00TPolicyClient(
            host=self.config.host,
            port=self.config.port,
            timeout_ms=self.config.timeout_ms,
            api_token=self.config.api_token,
        )
        return self._client

    def _get_modality_config(self) -> dict[str, Any]:
        if self._modality_config is None:
            client = self._get_client()
            modality_config = client.get_modality_config()
            if not isinstance(modality_config, Mapping):
                raise TypeError(
                    f"GR00T server `get_modality_config` returned {type(modality_config)}, expected a mapping."
                )
            missing = [
                modality
                for modality in ("video", "state", "action", "language")
                if modality not in modality_config or not self._modality_keys(modality_config, modality)
            ]
            if missing:
                raise ValueError(
                    "GR00T server modality config is missing required modality keys "
                    f"{missing}. Check the server `--embodiment-tag` and checkpoint."
                )
            self._modality_config = dict(modality_config)
        return self._modality_config

    def get_optim_params(self) -> dict:
        return {}

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._video_history.clear() if hasattr(self, "_video_history") else None
        self._state_history.clear() if hasattr(self, "_state_history") else None
        client = getattr(self, "_client", None)
        if client is not None and hasattr(client, "reset"):
            client.reset()

    @staticmethod
    def _modality_keys(modality_config: dict[str, Any], modality: str) -> list[str]:
        cfg = modality_config.get(modality)
        keys = getattr(cfg, "modality_keys", None)
        return list(keys or [])

    @staticmethod
    def _modality_horizon(modality_config: dict[str, Any], modality: str, default: int = 1) -> int:
        cfg = modality_config.get(modality)
        delta_indices = getattr(cfg, "delta_indices", None)
        return max(1, len(delta_indices or [])) if cfg is not None else default

    def _tensor_to_numpy(self, value: Any) -> Any:
        if not isinstance(value, torch.Tensor):
            return value
        value = value.detach().cpu()
        if value.ndim > 0 and value.shape[0] == 1:
            value = value.squeeze(0)
        return value.numpy()

    def _prepare_image(self, value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            image = value.detach().cpu()
            if image.ndim == 4:
                if image.shape[0] != 1:
                    raise ValueError(f"GR00T remote inference expects batch size 1, got image {image.shape}.")
                image = image.squeeze(0)
            if image.ndim != 3:
                raise ValueError(f"Expected image tensor with 3 dims after squeeze, got {image.shape}.")
            if image.shape[0] in (1, 3, 4):
                image = image[:3]
            else:
                image = image.permute(2, 0, 1)[:3]
            image = image.to(torch.float32)
            if image.max() > 1.5:
                image = image / 255.0
            if self.config.resize_images:
                image = self._resize_with_pad(image, self.config.image_size, self.config.image_size)
            image = image.clamp(0.0, 1.0)
            return (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

        image_np = np.asarray(value)
        if image_np.ndim == 4:
            if image_np.shape[0] != 1:
                raise ValueError(f"GR00T remote inference expects batch size 1, got image {image_np.shape}.")
            image_np = image_np[0]
        if image_np.ndim == 3 and image_np.shape[0] in (1, 3, 4):
            image_np = np.moveaxis(image_np[:3], 0, -1)
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0.0, 1.0) * 255.0 if image_np.max() <= 1.5 else image_np
            image_np = np.clip(image_np, 0, 255).round().astype(np.uint8)
        return image_np

    @staticmethod
    def _resize_with_pad(image: Tensor, height: int, width: int) -> Tensor:
        _, cur_height, cur_width = image.shape
        ratio = max(cur_width / width, cur_height / height)
        resized_height = max(1, int(cur_height / ratio))
        resized_width = max(1, int(cur_width / ratio))
        image = F.interpolate(
            image.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        pad_h0, remainder_h = divmod(height - resized_height, 2)
        pad_h1 = pad_h0 + remainder_h
        pad_w0, remainder_w = divmod(width - resized_width, 2)
        pad_w1 = pad_w0 + remainder_w
        return F.pad(image, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=0.0)

    @staticmethod
    def _append_history(history: dict[str, deque[np.ndarray]], key: str, value: np.ndarray, horizon: int) -> np.ndarray:
        queue = history.get(key)
        if queue is None or queue.maxlen != horizon:
            queue = deque(maxlen=horizon)
            history[key] = queue
        queue.append(value)
        while len(queue) < horizon:
            queue.appendleft(value)
        return np.stack(list(queue), axis=0)

    def _source_for_video_key(self, batch: Mapping[str, Any], video_key: str) -> str:
        inverse = {target: source for source, target in self.config.video_input_map.items()}
        candidates = []
        if video_key in inverse:
            candidates.append(inverse[video_key])
        candidates.extend(_VIDEO_SOURCE_ALIASES.get(video_key, ()))
        candidates.append(f"{OBS_IMAGES}.{video_key}")

        for candidate in candidates:
            if candidate in batch:
                return candidate
        available = sorted(key for key in batch if key.startswith(f"{OBS_IMAGES}."))
        raise KeyError(
            f"Could not map GR00T video key {video_key!r} to a LeRobot image. "
            f"Available image keys: {available}. Set `--policy.video_input_map` if needed."
        )

    def _state_array(self, batch: Mapping[str, Any]) -> np.ndarray:
        if OBS_STATE not in batch:
            raise KeyError(f"Missing required state key {OBS_STATE!r} in policy batch.")
        state = np.asarray(self._tensor_to_numpy(batch[OBS_STATE]), dtype=np.float32)
        if state.ndim != 1:
            state = state.reshape(-1)
        return state

    def _state_slice_for_key(self, state_key: str, state_dim: int) -> tuple[int, int] | None:
        configured = self.config.state_slices.get(state_key)
        if configured is not None:
            if len(configured) != 2:
                raise ValueError(f"`state_slices[{state_key}]` must be [start, end], got {configured}.")
            return int(configured[0]), int(configured[1])

        if state_dim == 14 and state_key in _BIMANUAL_EE_RPY_SLICES:
            return _BIMANUAL_EE_RPY_SLICES[state_key]
        return None

    def _make_video_observation(self, batch: Mapping[str, Any], modality_config: dict[str, Any]) -> dict[str, np.ndarray]:
        video_keys = self._modality_keys(modality_config, "video")
        if not video_keys:
            video_keys = list(dict.fromkeys(self.config.video_input_map.values()))
        horizon = self._modality_horizon(modality_config, "video")

        video: dict[str, np.ndarray] = {}
        for video_key in video_keys:
            source_key = self._source_for_video_key(batch, video_key)
            frame = self._prepare_image(batch[source_key])
            video[video_key] = self._append_history(self._video_history, video_key, frame, horizon)[
                np.newaxis, ...
            ]
        return video

    def _make_state_observation(self, batch: Mapping[str, Any], modality_config: dict[str, Any]) -> dict[str, np.ndarray]:
        state = self._state_array(batch)
        state_keys = self._modality_keys(modality_config, "state")
        if not state_keys:
            state_keys = list(dict.fromkeys(self.config.state_input_map.values()))
        horizon = self._modality_horizon(modality_config, "state")

        state_obs: dict[str, np.ndarray] = {}
        for state_key in state_keys:
            state_slice = self._state_slice_for_key(state_key, state.shape[0])
            if state_slice is None:
                if len(state_keys) > 1:
                    raise ValueError(
                        f"GR00T server expects multiple state streams {state_keys}, but no slice is known for "
                        f"{state_key!r}. Set `--policy.state_slices` with [start, end] per stream."
                    )
                value = state
            else:
                start, end = state_slice
                value = state[start:end]
            state_obs[state_key] = self._append_history(
                self._state_history, state_key, np.asarray(value, dtype=np.float32), horizon
            )[np.newaxis, ...]
        return state_obs

    def _make_language_observation(self, batch: Mapping[str, Any], modality_config: dict[str, Any]) -> dict[str, list[list[str]]]:
        language_keys = self._modality_keys(modality_config, "language")
        if not language_keys:
            language_keys = list(dict.fromkeys(self.config.language_input_map.values())) or ["task"]
        horizon = self._modality_horizon(modality_config, "language")

        language: dict[str, list[list[str]]] = {}
        for language_key in language_keys:
            source_key = next(
                (source for source, target in self.config.language_input_map.items() if target == language_key),
                "task",
            )
            task = batch.get(source_key, batch.get("task", ""))
            if isinstance(task, (list, tuple)):
                task = task[0] if task else ""
            language[language_key] = [[str(task)] * horizon]
        return language

    def _to_gr00t_observation(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        modality_config = self._get_modality_config()
        return {
            "video": self._make_video_observation(batch, modality_config),
            "state": self._make_state_observation(batch, modality_config),
            "language": self._make_language_observation(batch, modality_config),
        }

    def _action_key_order(self, actions: Mapping[str, Any]) -> list[str]:
        if self.config.action_keys:
            return list(self.config.action_keys)
        modality_keys = self._modality_keys(self._get_modality_config(), "action")
        return [key for key in modality_keys if key in actions] or list(actions.keys())

    def _actions_to_tensor(self, actions: Mapping[str, Any]) -> Tensor:
        chunks: list[Tensor] = []
        missing: list[str] = []
        for key in self._action_key_order(actions):
            if key not in actions:
                missing.append(key)
                continue
            chunk = torch.as_tensor(actions[key], dtype=torch.float32)
            if chunk.ndim == 2:
                chunk = chunk.unsqueeze(0)
            if chunk.ndim != 3:
                raise ValueError(f"Expected GR00T action {key!r} with shape (H, D) or (B, H, D), got {chunk.shape}.")
            chunks.append(chunk)

        if missing:
            raise KeyError(f"GR00T response is missing action keys {missing}; available keys: {sorted(actions.keys())}.")
        if not chunks:
            raise KeyError(f"GR00T response has no action streams; response keys: {sorted(actions.keys())}.")

        batch_sizes = {chunk.shape[0] for chunk in chunks}
        horizons = {chunk.shape[1] for chunk in chunks}
        if batch_sizes != {1}:
            raise ValueError(f"GR00T remote policy currently supports batch size 1, got {sorted(batch_sizes)}.")
        if len(horizons) != 1:
            raise ValueError(f"GR00T action streams have inconsistent horizons: {sorted(horizons)}.")
        return torch.cat(chunks, dim=-1)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        client = self._get_client()
        observation = self._to_gr00t_observation(batch)
        actions, _info = client.get_action(observation)
        if not isinstance(actions, Mapping):
            raise TypeError(f"Expected GR00T actions to be a dict, got {type(actions)}.")

        action_tensor = self._actions_to_tensor(actions)
        expected_dim = self.config.output_features[ACTION].shape[0]
        if action_tensor.shape[-1] != expected_dim:
            raise ValueError(f"GR00T returned action_dim={action_tensor.shape[-1]}, expected {expected_dim}.")
        return action_tensor[:, : self.config.chunk_size]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("`gr00t_remote` only supports inference via `select_action`.")
