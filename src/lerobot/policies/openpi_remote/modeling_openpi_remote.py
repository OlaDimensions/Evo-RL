#!/usr/bin/env python

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.openpi_remote.configuration_openpi_remote import OpenPIRemoteConfig
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


class OpenPIRemotePolicy(PreTrainedPolicy):
    """Thin LeRobot policy wrapper around an OpenPI WebSocket policy server."""

    config_class = OpenPIRemoteConfig
    name = "openpi_remote"

    def __init__(self, config: OpenPIRemoteConfig, **kwargs):
        super().__init__(config)
        self.config = config
        self.reset()
        self._client = None

    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_name_or_path: str,
        *,
        config: PreTrainedConfig | None = None,
        **kwargs,
    ) -> T:
        if config is None:
            raise ValueError("`openpi_remote` does not load local weights; pass a config instead.")
        return cls(config=config, **kwargs)

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from openpi_client import websocket_client_policy
        except ImportError as exc:
            raise ImportError(
                "`policy.type=openpi_remote` requires the OpenPI Python client in the Evo-RL runtime. "
                "Install it or add `/home/ola/code/openpi/packages/openpi-client/src` to PYTHONPATH."
            ) from exc

        self._client = websocket_client_policy.WebsocketClientPolicy(
            host=self.config.host,
            port=self.config.port,
            api_key=self.config.api_key,
        )
        return self._client

    def get_optim_params(self) -> dict:
        return {}

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        client = getattr(self, "_client", None)
        if client is not None and hasattr(client, "reset"):
            client.reset()

    def _map_input_key(self, key: str) -> str:
        if key in self.config.input_map:
            return self.config.input_map[key]
        if key == OBS_STATE:
            return self.config.state_key
        if key == "task":
            return self.config.prompt_key
        if key.startswith(f"{OBS_IMAGES}."):
            camera_name = key.removeprefix(f"{OBS_IMAGES}.")
            return f"observation/{camera_name}"
        return key

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
                    raise ValueError(f"OpenPI remote inference expects batch size 1, got image {image.shape}.")
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
                raise ValueError(f"OpenPI remote inference expects batch size 1, got image {image_np.shape}.")
            image_np = image_np[0]
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

    def _to_openpi_observation(self, batch: dict[str, Any]) -> dict[str, Any]:
        observation: dict[str, Any] = {}
        for key, value in batch.items():
            if key == "robot_type":
                continue
            output_key = self._map_input_key(key)
            if key.startswith(f"{OBS_IMAGES}."):
                observation[output_key] = self._prepare_image(value)
            elif key == "task":
                observation[output_key] = value if value is not None else ""
            else:
                observation[output_key] = self._tensor_to_numpy(value)
        return observation

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        client = self._get_client()
        openpi_observation = self._to_openpi_observation(batch)
        result = client.infer(openpi_observation)
        if self.config.action_key not in result:
            raise KeyError(
                f"OpenPI server response is missing action key {self.config.action_key!r}; "
                f"available keys: {sorted(result.keys())}."
            )
        actions = torch.as_tensor(result[self.config.action_key], dtype=torch.float32)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.ndim != 3:
            raise ValueError(f"Expected OpenPI actions with shape (H, D) or (B, H, D), got {actions.shape}.")
        if actions.shape[0] != 1:
            raise ValueError(f"OpenPI remote policy currently supports batch size 1, got {actions.shape[0]}.")

        expected_dim = self.config.output_features[ACTION].shape[0]
        if actions.shape[-1] < expected_dim:
            raise ValueError(f"OpenPI returned action_dim={actions.shape[-1]}, expected at least {expected_dim}.")
        return actions[:, : self.config.chunk_size, :expected_dim]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("`openpi_remote` only supports inference via `select_action`.")
