from dataclasses import dataclass
from pathlib import Path

from ..configs import CameraConfig, ColorMode, Cv2Backends, Cv2Rotation

__all__ = ["GoProCameraConfig", "ColorMode", "Cv2Rotation", "Cv2Backends"]


@CameraConfig.register_subclass("gopro")
@dataclass
class GoProCameraConfig(CameraConfig):
    """OpenCV-backed GoPro capture with UMI replay-buffer-style image resizing."""

    index_or_path: int | Path
    capture_width: int | None = None
    capture_height: int | None = None
    crop_ratio: float = 1.0
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    fourcc: str | None = None
    backend: Cv2Backends = Cv2Backends.ANY

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)
        self.backend = Cv2Backends(self.backend)

        if self.width is None or self.height is None:
            raise ValueError("GoPro camera config requires final `width` and `height`.")
        if self.capture_width is None or self.capture_height is None:
            raise ValueError("GoPro camera config requires raw `capture_width` and `capture_height`.")
        if self.crop_ratio <= 0:
            raise ValueError(f"`crop_ratio` must be positive, but {self.crop_ratio} is provided.")
        if self.fourcc is not None and (not isinstance(self.fourcc, str) or len(self.fourcc) != 4):
            raise ValueError(
                f"`fourcc` must be a 4-character string (e.g., 'MJPG', 'YUYV'), but '{self.fourcc}' is provided."
            )
