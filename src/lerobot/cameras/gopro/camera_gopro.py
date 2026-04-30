from typing import Any

import cv2  # type: ignore  # TODO: add type stubs for OpenCV
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

from ..opencv.camera_opencv import OpenCVCamera
from .configuration_gopro import GoProCameraConfig


def center_crop_then_resize(
    image: NDArray[Any],
    out_width: int,
    out_height: int,
    crop_ratio: float = 1.0,
) -> NDArray[Any]:
    """Match UMI replay-buffer get_image_transform: center crop, then resize."""

    ih, iw, _ = image.shape
    ch = round(ih * crop_ratio)
    cw = round(ih * crop_ratio / out_height * out_width)

    if ch <= 0 or cw <= 0:
        raise ValueError(f"Invalid crop size ({cw}, {ch}) from crop_ratio={crop_ratio}.")
    if ch > ih or cw > iw:
        raise ValueError(
            f"Requested center crop ({cw}, {ch}) exceeds input image size ({iw}, {ih}). "
            "Decrease crop_ratio or adjust the output aspect ratio."
        )

    w_slice_start = (iw - cw) // 2
    h_slice_start = (ih - ch) // 2
    image = image[h_slice_start : h_slice_start + ch, w_slice_start : w_slice_start + cw, :]
    return cv2.resize(image, (out_width, out_height), interpolation=cv2.INTER_AREA)


class GoProCamera(OpenCVCamera):
    """OpenCV camera with GoPro-specific UMI-compatible postprocessing."""

    config: GoProCameraConfig

    def __init__(self, config: GoProCameraConfig):
        super().__init__(config)
        self.config = config
        self.capture_width = config.capture_width
        self.capture_height = config.capture_height
        self.output_width = config.width
        self.output_height = config.height
        self.crop_ratio = config.crop_ratio

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.index_or_path})"

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        image = super()._postprocess_image(image)
        return center_crop_then_resize(
            image,
            out_width=self.output_width,
            out_height=self.output_height,
            crop_ratio=self.crop_ratio,
        )
