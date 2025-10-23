"""
Legacy-compatible ByteDance custom nodes for Windows portable builds of ComfyUI.

These nodes expose the traditional NODE_CLASS_MAPPINGS interface so that older
ComfyUI versions (e.g. v0.3.64) can discover and register them without relying
on the newer comfy_api extension system.
"""

from .byteplus_image import ByteDanceImageCustomWin
from .byteplus_image_edit import ByteDanceImageEditCustomWin
from .byteplus_seedream import ByteDanceSeedreamCustomWin
from .byteplus_video import (
    ByteDanceTextToVideoCustomWin,
    ByteDanceImageToVideoCustomWin,
    ByteDanceFirstLastFrameCustomWin,
    ByteDanceReferenceImagesVideoCustomWin,
)
from .byteplus_seed import ByteDanceSeedCustomWin

NODE_CLASS_MAPPINGS = {
    "ByteDanceImageCustomWin": ByteDanceImageCustomWin,
    "ByteDanceImageEditCustomWin": ByteDanceImageEditCustomWin,
    "ByteDanceSeedreamCustomWin": ByteDanceSeedreamCustomWin,
    "ByteDanceTextToVideoCustomWin": ByteDanceTextToVideoCustomWin,
    "ByteDanceImageToVideoCustomWin": ByteDanceImageToVideoCustomWin,
    "ByteDanceFirstLastFrameCustomWin": ByteDanceFirstLastFrameCustomWin,
    "ByteDanceReferenceImagesVideoCustomWin": ByteDanceReferenceImagesVideoCustomWin,
    "ByteDanceSeedCustomWin": ByteDanceSeedCustomWin,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ByteDanceImageCustomWin": "ByteDance Image (Custom - Legacy)",
    "ByteDanceImageEditCustomWin": "ByteDance Image Edit (Custom - Legacy)",
    "ByteDanceSeedreamCustomWin": "ByteDance Seedream 4 (Custom - Legacy)",
    "ByteDanceTextToVideoCustomWin": "ByteDance Text to Video (Custom - Legacy)",
    "ByteDanceImageToVideoCustomWin": "ByteDance Image to Video (Custom - Legacy)",
    "ByteDanceFirstLastFrameCustomWin": "ByteDance First-Last-Frame to Video (Legacy)",
    "ByteDanceReferenceImagesVideoCustomWin": "ByteDance Reference Images to Video (Legacy)",
    "ByteDanceSeedCustomWin": "ByteDance Seed 1.6 (Custom - Legacy)",
}

__all__ = [
    "ByteDanceImageCustomWin",
    "ByteDanceImageEditCustomWin",
    "ByteDanceSeedreamCustomWin",
    "ByteDanceTextToVideoCustomWin",
    "ByteDanceImageToVideoCustomWin",
    "ByteDanceFirstLastFrameCustomWin",
    "ByteDanceReferenceImagesVideoCustomWin",
    "ByteDanceSeedCustomWin",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
