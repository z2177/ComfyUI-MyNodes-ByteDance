"""
Legacy-compatible ByteDance custom nodes for Windows portable builds of ComfyUI.

These nodes expose the traditional NODE_CLASS_MAPPINGS interface so that older
ComfyUI versions (e.g. v0.3.64) can discover and register them without relying
on the newer comfy_api extension system.
"""

from .byteplus_image import ByteDanceImageCustomWin

NODE_CLASS_MAPPINGS = {
    "ByteDanceImageCustomWin": ByteDanceImageCustomWin,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ByteDanceImageCustomWin": "ByteDance Image (Custom - Legacy)",
}

__all__ = [
    "ByteDanceImageCustomWin",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
