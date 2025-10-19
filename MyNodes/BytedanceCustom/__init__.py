# Re-export so relative imports resolve cleanly when the package is loaded by ComfyUI.
from .nodes_bytedance_custom import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
