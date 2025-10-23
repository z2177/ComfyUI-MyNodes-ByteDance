"""Seedream 4 text/image generation for ByteDance API."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from .common import (
    BYTEPLUS_IMAGE_ENDPOINT,
    build_headers,
    concat_tensors,
    encode_tensor_list,
    extract_all_image_urls,
    extract_first_image,
    post_json,
    resolve_model_id,
    sanitize_base_url,

    validate_string,
)

LOGGER = logging.getLogger(__name__)

RECOMMENDED_PRESETS_SEEDREAM_4: Tuple[Tuple[str, Optional[int], Optional[int]], ...] = (
    ("2048x2048 (1:1)", 2048, 2048),
    ("2304x1728 (4:3)", 2304, 1728),
    ("1728x2304 (3:4)", 1728, 2304),
    ("2560x1440 (16:9)", 2560, 1440),
    ("1440x2560 (9:16)", 1440, 2560),
    ("2496x1664 (3:2)", 2496, 1664),
    ("1664x2496 (2:3)", 1664, 2496),
    ("3024x1296 (21:9)", 3024, 1296),
    ("4096x4096 (1:1)", 4096, 4096),
    ("Custom", None, None),
)


def _pick_dimensions(size_label: str, width: int, height: int) -> Tuple[int, int]:
    for label, preset_w, preset_h in RECOMMENDED_PRESETS_SEEDREAM_4:
        if label == size_label and preset_w and preset_h:
            return preset_w, preset_h
    if not (1024 <= width <= 4096 and 1024 <= height <= 4096):
        raise ValueError("Width and height must be within [1024, 4096] when using a custom size.")
    return width, height


class ByteDanceSeedreamCustomWin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "custom_base_url": (
                    "STRING",
                    {
                        "default": "https://ark.cn-beijing.volces.com/api/v3",
                        "multiline": False,
                    },
                ),
                "custom_api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "password": True,
                    },
                ),
                "custom_model": (
                    "STRING",
                    {
                        "default": "seedream-4-0-250828",
                        "multiline": False,
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "image": ("IMAGE", {"optional": True}),
                "size_preset": (
                    [label for label, _, _ in RECOMMENDED_PRESETS_SEEDREAM_4],
                ),
                "width": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 1024,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 1024,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "sequential_image_generation": (
                    ["disabled", "auto"],
                ),
                "max_images": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 15,
                        "step": 1,
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                    },
                ),
                "watermark": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
                "fail_on_partial": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "MyNodes/image"

    def generate(
        self,
        custom_base_url: str,
        custom_api_key: str,
        custom_model: str,
        prompt: str,
        image,
        size_preset: str,
        width: int,
        height: int,
        sequential_image_generation: str,
        max_images: int,
        seed: int,
        watermark: bool,
        fail_on_partial: bool,
    ):
        validate_string(prompt)
        base_url = sanitize_base_url(custom_base_url)
        headers = build_headers(custom_api_key)
        model_id = resolve_model_id(custom_model)
        resolved_width, resolved_height = _pick_dimensions(size_preset, width, height)

        reference_images = encode_tensor_list(image, 10)
        if sequential_image_generation == "auto" and len(reference_images) + max_images > 15:
            raise ValueError("Reference images plus generated images cannot exceed 15.")

        payload: Dict[str, object] = {
            "model": model_id,
            "prompt": prompt.strip(),
            "size": f"{resolved_width}x{resolved_height}",
            "seed": seed,
            "watermark": watermark,
            "sequential_image_generation": sequential_image_generation,
        }
        if reference_images:
            payload["image"] = reference_images
        if sequential_image_generation == "auto":
            payload["sequential_image_generation_options"] = {"max_images": max_images}

        LOGGER.info(
            "Requesting Seedream 4 generation (model=%s size=%sx%s sequential=%s)",
            model_id,
            resolved_width,
            resolved_height,
            sequential_image_generation,
        )

        body = post_json(base_url, BYTEPLUS_IMAGE_ENDPOINT, headers, payload)
        urls = extract_all_image_urls(body)
        if not urls:
            image_tensor = extract_first_image(body)
            return (image_tensor,)
        if fail_on_partial and len(urls) < len(body.get("data", [])):
            raise RuntimeError(f"Only {len(urls)} of {len(body.get('data', []))} images were generated before error.")
        combined = concat_tensors(urls)
        return (combined,)
