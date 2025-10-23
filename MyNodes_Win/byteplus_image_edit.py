"""Image-to-image editing for ByteDance API (legacy interface)."""

from __future__ import annotations

import json
import logging
from typing import Dict

from .common import (
    BYTEPLUS_IMAGE_ENDPOINT,
    build_headers,
    ensure_single_image,
    extract_first_image,
    sanitize_base_url,
    tensor_to_data_uri,
    validate_aspect_ratio_range,
    validate_string,
    resolve_model_id,
    post_json,
)

LOGGER = logging.getLogger(__name__)


class ByteDanceImageEditCustomWin:
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
                        "default": "seededit-3-0-i2i-250628",
                        "multiline": False,
                    },
                ),
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
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
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 5.5,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.05,
                    },
                ),
                "watermark": (
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
        image,
        prompt: str,
        seed: int,
        guidance_scale: float,
        watermark: bool,
    ):
        validate_string(prompt)
        ensure_single_image(image)
        validate_aspect_ratio_range(image, (1, 3), (3, 1))

        base_url = sanitize_base_url(custom_base_url)
        headers = build_headers(custom_api_key)
        model_id = resolve_model_id(custom_model)
        source_url = tensor_to_data_uri(image, "image/png")

        payload: Dict[str, object] = {
            "model": model_id,
            "prompt": prompt.strip(),
            "image": source_url,
            "seed": seed,
            "watermark": watermark,
            "response_format": "url",
        }
        if guidance_scale is not None and "seedream-4" not in model_id.lower():
            payload["guidance_scale"] = guidance_scale

        LOGGER.info("Requesting ByteDance image edit (model=%s seed=%s)", model_id, seed)

        body = post_json(base_url, BYTEPLUS_IMAGE_ENDPOINT, headers, payload)
        image_tensor = extract_first_image(body)
        return (image_tensor,)
