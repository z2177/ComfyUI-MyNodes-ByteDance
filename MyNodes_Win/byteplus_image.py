"""
ByteDance image generation node implemented with the legacy ComfyUI custom node interface.

This variant mirrors the behaviour of the async comfy_api based node but avoids those
dependencies so that Windows portable builds (e.g., v0.3.64) can load it.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
import torch
from PIL import Image

LOGGER = logging.getLogger(__name__)

BYTEPLUS_IMAGE_ENDPOINT = "images/generations"

RECOMMENDED_PRESETS: Tuple[Tuple[str, Optional[int], Optional[int]], ...] = (
    ("1024x1024 (1:1)", 1024, 1024),
    ("864x1152 (3:4)", 864, 1152),
    ("1152x864 (4:3)", 1152, 864),
    ("1280x720 (16:9)", 1280, 720),
    ("720x1280 (9:16)", 720, 1280),
    ("832x1248 (2:3)", 832, 1248),
    ("1248x832 (3:2)", 1248, 832),
    ("1512x648 (21:9)", 1512, 648),
    ("2048x2048 (1:1)", 2048, 2048),
    ("Custom", None, None),
)


def _sanitize_base_url(base_url: str) -> str:
    candidate = (base_url or "").strip()
    if not candidate:
        raise ValueError("Custom Base URL cannot be empty.")
    parsed = urlparse(candidate)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Custom Base URL must include scheme and host, e.g., https://example.com/api/v3")
    return candidate.rstrip("/") + "/"


def _build_headers(api_key: str) -> Dict[str, str]:
    token = (api_key or "").strip()
    if not token:
        raise ValueError("Custom API Key cannot be empty.")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


def _resolve_model_id(model: str) -> str:
    resolved = (model or "").strip()
    if not resolved:
        raise ValueError("Custom Model / Endpoint ID cannot be empty.")
    return resolved


def _pick_dimensions(size_label: str, width: int, height: int) -> Tuple[int, int]:
    for label, preset_w, preset_h in RECOMMENDED_PRESETS:
        if label == size_label and preset_w and preset_h:
            return preset_w, preset_h
    if not (512 <= width <= 2048 and 512 <= height <= 2048):
        raise ValueError("Width and height must be within [512, 2048] when using a custom size.")
    return width, height


def _tensor_from_base64(data: str) -> torch.Tensor:
    if "," in data:
        data = data.split(",", 1)[1]
    image_bytes = base64.b64decode(data)
    return _tensor_from_bytes(image_bytes)


def _tensor_from_bytes(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _tensor_from_url(url: str, timeout: int = 120) -> torch.Tensor:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return _tensor_from_bytes(response.content)


def _extract_first_image(data: Dict[str, object]) -> torch.Tensor:
    items = data.get("data")
    if not isinstance(items, (list, tuple)):
        raise RuntimeError("API response did not include any image data.")
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("b64_json"):
            return _tensor_from_base64(str(item["b64_json"]))
        if item.get("url"):
            return _tensor_from_url(str(item["url"]))
    raise RuntimeError("API response did not include a usable image payload.")


class ByteDanceImageCustomWin:
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
                        "default": "seedream-3-0-t2i-250415",
                        "multiline": False,
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "Describe the scene you want to generate.",
                        "multiline": True,
                    },
                ),
                "size_preset": (
                    [label for label, _, _ in RECOMMENDED_PRESETS],
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 512,
                        "max": 2048,
                        "step": 64,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 512,
                        "max": 2048,
                        "step": 64,
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
                        "default": 2.5,
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
    CATEGORY = "MyNodes-Win/image"

    def generate(
        self,
        custom_base_url: str,
        custom_api_key: str,
        custom_model: str,
        prompt: str,
        size_preset: str,
        width: int,
        height: int,
        seed: int,
        guidance_scale: float,
        watermark: bool,
    ):
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        base_url = _sanitize_base_url(custom_base_url)
        headers = _build_headers(custom_api_key)
        model_id = _resolve_model_id(custom_model)
        resolved_width, resolved_height = _pick_dimensions(size_preset, width, height)

        endpoint = urljoin(base_url, BYTEPLUS_IMAGE_ENDPOINT)
        payload = {
            "model": model_id,
            "prompt": prompt,
            "size": f"{resolved_width}x{resolved_height}",
            "seed": seed,
            "guidance_scale": guidance_scale,
            "watermark": watermark,
            "response_format": "url",
        }

        LOGGER.info(
            "Requesting ByteDance image generation (model=%s size=%sx%s seed=%s)",
            model_id,
            resolved_width,
            resolved_height,
            seed,
        )
        LOGGER.debug("POST %s payload=%s", endpoint, json.dumps(payload))

        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=180)
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to reach ByteDance API: {exc}") from exc

        if response.status_code >= 400:
            try:
                error_payload = response.json()
            except ValueError:
                error_payload = response.text
            raise RuntimeError(
                f"ByteDance API returned HTTP {response.status_code}: {error_payload}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError("ByteDance API returned a non-JSON response.") from exc

        if isinstance(body, dict) and body.get("error"):
            raise RuntimeError(
                f"ByteDance API error: {body['error'].get('message', 'Unknown error')}"
            )

        image_tensor = _extract_first_image(body)
        return (image_tensor,)
