"""Video generation nodes using ByteDance API."""

from __future__ import annotations

import logging
import math
from typing import Dict, List

from .common import (
    BYTEPLUS_TASK_ENDPOINT,
    BYTEPLUS_TASK_STATUS_ENDPOINT,
    build_headers,
    download_video_output,
    encode_tensor_list,
    ensure_image_count_between,
    poll_task,
    post_json,
    resolve_model_id,
    sanitize_base_url,
    tensor_to_data_uri,
    validate_aspect_ratio_range,
    validate_image_dimensions,
    validate_string,
)

LOGGER = logging.getLogger(__name__)

VIDEO_TASKS_EXECUTION_TIME = {
    "seedance-1-0-lite-t2v-250428": {
        "480p": 40,
        "720p": 60,
        "1080p": 90,
    },
    "seedance-1-0-lite-i2v-250428": {
        "480p": 40,
        "720p": 60,
        "1080p": 90,
    },
    "seedance-1-0-pro-250528": {
        "480p": 70,
        "720p": 85,
        "1080p": 115,
    },
}


def _raise_if_text_params(prompt: str, text_params: List[str]) -> None:
    lowered = prompt.lower()
    for token in text_params:
        if f"--{token}" in lowered:
            raise ValueError(f"Remove --{token} from the prompt; use the node inputs instead.")


def _compose_video_prompt(
    base_prompt: str,
    *,
    resolution: str,
    aspect_ratio: str,
    duration: int,
    seed: int | None,
    camera_fixed: bool | None,
    watermark: bool | None,
) -> str:
    parts = [base_prompt.strip()]
    parts.append(f"--resolution {resolution}")
    parts.append(f"--ratio {aspect_ratio}")
    parts.append(f"--duration {duration}")
    if seed is not None:
        parts.append(f"--seed {seed}")
    if camera_fixed is not None:
        parts.append(f"--camerafixed {str(bool(camera_fixed)).lower()}")
    if watermark is not None:
        parts.append(f"--watermark {str(bool(watermark)).lower()}")
    return " ".join(parts)


def _submit_video_task(base_url: str, headers: Dict[str, str], payload: Dict[str, object]) -> str:
    body = post_json(base_url, BYTEPLUS_TASK_ENDPOINT, headers, payload)
    task_id = body.get("id") or body.get("data", {}).get("id") if isinstance(body.get("data"), dict) else None
    if not task_id:
        raise RuntimeError(f"Task creation response missing id: {body}")
    return str(task_id)


def _await_video(base_url: str, headers: Dict[str, str], task_id: str) -> Dict[str, object]:
    status = poll_task(
        base_url,
        task_id,
        headers,
        poll_path=BYTEPLUS_TASK_STATUS_ENDPOINT,
        interval=3.0,
        timeout=900.0,
    )
    content = status.get("content") or {}
    video_url = content.get("video_url")
    if not video_url:
        raise RuntimeError(f"Task succeeded but no video URL returned: {status}")
    LOGGER.info("ByteDance video task %s succeeded. URL=%s", task_id, video_url)
    return download_video_output(video_url)


class ByteDanceTextToVideoCustomWin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "custom_base_url": ("STRING", {"default": "https://ark.cn-beijing.volces.com/api/v3"}),
                "custom_api_key": ("STRING", {"default": "", "password": True}),
                "custom_model": ("STRING", {"default": "seedance-1-0-pro-250528"}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the video."}),
                "resolution": (["480p", "720p", "1080p"],),
                "aspect_ratio": (["16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],),
                "duration": ("INT", {"default": 5, "min": 3, "max": 12, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "camera_fixed": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "generate"
    CATEGORY = "MyNodes/video"

    def generate(
        self,
        custom_base_url: str,
        custom_api_key: str,
        custom_model: str,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        camera_fixed: bool,
        watermark: bool,
    ):
        validate_string(prompt)
        _raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "camerafixed", "watermark"])

        base_url = sanitize_base_url(custom_base_url)
        headers = build_headers(custom_api_key)
        model_id = resolve_model_id(custom_model)
        effective_seed = seed if seed is not None else None
        composed_prompt = _compose_video_prompt(
            prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration=duration,
            seed=effective_seed,
            camera_fixed=camera_fixed,
            watermark=watermark,
        )

        payload = {
            "model": model_id,
            "content": [
                {"type": "text", "text": composed_prompt},
            ],
        }

        LOGGER.info("Submitting text-to-video task (model=%s, resolution=%s)", model_id, resolution)
        task_id = _submit_video_task(base_url, headers, payload)
        return (_await_video(base_url, headers, task_id),)


class ByteDanceImageToVideoCustomWin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "custom_base_url": ("STRING", {"default": "https://ark.cn-beijing.volces.com/api/v3"}),
                "custom_api_key": ("STRING", {"default": "", "password": True}),
                "custom_model": ("STRING", {"default": "seedance-1-0-pro-250528"}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the video."}),
                "image": ("IMAGE",),
                "resolution": (["480p", "720p", "1080p"],),
                "aspect_ratio": (["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],),
                "duration": ("INT", {"default": 5, "min": 3, "max": 12, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "camera_fixed": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "generate"
    CATEGORY = "MyNodes/video"

    def generate(
        self,
        custom_base_url: str,
        custom_api_key: str,
        custom_model: str,
        prompt: str,
        image,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        camera_fixed: bool,
        watermark: bool,
    ):
        validate_string(prompt)
        _raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "camerafixed", "watermark"])

        validate_image_dimensions(image, min_width=300, min_height=300, max_width=6000, max_height=6000)
        validate_aspect_ratio_range(image, (2, 5), (5, 2))

        base_url = sanitize_base_url(custom_base_url)
        headers = build_headers(custom_api_key)
        model_id = resolve_model_id(custom_model)
        image_url = tensor_to_data_uri(image, "image/png")
        effective_seed = seed if seed is not None else None
        composed_prompt = _compose_video_prompt(
            prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration=duration,
            seed=effective_seed,
            camera_fixed=camera_fixed,
            watermark=watermark,
        )

        payload = {
            "model": model_id,
            "content": [
                {"type": "text", "text": composed_prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }

        LOGGER.info("Submitting image-to-video task (model=%s, resolution=%s)", model_id, resolution)
        task_id = _submit_video_task(base_url, headers, payload)
        return (_await_video(base_url, headers, task_id),)


class ByteDanceFirstLastFrameCustomWin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "custom_base_url": ("STRING", {"default": "https://ark.cn-beijing.volces.com/api/v3"}),
                "custom_api_key": ("STRING", {"default": "", "password": True}),
                "custom_model": ("STRING", {"default": "seedance-1-0-pro-250528"}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the video."}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "resolution": (["480p", "720p", "1080p"],),
                "aspect_ratio": (["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],),
                "duration": ("INT", {"default": 5, "min": 3, "max": 12, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "camera_fixed": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "generate"
    CATEGORY = "MyNodes/video"

    def generate(
        self,
        custom_base_url: str,
        custom_api_key: str,
        custom_model: str,
        prompt: str,
        first_frame,
        last_frame,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        camera_fixed: bool,
        watermark: bool,
    ):
        validate_string(prompt)
        _raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "camerafixed", "watermark"])

        for frame in (first_frame, last_frame):
            validate_image_dimensions(frame, min_width=300, min_height=300, max_width=6000, max_height=6000)
            validate_aspect_ratio_range(frame, (2, 5), (5, 2))

        base_url = sanitize_base_url(custom_base_url)
        headers = build_headers(custom_api_key)
        model_id = resolve_model_id(custom_model)
        composed_prompt = _compose_video_prompt(
            prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration=duration,
            seed=seed if seed is not None else None,
            camera_fixed=camera_fixed,
            watermark=watermark,
        )

        payload = {
            "model": model_id,
            "content": [
                {"type": "text", "text": composed_prompt},
                {"type": "image_url", "image_url": {"url": tensor_to_data_uri(first_frame, "image/png")}, "role": "first_frame"},
                {"type": "image_url", "image_url": {"url": tensor_to_data_uri(last_frame, "image/png")}, "role": "last_frame"},
            ],
        }

        LOGGER.info("Submitting first/last frame video task (model=%s, resolution=%s)", model_id, resolution)
        task_id = _submit_video_task(base_url, headers, payload)
        return (_await_video(base_url, headers, task_id),)


class ByteDanceReferenceImagesVideoCustomWin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "custom_base_url": ("STRING", {"default": "https://ark.cn-beijing.volces.com/api/v3"}),
                "custom_api_key": ("STRING", {"default": "", "password": True}),
                "custom_model": ("STRING", {"default": "seedance-1-0-lite-i2v-250428"}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the video."}),
                "images": ("IMAGE",),
                "resolution": (["480p", "720p"],),
                "aspect_ratio": (["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],),
                "duration": ("INT", {"default": 5, "min": 3, "max": 12, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "camera_fixed": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "generate"
    CATEGORY = "MyNodes/video"

    def generate(
        self,
        custom_base_url: str,
        custom_api_key: str,
        custom_model: str,
        prompt: str,
        images,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        camera_fixed: bool,
        watermark: bool,
    ):
        validate_string(prompt)
        _raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "camerafixed", "watermark"])

        ensure_image_count_between(images, 1, 4)
        base_url = sanitize_base_url(custom_base_url)
        headers = build_headers(custom_api_key)
        model_id = resolve_model_id(custom_model)
        composed_prompt = _compose_video_prompt(
            prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration=duration,
            seed=seed if seed is not None else None,
            camera_fixed=camera_fixed,
            watermark=watermark,
        )

        reference_images = encode_tensor_list(images, 4)
        content = [{"type": "text", "text": composed_prompt}]
        for ref in reference_images:
            content.append({"type": "image_url", "image_url": {"url": ref}, "role": "reference_image"})

        payload = {
            "model": model_id,
            "content": content,
        }

        LOGGER.info("Submitting reference-image video task (model=%s, refs=%d)", model_id, len(reference_images))
        task_id = _submit_video_task(base_url, headers, payload)
        return (_await_video(base_url, headers, task_id),)
