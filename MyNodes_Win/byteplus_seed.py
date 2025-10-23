"""Seed 1.6 Responses API node (legacy interface)."""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from .common import (
    BYTEPLUS_RESPONSES_ENDPOINT,
    build_headers,
    post_json,
    resolve_model_id,
    sanitize_base_url,
    tensor_to_data_uri,
    validate_string,
)

LOGGER = logging.getLogger(__name__)

MODEL_PRESETS: Dict[str, str] = {
    "Seed 1.6 (Chat)": "doubao-seed-1-6-251015",
    "Seed 1.6 Lite": "doubao-seed-1-6-lite-251015",
    "Seed 1.6 Flash": "doubao-seed-1-6-flash-250828",
    "Seed 1.6 Vision": "doubao-seed-1-6-vision-250815",
}


def _optional_str(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    stripped = value.strip()
    return stripped or None


def _build_thinking(mode: str) -> Optional[Dict[str, str]]:
    if mode in ("enabled", "disabled", "auto"):
        return {"type": mode}
    return None


def _build_caching(enabled: bool) -> Optional[Dict[str, str]]:
    return {"type": "enabled"} if enabled else None


def _safe_int(value: Optional[int]) -> Optional[int]:
    if value is None or value == 0:
        return None
    return value


def _safe_float(value: Optional[float]) -> Optional[float]:
    return value if value is not None else None


def _build_stop_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    entries = [line.strip() for line in value.replace("\r", "").split("\n")]
    stops = [item for item in entries if item]
    return stops or None


def _parse_metadata(value: Optional[str]) -> Optional[Dict[str, object]]:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Metadata must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Metadata JSON must be an object.")
    return parsed


def _collect_assistant_text(response: Dict[str, object]) -> str:
    texts: List[str] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") == "output_text" and part.get("text"):
                texts.append(str(part["text"]).strip())
    return "\n\n".join(texts).strip()


def _collect_reasoning_summary(response: Dict[str, object]) -> str:
    summaries: List[str] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        for part in item.get("summary", []) or []:
            if isinstance(part, dict) and part.get("text"):
                summaries.append(str(part["text"]).strip())
    return "\n\n".join(summaries).strip()


class ByteDanceSeedCustomWin:
    @classmethod
    def INPUT_TYPES(cls):
        presets = list(MODEL_PRESETS.keys())
        return {
            "required": {
                "custom_base_url": ("STRING", {"default": "https://ark.cn-beijing.volces.com/api/v3"}),
                "custom_api_key": ("STRING", {"default": "", "password": True}),
                "model_preset": (presets, {"default": presets[0]}),
                "custom_model": ("STRING", {"optional": True, "default": ""}),
                "primary_image": ("IMAGE", {"optional": True}),
                "system_prompt": ("STRING", {"multiline": True, "optional": True, "default": ""}),
                "user_prompt": ("STRING", {"multiline": True, "optional": True, "default": ""}),
                "previous_response_id": ("STRING", {"optional": True, "default": ""}),
                "thinking_mode": (["model_default", "enabled", "disabled", "auto"], {"default": "model_default"}),
                "store_history": ("BOOLEAN", {"default": True}),
                "use_cache": ("BOOLEAN", {"default": False}),
                "max_output_tokens": ("INT", {"default": 0, "optional": True}),
                "temperature": ("FLOAT", {"default": 0.7, "optional": True}),
                "top_p": ("FLOAT", {"default": 0.95, "optional": True}),
                "top_k": ("INT", {"default": 0, "optional": True}),
                "stop_sequences": ("STRING", {"multiline": True, "optional": True, "default": ""}),
                "metadata_json": ("STRING", {"multiline": True, "optional": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    FUNCTION = "generate"
    CATEGORY = "MyNodes/ByteDance"

    def generate(
        self,
        custom_base_url: str,
        custom_api_key: str,
        model_preset: str,
        custom_model: str,
        primary_image,
        system_prompt: str,
        user_prompt: str,
        previous_response_id: str,
        thinking_mode: str,
        store_history: bool,
        use_cache: bool,
        max_output_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_sequences: str,
        metadata_json: str,
    ):
        sys_prompt = _optional_str(system_prompt)
        usr_prompt = _optional_str(user_prompt)
        if not sys_prompt and not usr_prompt and primary_image is None:
            raise ValueError("Provide at least one of system prompt, user prompt, or an input image.")

        preset_model = MODEL_PRESETS.get(model_preset, next(iter(MODEL_PRESETS.values())))
        model_id = resolve_model_id(_optional_str(custom_model) or preset_model)

        base_url = sanitize_base_url(custom_base_url)
        headers = build_headers(custom_api_key)

        messages: List[Dict[str, object]] = []
        if sys_prompt:
            messages.append({"role": "system", "content": [{"type": "input_text", "text": sys_prompt}]})

        user_content: List[Dict[str, object]] = []
        if primary_image is not None:
            user_content.append({"type": "input_image", "image_url": tensor_to_data_uri(primary_image, "image/png")})
        if usr_prompt:
            user_content.append({"type": "input_text", "text": usr_prompt})
        if user_content:
            messages.append({"role": "user", "content": user_content})

        request_body: Dict[str, object] = {
            "model": model_id,
            "input": messages,
            "store": bool(store_history),
        }

        thinking = _build_thinking(thinking_mode)
        if thinking:
            request_body["thinking"] = thinking
        caching = _build_caching(use_cache)
        if caching:
            request_body["caching"] = caching

        prev_id = _optional_str(previous_response_id)
        if prev_id:
            request_body["previous_response_id"] = prev_id

        mot = _safe_int(max_output_tokens)
        if mot is not None:
            request_body["max_output_tokens"] = mot
        temp_val = _safe_float(temperature)
        if temp_val is not None:
            request_body["temperature"] = temp_val
        top_p_val = _safe_float(top_p)
        if top_p_val is not None:
            request_body["top_p"] = top_p_val
        top_k_val = _safe_int(top_k)
        if top_k_val is not None:
            request_body["top_k"] = top_k_val

        stops = _build_stop_list(stop_sequences)
        if stops:
            request_body["stop"] = stops

        metadata = _parse_metadata(metadata_json)
        if metadata:
            request_body["metadata"] = metadata

        LOGGER.info("Submitting Seed responses request (model=%s)", model_id)
        body = post_json(base_url, BYTEPLUS_RESPONSES_ENDPOINT, headers, request_body, timeout=600)

        assistant_text = _collect_assistant_text(body)
        reasoning_summary = _collect_reasoning_summary(body)
        response_id = str(body.get("id", ""))
        raw_json = json.dumps(body, ensure_ascii=False)

        return (
            assistant_text,
            response_id,
            reasoning_summary,
            raw_json,
        )
