from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any, Optional, Literal
from urllib.parse import urlparse

import torch
from PIL import Image
from pydantic import BaseModel, Field, ConfigDict

from comfy_api.latest import ComfyExtension, IO
from comfy_api_nodes.apis.client import ApiEndpoint, HttpMethod, SynchronousOperation

LOGGER = logging.getLogger(__name__)


class ResponsesThinking(BaseModel):
    type: Literal["enabled", "disabled", "auto"]


class ResponsesCaching(BaseModel):
    type: Literal["enabled", "disabled"]


class ResponsesContentItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: Optional[str] = None
    image_url: Optional[str] = None


class ResponsesInputMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: list[ResponsesContentItem]


class ResponsesCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: list[ResponsesInputMessage]
    previous_response_id: Optional[str] = None
    thinking: Optional[ResponsesThinking] = None
    caching: Optional[ResponsesCaching] = None
    store: Optional[bool] = None
    max_output_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    stop: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None


class ResponsesUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ResponsesOutputContent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: Optional[str] = None


class ResponsesOutputItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    type: str
    role: Optional[str] = None
    content: Optional[list[ResponsesOutputContent]] = None
    summary: Optional[list[ResponsesOutputContent]] = None
    status: Optional[str] = None


class ResponsesCreateResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    status: str
    model: str
    output: list[ResponsesOutputItem] = Field(default_factory=list)
    usage: Optional[ResponsesUsage] = None
    max_output_tokens: Optional[int] = None
    service_tier: Optional[str] = None
    store: Optional[bool] = None
    caching: Optional[dict[str, Any]] = None
    expire_at: Optional[int] = None


RESPONSES_ENDPOINT = ApiEndpoint(
    path="/responses",
    method=HttpMethod.POST,
    request_model=ResponsesCreateRequest,
    response_model=ResponsesCreateResponse,
)

ResponsesThinking.model_rebuild()
ResponsesCaching.model_rebuild()
ResponsesContentItem.model_rebuild()
ResponsesInputMessage.model_rebuild()
ResponsesCreateRequest.model_rebuild()
ResponsesOutputContent.model_rebuild()
ResponsesOutputItem.model_rebuild()
ResponsesCreateResponse.model_rebuild()

MODEL_PRESETS: dict[str, str] = {
    "Seed 1.6 (Chat)": "doubao-seed-1-6-251015",
    "Seed 1.6 Lite": "doubao-seed-1-6-lite-251015",
    "Seed 1.6 Flash": "doubao-seed-1-6-flash-250828",
    "Seed 1.6 Vision": "doubao-seed-1-6-vision-250815",
}


def _sanitize_base_url(base_url: Optional[str]) -> str:
    candidate = (base_url or "").strip()
    if not candidate:
        raise ValueError("Custom base URL is required.")
    parsed = urlparse(candidate)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Custom base URL must include scheme and host.")
    return candidate.rstrip("/") + "/"


def _build_auth_kwargs(custom_api_key: Optional[str]) -> dict[str, Optional[str]]:
    token = (custom_api_key or "").strip()
    if not token:
        raise ValueError("Custom API Key is required.")
    return {
        "auth_token": token,
        "comfy_api_key": None,
    }


def _prepare_request_context(
    *,
    custom_base_url: str,
    custom_api_key: str,
    custom_model: str,
) -> tuple[str, dict[str, Optional[str]], str]:
    model_id = (custom_model or "").strip()
    if not model_id:
        raise ValueError("Model ID / Endpoint ID cannot be empty.")
    api_base = _sanitize_base_url(custom_base_url)
    auth_kwargs = _build_auth_kwargs(custom_api_key)
    return api_base, auth_kwargs, model_id


def _to_optional_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


def _build_text_message(role: str, text: str) -> ResponsesInputMessage:
    return ResponsesInputMessage(
        role=role, content=[ResponsesContentItem(type="input_text", text=text)]
    )


def _build_stop_list(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    tokens = [item.strip() for item in value.replace("\r", "").split("\n")]
    cleaned = [token for token in tokens if token]
    return cleaned or None


def _parse_metadata(value: Optional[str]) -> Optional[dict[str, Any]]:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Metadata must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Metadata JSON must be an object.")
    return parsed


def _collect_assistant_text(response: ResponsesCreateResponse) -> str:
    texts: list[str] = []
    for item in response.output:
        if item.type != "message" or not item.content:
            continue
        for part in item.content:
            if part.type == "output_text" and part.text:
                texts.append(part.text)
    return "\n\n".join(texts).strip()


def _collect_reasoning_summary(response: ResponsesCreateResponse) -> str:
    summaries: list[str] = []
    for item in response.output:
        if item.type != "reasoning" or not item.summary:
            continue
        for part in item.summary:
            if part.text:
                summaries.append(part.text.strip())
    return "\n\n".join(summaries).strip()


async def _execute_responses_call(
    request: ResponsesCreateRequest,
    *,
    auth_kwargs: dict[str, Optional[str]],
    api_base: str,
) -> ResponsesCreateResponse:
    LOGGER.debug(
        "Submitting Responses API request (model=%s, previous=%s)",
        request.model,
        request.previous_response_id,
    )
    operation = SynchronousOperation(
        endpoint=RESPONSES_ENDPOINT,
        request=request,
        auth_kwargs=auth_kwargs,
        api_base=api_base,
    )
    return await operation.execute()


def _standard_outputs() -> list[IO.Output]:
    return [
        IO.String.Output("assistant_text", display_name="Assistant Text"),
        IO.String.Output("response_id", display_name="Response ID"),
        IO.String.Output("reasoning_summary", display_name="Reasoning Summary"),
        IO.String.Output("raw_response", display_name="Raw Response JSON"),
    ]


def _build_thinking(mode: str) -> Optional[ResponsesThinking]:
    if mode in ("enabled", "disabled", "auto"):
        return ResponsesThinking(type=mode)
    return None


def _build_caching(enabled: bool) -> Optional[ResponsesCaching]:
    return ResponsesCaching(type="enabled") if enabled else None


def _safe_int(value: Optional[int]) -> Optional[int]:
    if value is None or value == 0:
        return None
    return value


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value


def _tensor_to_data_uri(image: torch.Tensor, mime_type: str = "image/png") -> str:
    if image.dim() == 4:
        image = image[0]
    if image.dim() != 3:
        raise ValueError("Expected image tensor with shape [H, W, C] or [B, H, W, C].")
    channels = image.shape[-1]
    if channels not in (1, 3, 4):
        raise ValueError("Image tensor must have 1, 3, or 4 channels.")
    tensor = image.detach().cpu()
    if tensor.dtype not in (torch.float16, torch.float32, torch.float64):
        tensor = tensor.float()
    tensor = tensor.clamp(0.0, 1.0)
    array = (tensor * 255.0).round().byte().numpy()
    if channels == 1:
        array = array[:, :, 0]
    image_format = "PNG" if mime_type.lower().endswith("png") else "JPEG"
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


class ByteDanceSeedNode(IO.ComfyNode):
    NODE_ID = "ByteDanceSeedNode"
    DISPLAY_NAME = "ByteDance Seed 1.6 (Custom)"

    @classmethod
    def define_schema(cls):
        preset_names = list(MODEL_PRESETS.keys())
        default_preset = preset_names[0]
        return IO.Schema(
            node_id=cls.NODE_ID,
            display_name=cls.DISPLAY_NAME,
            category="MyNodes/ByteDance",
            description="Invoke Doubao Seed 1.6 (chat/lite/flash/vision) via Responses API with optional multimodal input.",
            inputs=[
                IO.String.Input(
                    "custom_base_url",
                    display_name="Custom Base URL",
                    tooltip="Base endpoint, e.g. https://ark.cn-beijing.volces.com/api/v3",
                    default="https://ark.cn-beijing.volces.com/api/v3",
                ),
                IO.String.Input(
                    "custom_api_key",
                    display_name="Custom API Key",
                    tooltip="Bearer token from Volcengine console.",
                ),
                IO.Combo.Input(
                    "model_preset",
                    options=preset_names,
                    default=default_preset,
                    tooltip="Choose a Seed 1.6 model preset; leave Custom Model empty to use this value.",
                ),
                IO.String.Input(
                    "custom_model",
                    display_name="Custom Model",
                    optional=True,
                    tooltip="Override with a specific model/version or Endpoint ID (e.g. ep-***).",
                ),
                IO.Image.Input(
                    "primary_image",
                    display_name="Primary Image",
                    optional=True,
                    tooltip="Optional image tensor; if provided, will be encoded as input_image content.",
                ),
                IO.String.Input(
                    "system_prompt",
                    display_name="System Prompt",
                    multiline=True,
                    optional=True,
                    tooltip="Optional system message sent as the first turn.",
                ),
                IO.String.Input(
                    "user_prompt",
                    display_name="User Prompt",
                    multiline=True,
                    optional=True,
                    tooltip="Primary user message or instructions.",
                ),
                IO.String.Input(
                    "previous_response_id",
                    display_name="Previous Response ID",
                    optional=True,
                    tooltip="Chain with a prior Responses ID to continue dialogue or reuse cache.",
                ),
                IO.Combo.Input(
                    "thinking_mode",
                    options=["model_default", "enabled", "disabled", "auto"],
                    default="model_default",
                    tooltip="Control deep thinking; leave as model_default to follow server behaviour.",
                ),
                IO.Boolean.Input(
                    "store_history",
                    display_name="Store History",
                    default=True,
                    tooltip="Send store=true so the turn is persisted for later reuse.",
                ),
                IO.Boolean.Input(
                    "use_cache",
                    display_name="Use Cache",
                    default=False,
                    tooltip="Enable Responses caching by sending caching.type=enabled.",
                ),
                IO.Int.Input(
                    "max_output_tokens",
                    default=0,
                    optional=True,
                    tooltip="Optional cap for completion tokens; leave 0 to use server default.",
                ),
                IO.Float.Input(
                    "temperature",
                    default=0.7,
                    optional=True,
                    tooltip="Sampling temperature (0-2). Leave blank for server default.",
                ),
                IO.Float.Input(
                    "top_p",
                    default=0.95,
                    optional=True,
                    tooltip="Top-p nucleus sampling parameter.",
                ),
                IO.Int.Input(
                    "top_k",
                    default=0,
                    optional=True,
                    tooltip="Top-k sampling parameter; 0 to skip.",
                ),
                IO.String.Input(
                    "stop_sequences",
                    display_name="Stop Sequences",
                    multiline=True,
                    optional=True,
                    tooltip="One stop sequence per line.",
                ),
                IO.String.Input(
                    "metadata_json",
                    display_name="Metadata JSON",
                    multiline=True,
                    optional=True,
                    tooltip="Optional JSON object forwarded to metadata field.",
                ),
            ],
            outputs=_standard_outputs(),
            hidden=[IO.Hidden.unique_id],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        custom_base_url: str,
        custom_api_key: str,
        model_preset: str,
        custom_model: Optional[str] = None,
        primary_image: Optional[torch.Tensor] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        thinking_mode: str = "model_default",
        store_history: bool = True,
        use_cache: bool = False,
        max_output_tokens: Optional[int] = None,
        max_prompt_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[str] = None,
        metadata_json: Optional[str] = None,
    ) -> IO.NodeOutput:
        sys_prompt = _to_optional_str(system_prompt)
        usr_prompt = _to_optional_str(user_prompt)

        if not sys_prompt and not usr_prompt and primary_image is None:
            raise ValueError("Provide at least one of System Prompt, User Prompt, or an input image.")

        preset_model = MODEL_PRESETS.get(model_preset, next(iter(MODEL_PRESETS.values())))
        model_id = _to_optional_str(custom_model) or preset_model

        api_base, auth_kwargs, resolved_model = _prepare_request_context(
            custom_base_url=custom_base_url,
            custom_api_key=custom_api_key,
            custom_model=model_id,
        )

        messages: list[ResponsesInputMessage] = []
        if sys_prompt:
            messages.append(_build_text_message("system", sys_prompt))

        user_content: list[ResponsesContentItem] = []
        if primary_image is not None:
            user_content.append(
                ResponsesContentItem(type="input_image", image_url=_tensor_to_data_uri(primary_image, "image/png"))
            )
        if usr_prompt:
            user_content.append(ResponsesContentItem(type="input_text", text=usr_prompt))

        if user_content:
            messages.append(ResponsesInputMessage(role="user", content=user_content))

        request = ResponsesCreateRequest(
            model=resolved_model,
            input=messages,
            previous_response_id=_to_optional_str(previous_response_id),
            thinking=_build_thinking(thinking_mode),
            caching=_build_caching(use_cache),
            store=store_history,
            max_output_tokens=_safe_int(max_output_tokens),
            temperature=_safe_float(temperature),
            top_p=_safe_float(top_p),
            top_k=_safe_int(top_k),
            stop=_build_stop_list(stop_sequences),
            metadata=_parse_metadata(metadata_json),
        )

        response = await _execute_responses_call(
            request,
            auth_kwargs=auth_kwargs,
            api_base=api_base,
        )

        assistant_text = _collect_assistant_text(response)
        reasoning_summary = _collect_reasoning_summary(response)
        raw_json = json.dumps(response.model_dump(mode="json"), ensure_ascii=False)

        return IO.NodeOutput(
            assistant_text,
            response.id,
            reasoning_summary,
            raw_json,
        )


class SeedModelExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [ByteDanceSeedNode]


async def comfy_entrypoint() -> SeedModelExtension:
    return SeedModelExtension()
