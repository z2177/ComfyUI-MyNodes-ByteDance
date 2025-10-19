"""Custom ByteDance API nodes for ComfyUI.

Each node proxies requests through ByteDance/BytePlus endpoints supplied by the user
at runtime (base URL, API key, and model/endpoint ID). The implementation focuses on
clear validation and logging so that workflow authors receive actionable feedback
when a request fails.
"""

import logging
import math
from enum import Enum
from typing import Literal, Optional, Type, Union
from typing_extensions import override
from urllib.parse import urlparse

import torch
from pydantic import BaseModel, Field

from comfy_api.latest import ComfyExtension, IO
from comfy_api_nodes.util.validation_utils import (
    validate_image_aspect_ratio_range,
    get_number_of_images,
    validate_image_dimensions,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    EmptyRequest,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    T,
)
from comfy_api_nodes.apinode_utils import (
    download_url_to_image_tensor,
    download_url_to_video_output,
    validate_string,
    tensor_to_data_uri,
)

LOGGER = logging.getLogger(__name__)


BYTEPLUS_IMAGE_ENDPOINT = "/images/generations"

# Long-running tasks endpoints(e.g., video)
BYTEPLUS_TASK_ENDPOINT = "/contents/generations/tasks"
BYTEPLUS_TASK_STATUS_ENDPOINT = "/contents/generations/tasks"  # + /{task_id}


def _connection_inputs() -> list[IO.String.Input]:
    return [
        IO.String.Input(
            "custom_base_url",
            display_name="Custom Base URL",
            tooltip="ByteDance API 根路径，例如 https://ark.cn-beijing.volces.com/api/v3",
        ),
        IO.String.Input(
            "custom_api_key",
            display_name="Custom API Key",
            tooltip="用于鉴权的 API Key，将作为 Bearer Token 发送。",
        ),
        IO.String.Input(
            "custom_model",
            display_name="Custom Model",
            tooltip="ByteDance 控制台上的模型或 Endpoint ID（例如 ep-***）。",
        ),
    ]


def _sanitize_base_url(base_url: Optional[str]) -> str:
    if not base_url or not base_url.strip():
        raise ValueError("Custom base URL is required.")
    candidate = base_url.strip()
    parsed = urlparse(candidate)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Custom base URL must include scheme and host, e.g., https://example.com.")
    return candidate.rstrip("/") + "/"  # ensure directory semantics for urljoin


def _build_auth_kwargs(custom_api_key: Optional[str]) -> dict[str, Optional[str]]:
    key = (custom_api_key or "").strip()
    if not key:
        raise ValueError("Custom API Key is required.")
    return {
        "auth_token": key,
        "comfy_api_key": None,
    }


def _prepare_request_context(
    *,
    custom_base_url: str,
    custom_api_key: str,
    custom_model: str,
) -> tuple[str, dict[str, Optional[str]], str]:
    """Validate connection inputs and return sanitized base URL, auth kwargs, and model ID."""
    model_id = (custom_model or "").strip()
    if not model_id:
        raise ValueError("Custom model cannot be empty.")
    api_base = _sanitize_base_url(custom_base_url)
    auth_kwargs = _build_auth_kwargs(custom_api_key)
    return api_base, auth_kwargs, model_id


def _compose_video_prompt(
    base_prompt: str,
    *,
    resolution: str,
    aspect_ratio: str,
    duration: int,
    seed: Optional[int],
    camera_fixed: Optional[bool],
    watermark: Optional[bool],
) -> str:
    """Compose the control string expected by the ByteDance video endpoints."""
    parts: list[str] = [base_prompt.strip()]
    parts.append(f"--resolution {resolution}")
    parts.append(f"--ratio {aspect_ratio}")
    parts.append(f"--duration {duration}")
    if seed is not None:
        parts.append(f"--seed {seed}")
    cf_value = str(bool(camera_fixed)).lower() if camera_fixed is not None else "false"
    parts.append(f"--camerafixed {cf_value}")
    watermark_value = str(bool(watermark)).lower() if watermark is not None else "true"
    parts.append(f"--watermark {watermark_value}")
    return " ".join(parts).strip()


class Text2ImageModelName(str, Enum):
    seedream_3 = "seedream-3-0-t2i-250415"


class Image2ImageModelName(str, Enum):
    seededit_3 = "seededit-3-0-i2i-250628"


class Text2VideoModelName(str, Enum):
    seedance_1_pro  = "seedance-1-0-pro-250528"
    seedance_1_lite = "seedance-1-0-lite-t2v-250428"


class Image2VideoModelName(str, Enum):
    """note(August 31): Pro model only supports FirstFrame: https://docs.byteplus.com/en/docs/ModelArk/1520757"""
    seedance_1_pro  = "seedance-1-0-pro-250528"
    seedance_1_lite = "seedance-1-0-lite-i2v-250428"


class Text2ImageTaskCreationRequest(BaseModel):
    model: str = Field(Text2ImageModelName.seedream_3.value)
    prompt: str = Field(...)
    response_format: Optional[str] = Field("url")
    size: Optional[str] = Field(None)
    seed: Optional[int] = Field(0, ge=0, le=2147483647)
    guidance_scale: Optional[float] = Field(..., ge=1.0, le=10.0)
    watermark: Optional[bool] = Field(True)


class Image2ImageTaskCreationRequest(BaseModel):
    model: str = Field(Image2ImageModelName.seededit_3.value)
    prompt: str = Field(...)
    response_format: Optional[str] = Field("url")
    image: str = Field(..., description="Base64 encoded string or image URL")
    size: Optional[str] = Field("adaptive")
    seed: Optional[int] = Field(..., ge=0, le=2147483647)
    guidance_scale: Optional[float] = Field(..., ge=1.0, le=10.0)
    watermark: Optional[bool] = Field(True)


class Seedream4Options(BaseModel):
    max_images: int = Field(15)


class Seedream4TaskCreationRequest(BaseModel):
    model: str = Field("seedream-4-0-250828")
    prompt: str = Field(...)
    response_format: str = Field("url")
    image: Optional[list[str]] = Field(None, description="Image URLs")
    size: str = Field(...)
    seed: int = Field(..., ge=0, le=2147483647)
    sequential_image_generation: str = Field("disabled")
    sequential_image_generation_options: Seedream4Options = Field(Seedream4Options(max_images=15))
    watermark: bool = Field(True)


class ImageTaskCreationResponse(BaseModel):
    model: str = Field(...)
    created: int = Field(..., description="Unix timestamp (in seconds) indicating time when the request was created.")
    data: list = Field([], description="Contains information about the generated image(s).")
    error: dict = Field({}, description="Contains `code` and `message` fields in case of error.")


class TaskTextContent(BaseModel):
    type: str = Field("text")
    text: str = Field(...)


class TaskImageContentUrl(BaseModel):
    url: str = Field(...)


class TaskImageContent(BaseModel):
    type: str = Field("image_url")
    image_url: TaskImageContentUrl = Field(...)
    role: Optional[Literal["first_frame", "last_frame", "reference_image"]] = Field(None)


class Text2VideoTaskCreationRequest(BaseModel):
    model: str = Field(Text2VideoModelName.seedance_1_pro.value)
    content: list[TaskTextContent] = Field(..., min_length=1)


class Image2VideoTaskCreationRequest(BaseModel):
    model: str = Field(Image2VideoModelName.seedance_1_pro.value)
    content: list[Union[TaskTextContent, TaskImageContent]] = Field(..., min_length=2)


class TaskCreationResponse(BaseModel):
    id: str = Field(...)


class TaskStatusError(BaseModel):
    code: str = Field(...)
    message: str = Field(...)


class TaskStatusResult(BaseModel):
    video_url: str = Field(...)


class TaskStatusResponse(BaseModel):
    id: str = Field(...)
    model: str = Field(...)
    status: Literal["queued", "running", "cancelled", "succeeded", "failed"] = Field(...)
    error: Optional[TaskStatusError] = Field(None)
    content: Optional[TaskStatusResult] = Field(None)


RECOMMENDED_PRESETS = [
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
]

RECOMMENDED_PRESETS_SEEDREAM_4 = [
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
]

# The time in this dictionary are given for 10 seconds duration.
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


def get_image_url_from_response(response: ImageTaskCreationResponse) -> str:
    if response.error:
        error_msg = f"ByteDance request failed. Code: {response.error['code']}, message: {response.error['message']}"
        LOGGER.error(error_msg)
        raise RuntimeError(error_msg)
    if not response.data:
        raise RuntimeError("ByteDance request succeeded but returned no image data.")
    image_url = response.data[0].get("url")
    if not image_url:
        raise RuntimeError("ByteDance request succeeded but image URL is missing in the response payload.")
    LOGGER.debug("ByteDance task succeeded, image URL: %s", image_url)
    return image_url


def get_video_url_from_task_status(response: TaskStatusResponse) -> Union[str, None]:
    """Returns the video URL from the task status response if it exists."""
    if hasattr(response, "content") and response.content:
        return response.content.video_url
    return None


async def poll_until_finished(
    auth_kwargs: dict[str, str],
    task_id: str,
    estimated_duration: Optional[int] = None,
    node_id: Optional[str] = None,
    api_base: Optional[str] = None,
) -> TaskStatusResponse:
    """Polls the ByteDance API endpoint until the task reaches a terminal state, then returns the response."""
    LOGGER.debug("Polling task %s at %s", task_id, api_base)
    return await PollingOperation(
        poll_endpoint=ApiEndpoint(
            path=f"{BYTEPLUS_TASK_STATUS_ENDPOINT}/{task_id}",
            method=HttpMethod.GET,
            request_model=EmptyRequest,
            response_model=TaskStatusResponse,
        ),
        completed_statuses=[
            "succeeded",
        ],
        failed_statuses=[
            "cancelled",
            "failed",
        ],
        status_extractor=lambda response: response.status,
        auth_kwargs=auth_kwargs,
        result_url_extractor=get_video_url_from_task_status,
        estimated_duration=estimated_duration,
        node_id=node_id,
        api_base=api_base,
    ).execute()


class CustomByteDanceImageNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CustomByteDanceImageNode",
            display_name="ByteDance Image (Custom)",
            category="MyNodes/image",
            description="Generate images using ByteDance models via api based on prompt",
            inputs=[
                *_connection_inputs(),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the image",
                ),
                IO.Combo.Input(
                    "size_preset",
                    options=[label for label, _, _ in RECOMMENDED_PRESETS],
                    tooltip="Pick a recommended size. Select Custom to use the width and height below",
                ),
                IO.Int.Input(
                    "width",
                    default=1024,
                    min=512,
                    max=2048,
                    step=64,
                    tooltip="Custom width for image. Value is working only if `size_preset` is set to `Custom`",
                ),
                IO.Int.Input(
                    "height",
                    default=1024,
                    min=512,
                    max=2048,
                    step=64,
                    tooltip="Custom height for image. Value is working only if `size_preset` is set to `Custom`",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation",
                    optional=True,
                ),
                IO.Float.Input(
                    "guidance_scale",
                    default=2.5,
                    min=1.0,
                    max=10.0,
                    step=0.01,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Higher value makes the image follow the prompt more closely",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=True,
                    tooltip="Whether to add an \"AI generated\" watermark to the image",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
            ],
            hidden=[],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
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
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        api_base, auth_kwargs, model_id = _prepare_request_context(
            custom_base_url=custom_base_url,
            custom_api_key=custom_api_key,
            custom_model=custom_model,
        )
        LOGGER.debug(
            "Executing ByteDance image generation (model=%s, size_preset=%s, seed=%s)",
            model_id,
            size_preset,
            seed,
        )
        w = h = None
        for label, tw, th in RECOMMENDED_PRESETS:
            if label == size_preset:
                w, h = tw, th
                break

        if w is None or h is None:
            w, h = width, height
            if not (512 <= w <= 2048) or not (512 <= h <= 2048):
                raise ValueError(
                    f"Custom size out of range: {w}x{h}. "
                    "Both width and height must be between 512 and 2048 pixels."
                )

        payload = Text2ImageTaskCreationRequest(
            model=model_id,
            prompt=prompt,
            size=f"{w}x{h}",
            seed=seed,
            guidance_scale=guidance_scale,
            watermark=watermark,
        )
        try:
            response = await SynchronousOperation(
                endpoint=ApiEndpoint(
                    path=BYTEPLUS_IMAGE_ENDPOINT,
                    method=HttpMethod.POST,
                    request_model=Text2ImageTaskCreationRequest,
                    response_model=ImageTaskCreationResponse,
                ),
                request=payload,
                auth_kwargs=auth_kwargs,
                api_base=api_base,
            ).execute()
        except Exception as exc:
            LOGGER.error("ByteDance image generation failed: %s", exc)
            raise
        return IO.NodeOutput(await download_url_to_image_tensor(get_image_url_from_response(response)))


class CustomByteDanceImageEditNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CustomByteDanceImageEditNode",
            display_name="ByteDance Image Edit (Custom)",
            category="MyNodes/image",
            description="Edit images using ByteDance models via api based on prompt",
            inputs=[
                *_connection_inputs(),
                IO.Image.Input(
                    "image",
                    tooltip="The base image to edit",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Instruction to edit image",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation",
                    optional=True,
                ),
                IO.Float.Input(
                    "guidance_scale",
                    default=5.5,
                    min=1.0,
                    max=10.0,
                    step=0.01,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Higher value makes the image follow the prompt more closely",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=True,
                    tooltip="Whether to add an \"AI generated\" watermark to the image",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
            ],
            hidden=[],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        custom_base_url: str,
        custom_api_key: str,
        custom_model: str,
        image: torch.Tensor,
        prompt: str,
        seed: int,
        guidance_scale: float,
        watermark: bool,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        api_base, auth_kwargs, model_id = _prepare_request_context(
            custom_base_url=custom_base_url,
            custom_api_key=custom_api_key,
            custom_model=custom_model,
        )
        LOGGER.debug("Executing ByteDance image edit (model=%s, seed=%s)", model_id, seed)
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        validate_image_aspect_ratio_range(image, (1, 3), (3, 1))
        source_url = tensor_to_data_uri(image, mime_type="image/png")
        payload = Image2ImageTaskCreationRequest(
            model=model_id,
            prompt=prompt,
            image=source_url,
            seed=seed,
            guidance_scale=guidance_scale,
            watermark=watermark,
        )
        try:
            response = await SynchronousOperation(
                endpoint=ApiEndpoint(
                    path=BYTEPLUS_IMAGE_ENDPOINT,
                    method=HttpMethod.POST,
                    request_model=Image2ImageTaskCreationRequest,
                    response_model=ImageTaskCreationResponse,
                ),
                request=payload,
                auth_kwargs=auth_kwargs,
                api_base=api_base,
            ).execute()
        except Exception as exc:
            LOGGER.error("ByteDance image edit failed: %s", exc)
            raise
        return IO.NodeOutput(await download_url_to_image_tensor(get_image_url_from_response(response)))


class CustomByteDanceSeedreamNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CustomByteDanceSeedreamNode",
            display_name="ByteDance Seedream 4 (Custom)",
            category="MyNodes/image",
            description="Unified text-to-image generation and precise single-sentence editing at up to 4K resolution.",
            inputs=[
                *_connection_inputs(),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for creating or editing an image.",
                ),
                IO.Image.Input(
                    "image",
                    tooltip="Input image(s) for image-to-image generation. "
                            "List of 1-10 images for single or multi-reference generation.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "size_preset",
                    options=[label for label, _, _ in RECOMMENDED_PRESETS_SEEDREAM_4],
                    tooltip="Pick a recommended size. Select Custom to use the width and height below.",
                ),
                IO.Int.Input(
                    "width",
                    default=2048,
                    min=1024,
                    max=4096,
                    step=64,
                    tooltip="Custom width for image. Value is working only if `size_preset` is set to `Custom`",
                    optional=True,
                ),
                IO.Int.Input(
                    "height",
                    default=2048,
                    min=1024,
                    max=4096,
                    step=64,
                    tooltip="Custom height for image. Value is working only if `size_preset` is set to `Custom`",
                    optional=True,
                ),
                IO.Combo.Input(
                    "sequential_image_generation",
                    options=["disabled", "auto"],
                    tooltip="Group image generation mode. "
                            "'disabled' generates a single image. "
                            "'auto' lets the model decide whether to generate multiple related images "
                            "(e.g., story scenes, character variations).",
                    optional=True,
                ),
                IO.Int.Input(
                    "max_images",
                    default=1,
                    min=1,
                    max=15,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Maximum number of images to generate when sequential_image_generation='auto'. "
                            "Total images (input + generated) cannot exceed 15.",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=True,
                    tooltip="Whether to add an \"AI generated\" watermark to the image.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "fail_on_partial",
                    default=True,
                    tooltip="If enabled, abort execution if any requested images are missing or return an error.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
            ],
            hidden=[],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        custom_base_url: str,
        custom_api_key: str,
        custom_model: str,
        image: torch.Tensor = None,
        size_preset: str = RECOMMENDED_PRESETS_SEEDREAM_4[0][0],
        width: int = 2048,
        height: int = 2048,
        sequential_image_generation: str = "disabled",
        max_images: int = 1,
        seed: int = 0,
        watermark: bool = True,
        fail_on_partial: bool = True,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        api_base, auth_kwargs, model_id = _prepare_request_context(
            custom_base_url=custom_base_url,
            custom_api_key=custom_api_key,
            custom_model=custom_model,
        )
        LOGGER.debug(
            "Executing Seedream-4 task (model=%s, size_preset=%s, sequential=%s)",
            model_id,
            size_preset,
            sequential_image_generation,
        )
        w = h = None
        for label, tw, th in RECOMMENDED_PRESETS_SEEDREAM_4:
            if label == size_preset:
                w, h = tw, th
                break

        if w is None or h is None:
            w, h = width, height
            if not (1024 <= w <= 4096) or not (1024 <= h <= 4096):
                raise ValueError(
                    f"Custom size out of range: {w}x{h}. "
                    "Both width and height must be between 1024 and 4096 pixels."
                )
        n_input_images = get_number_of_images(image) if image is not None else 0
        if n_input_images > 10:
            raise ValueError(f"Maximum of 10 reference images are supported, but {n_input_images} received.")
        if sequential_image_generation == "auto" and n_input_images + max_images > 15:
            raise ValueError(
                "The maximum number of generated images plus the number of reference images cannot exceed 15."
            )
        reference_images_urls = []
        if n_input_images:
            for i in image:
                validate_image_aspect_ratio_range(i, (1, 3), (3, 1))
            reference_images_urls = [
                tensor_to_data_uri(img, mime_type="image/png") for img in image[:n_input_images]
            ]
        payload = Seedream4TaskCreationRequest(
            model=model_id,
            prompt=prompt,
            image=reference_images_urls,
            size=f"{w}x{h}",
            seed=seed,
            sequential_image_generation=sequential_image_generation,
            sequential_image_generation_options=Seedream4Options(max_images=max_images),
            watermark=watermark,
        )
        try:
            response = await SynchronousOperation(
                endpoint=ApiEndpoint(
                    path=BYTEPLUS_IMAGE_ENDPOINT,
                    method=HttpMethod.POST,
                    request_model=Seedream4TaskCreationRequest,
                    response_model=ImageTaskCreationResponse,
                ),
                request=payload,
                auth_kwargs=auth_kwargs,
                api_base=api_base,
            ).execute()
        except Exception as exc:
            LOGGER.error("Seedream-4 request failed: %s", exc)
            raise

        if len(response.data) == 1:
            return IO.NodeOutput(await download_url_to_image_tensor(get_image_url_from_response(response)))
        urls = [str(d["url"]) for d in response.data if isinstance(d, dict) and "url" in d]
        if fail_on_partial and len(urls) < len(response.data):
            raise RuntimeError(f"Only {len(urls)} of {len(response.data)} images were generated before error.")
        return IO.NodeOutput(torch.cat([await download_url_to_image_tensor(i) for i in urls]))


class CustomByteDanceTextToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CustomByteDanceTextToVideoNode",
            display_name="ByteDance Text to Video (Custom)",
            category="MyNodes/video",
            description="Generate video using ByteDance models via api based on prompt",
            inputs=[
                *_connection_inputs(),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the video.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p", "1080p"],
                    tooltip="The resolution of the output video.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                    tooltip="The aspect ratio of the output video.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=12,
                    step=1,
                    tooltip="The duration of the output video in seconds.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "camera_fixed",
                    default=False,
                    tooltip="Specifies whether to fix the camera. The platform appends an instruction "
                            "to fix the camera to your prompt, but does not guarantee the actual effect.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=True,
                    tooltip="Whether to add an \"AI generated\" watermark to the video.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
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
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "camerafixed", "watermark"])
        api_base, auth_kwargs, model_id = _prepare_request_context(
            custom_base_url=custom_base_url,
            custom_api_key=custom_api_key,
            custom_model=custom_model,
        )
        seed_value = seed if seed is not None else None
        composed_prompt = _compose_video_prompt(
            prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration=duration,
            seed=seed_value,
            camera_fixed=camera_fixed,
            watermark=watermark,
        )
        LOGGER.debug(
            "Executing text-to-video request (model=%s, resolution=%s, aspect_ratio=%s)",
            model_id,
            resolution,
            aspect_ratio,
        )
        model_timings = VIDEO_TASKS_EXECUTION_TIME.get(model_id)
        estimated_duration = None
        if model_timings and resolution in model_timings:
            estimated_duration = max(1, math.ceil(model_timings[resolution] * (duration / 10.0)))
        return await process_video_task(
            request_model=Text2VideoTaskCreationRequest,
            payload=Text2VideoTaskCreationRequest(
                model=model_id,
                content=[TaskTextContent(text=composed_prompt)],
            ),
            auth_kwargs=auth_kwargs,
            node_id=cls.hidden.unique_id,
            estimated_duration=estimated_duration,
            api_base=api_base,
        )


class CustomByteDanceImageToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CustomByteDanceImageToVideoNode",
            display_name="ByteDance Image to Video (Custom)",
            category="MyNodes/video",
            description="Generate video using ByteDance models via api based on image and prompt",
            inputs=[
                *_connection_inputs(),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the video.",
                ),
                IO.Image.Input(
                    "image",
                    tooltip="First frame to be used for the video.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p", "1080p"],
                    tooltip="The resolution of the output video.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                    tooltip="The aspect ratio of the output video.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=12,
                    step=1,
                    tooltip="The duration of the output video in seconds.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "camera_fixed",
                    default=False,
                    tooltip="Specifies whether to fix the camera. The platform appends an instruction "
                            "to fix the camera to your prompt, but does not guarantee the actual effect.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=True,
                    tooltip="Whether to add an \"AI generated\" watermark to the video.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        custom_base_url: str,
        custom_api_key: str,
        custom_model: str,
        prompt: str,
        image: torch.Tensor,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        camera_fixed: bool,
        watermark: bool,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "camerafixed", "watermark"])
        api_base, auth_kwargs, model_id = _prepare_request_context(
            custom_base_url=custom_base_url,
            custom_api_key=custom_api_key,
            custom_model=custom_model,
        )
        validate_image_dimensions(image, min_width=300, min_height=300, max_width=6000, max_height=6000)
        validate_image_aspect_ratio_range(image, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5

        image_url = tensor_to_data_uri(image, mime_type="image/png")
        seed_value = seed if seed is not None else None
        composed_prompt = _compose_video_prompt(
            prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration=duration,
            seed=seed_value,
            camera_fixed=camera_fixed,
            watermark=watermark,
        )

        LOGGER.debug(
            "Executing image-to-video request (model=%s, resolution=%s, aspect_ratio=%s)",
            model_id,
            resolution,
            aspect_ratio,
        )
        return await process_video_task(
            request_model=Image2VideoTaskCreationRequest,
            payload=Image2VideoTaskCreationRequest(
                model=model_id,
                content=[
                    TaskTextContent(text=composed_prompt),
                    TaskImageContent(image_url=TaskImageContentUrl(url=image_url)),
                ],
            ),
            auth_kwargs=auth_kwargs,
            node_id=cls.hidden.unique_id,
            estimated_duration=(
                max(1, math.ceil(VIDEO_TASKS_EXECUTION_TIME[model_id][resolution] * (duration / 10.0))))
            if VIDEO_TASKS_EXECUTION_TIME.get(model_id) and resolution in VIDEO_TASKS_EXECUTION_TIME[model_id]
            else None,
            api_base=api_base,
        )


class CustomByteDanceFirstLastFrameNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CustomByteDanceFirstLastFrameNode",
            display_name="ByteDance First-Last-Frame to Video (Custom)",
            category="MyNodes/video",
            description="Generate video using prompt and first and last frames.",
            inputs=[
                *_connection_inputs(),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the video.",
                ),
                IO.Image.Input(
                    "first_frame",
                    tooltip="First frame to be used for the video.",
                ),
                IO.Image.Input(
                    "last_frame",
                    tooltip="Last frame to be used for the video.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p", "1080p"],
                    tooltip="The resolution of the output video.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                    tooltip="The aspect ratio of the output video.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=12,
                    step=1,
                    tooltip="The duration of the output video in seconds.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "camera_fixed",
                    default=False,
                    tooltip="Specifies whether to fix the camera. The platform appends an instruction "
                            "to fix the camera to your prompt, but does not guarantee the actual effect.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=True,
                    tooltip="Whether to add an \"AI generated\" watermark to the video.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        custom_base_url: str,
        custom_api_key: str,
        custom_model: str,
        prompt: str,
        first_frame: torch.Tensor,
        last_frame: torch.Tensor,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        camera_fixed: bool,
        watermark: bool,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "camerafixed", "watermark"])
        api_base, auth_kwargs, model_id = _prepare_request_context(
            custom_base_url=custom_base_url,
            custom_api_key=custom_api_key,
            custom_model=custom_model,
        )
        for i in (first_frame, last_frame):
            validate_image_dimensions(i, min_width=300, min_height=300, max_width=6000, max_height=6000)
            validate_image_aspect_ratio_range(i, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5

        first_frame_data = tensor_to_data_uri(first_frame, mime_type="image/png")
        last_frame_data = tensor_to_data_uri(last_frame, mime_type="image/png")

        seed_value = seed if seed is not None else None
        composed_prompt = _compose_video_prompt(
            prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration=duration,
            seed=seed_value,
            camera_fixed=camera_fixed,
            watermark=watermark,
        )
        LOGGER.debug(
            "Executing first-last-frame video request (model=%s, resolution=%s, aspect_ratio=%s)",
            model_id,
            resolution,
            aspect_ratio,
        )

        return await process_video_task(
            request_model=Image2VideoTaskCreationRequest,
            payload=Image2VideoTaskCreationRequest(
                model=model_id,
                content=[
                    TaskTextContent(text=composed_prompt),
                    TaskImageContent(image_url=TaskImageContentUrl(url=first_frame_data), role="first_frame"),
                    TaskImageContent(image_url=TaskImageContentUrl(url=last_frame_data), role="last_frame"),
                ],
            ),
            auth_kwargs=auth_kwargs,
            node_id=cls.hidden.unique_id,
            estimated_duration=(
                max(1, math.ceil(VIDEO_TASKS_EXECUTION_TIME[model_id][resolution] * (duration / 10.0))))
            if VIDEO_TASKS_EXECUTION_TIME.get(model_id) and resolution in VIDEO_TASKS_EXECUTION_TIME[model_id]
            else None,
            api_base=api_base,
        )


class CustomByteDanceImageReferenceNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CustomByteDanceImageReferenceNode",
            display_name="ByteDance Reference Images to Video (Custom)",
            category="MyNodes/video",
            description="Generate video using prompt and reference images.",
            inputs=[
                *_connection_inputs(),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the video.",
                ),
                IO.Image.Input(
                    "images",
                    tooltip="One to four images.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p"],
                    tooltip="The resolution of the output video.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                    tooltip="The aspect ratio of the output video.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=12,
                    step=1,
                    tooltip="The duration of the output video in seconds.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=True,
                    tooltip="Whether to add an \"AI generated\" watermark to the video.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        custom_base_url: str,
        custom_api_key: str,
        custom_model: str,
        prompt: str,
        images: torch.Tensor,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        watermark: bool,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        raise_if_text_params(prompt, ["resolution", "ratio", "duration", "seed", "watermark"])
        api_base, auth_kwargs, model_id = _prepare_request_context(
            custom_base_url=custom_base_url,
            custom_api_key=custom_api_key,
            custom_model=custom_model,
        )
        for image in images:
            validate_image_dimensions(image, min_width=300, min_height=300, max_width=6000, max_height=6000)
            validate_image_aspect_ratio_range(image, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5

        max_upload = min(get_number_of_images(images), 4)
        image_urls = [
            tensor_to_data_uri(images[idx], mime_type="image/png") for idx in range(max_upload)
        ]

        seed_value = seed if seed is not None else None
        composed_prompt = _compose_video_prompt(
            prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration=duration,
            seed=seed_value,
            camera_fixed=None,
            watermark=watermark,
        )
        x = [
            TaskTextContent(text=composed_prompt),
            *[TaskImageContent(image_url=TaskImageContentUrl(url=str(i)), role="reference_image") for i in image_urls]
        ]
        LOGGER.debug(
            "Executing reference-image video request (model=%s, resolution=%s, refs=%s)",
            model_id,
            resolution,
            len(image_urls),
        )
        return await process_video_task(
            request_model=Image2VideoTaskCreationRequest,
            payload=Image2VideoTaskCreationRequest(
                model=model_id,
                content=x,
            ),
            auth_kwargs=auth_kwargs,
            node_id=cls.hidden.unique_id,
            estimated_duration=(
                max(1, math.ceil(VIDEO_TASKS_EXECUTION_TIME[model_id][resolution] * (duration / 10.0))))
            if VIDEO_TASKS_EXECUTION_TIME.get(model_id) and resolution in VIDEO_TASKS_EXECUTION_TIME[model_id]
            else None,
            api_base=api_base,
        )


async def process_video_task(
    request_model: Type[T],
    payload: Union[Text2VideoTaskCreationRequest, Image2VideoTaskCreationRequest],
    auth_kwargs: dict,
    node_id: str,
    estimated_duration: Optional[int],
    api_base: Optional[str] = None,
) -> IO.NodeOutput:
    LOGGER.debug(
        "Submitting video task (model=%s, node_id=%s)",
        payload.model,
        node_id,
    )
    try:
        initial_response = await SynchronousOperation(
            endpoint=ApiEndpoint(
                path=BYTEPLUS_TASK_ENDPOINT,
                method=HttpMethod.POST,
                request_model=request_model,
                response_model=TaskCreationResponse,
            ),
            request=payload,
            auth_kwargs=auth_kwargs,
            api_base=api_base,
        ).execute()
    except Exception as exc:
        LOGGER.error("Video task submission failed: %s", exc)
        raise
    response = await poll_until_finished(
        auth_kwargs,
        initial_response.id,
        estimated_duration=estimated_duration,
        node_id=node_id,
        api_base=api_base,
    )
    LOGGER.debug("Video task %s completed with status response", initial_response.id)
    return IO.NodeOutput(await download_url_to_video_output(get_video_url_from_task_status(response)))


def raise_if_text_params(prompt: str, text_params: list[str]) -> None:
    for i in text_params:
        if f"--{i} " in prompt:
            raise ValueError(
                f"--{i} is not allowed in the prompt, use the appropriated widget input to change this value."
            )


class CustomByteDanceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            CustomByteDanceImageNode,
            CustomByteDanceImageEditNode,
            CustomByteDanceSeedreamNode,
            CustomByteDanceTextToVideoNode,
            CustomByteDanceImageToVideoNode,
            CustomByteDanceFirstLastFrameNode,
            CustomByteDanceImageReferenceNode,
        ]

async def comfy_entrypoint() -> CustomByteDanceExtension:
    return CustomByteDanceExtension()
