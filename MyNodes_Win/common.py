"""Utility helpers shared by legacy ByteDance nodes."""

from __future__ import annotations

import base64
import importlib
import io
import json
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

LOGGER = logging.getLogger(__name__)

BYTEPLUS_IMAGE_ENDPOINT = 'images/generations'
BYTEPLUS_TASK_ENDPOINT = 'contents/generations/tasks'
BYTEPLUS_TASK_STATUS_ENDPOINT = 'contents/generations/tasks'
BYTEPLUS_RESPONSES_ENDPOINT = 'responses'


class DependencyError(ImportError):
    """Raised when a required runtime dependency is missing."""


def _lazy_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover
        raise DependencyError(
            f"The Python package '{module_name}' is required for ByteDance legacy nodes. "
            "Install it inside the ComfyUI environment."
        ) from exc


def sanitize_base_url(base_url: str) -> str:
    candidate = (base_url or "").strip()
    if not candidate:
        raise ValueError("Custom Base URL cannot be empty.")
    parsed = urlparse(candidate)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Custom Base URL must include scheme and host, e.g., https://example.com/api/v3")
    return candidate.rstrip("/") + "/"


def build_headers(api_key: str) -> Dict[str, str]:
    token = (api_key or "").strip()
    if not token:
        raise ValueError("Custom API Key cannot be empty.")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


def resolve_model_id(model: str) -> str:
    resolved = (model or "").strip()
    if not resolved:
        raise ValueError("Custom Model / Endpoint ID cannot be empty.")
    return resolved


def validate_string(value: Optional[str], *, min_length: int = 1) -> None:
    if value is None:
        raise ValueError("Prompt cannot be empty.")
    if len(value.strip()) < min_length:
        raise ValueError("Prompt cannot be empty.")


def tensor_to_data_uri(tensor, mime_type: str = "image/png") -> str:
    torch = _lazy_import("torch")
    np = _lazy_import("numpy")
    Image = _lazy_import("PIL.Image")

    if tensor is None:
        raise ValueError("Tensor must not be None.")
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = tensor.mul(255).clamp(0, 255).round().detach().cpu().to(torch.uint8)
    data = array.permute(1, 2, 0).numpy() if array.ndim == 3 else array.numpy()
    image = Image.fromarray(data)
    buffer = io.BytesIO()
    format_hint = "PNG" if mime_type.lower().endswith("png") else "JPEG"
    image.save(buffer, format=format_hint)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def encode_tensor_list(tensor, limit: int) -> List[str]:
    if tensor is None:
        return []
    count = tensor.size(0) if tensor.ndim == 4 else 1
    if count > limit:
        raise ValueError(f"A maximum of {limit} images is supported but {count} were provided.")
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return [tensor_to_data_uri(tensor[i : i + 1]) for i in range(tensor.shape[0])]


def tensor_from_base64(data: str):
    torch = _lazy_import("torch")
    np = _lazy_import("numpy")
    Image = _lazy_import("PIL.Image")

    if "," in data:
        data = data.split(",", 1)[1]
    image_bytes = base64.b64decode(data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(image).astype("float32") / 255.0
    return torch.from_numpy(arr)[None, ...]


def tensor_from_url(url: str, timeout: int = 120):
    requests = _lazy_import("requests")
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return tensor_from_base64("data:image/png;base64," + base64.b64encode(response.content).decode("ascii"))


def extract_first_image(data: Dict[str, Any]):
    items = data.get("data")
    if not isinstance(items, Iterable):
        raise RuntimeError("API response did not include any image data.")
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("b64_json"):
            return tensor_from_base64(str(item["b64_json"]))
        if item.get("url"):
            return tensor_from_url(str(item["url"]))
    raise RuntimeError("API response did not include a usable image payload.")


def extract_all_image_urls(data: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    items = data.get("data")
    if not isinstance(items, Iterable):
        return []
    for item in items:
        if isinstance(item, dict) and item.get("url"):
            urls.append(str(item["url"]))
    return urls


def post_json(base_url: str, path: str, headers: Dict[str, str], payload: Dict[str, Any], *, timeout: int = 180) -> Dict[str, Any]:
    requests = _lazy_import("requests")
    endpoint = urljoin(base_url, path)
    LOGGER.debug("POST %s payload=%s", endpoint, json.dumps(payload))
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:  # type: ignore[attr-defined]
        raise RuntimeError(f"Failed to reach ByteDance API: {exc}") from exc

    if response.status_code >= 400:
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = response.text
        raise RuntimeError(f"ByteDance API returned HTTP {response.status_code}: {error_payload}")

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError("ByteDance API returned a non-JSON response.") from exc

    if isinstance(body, dict) and body.get("error"):
        raise RuntimeError(f"ByteDance API error: {body['error'].get('message', 'Unknown error')}")

    return body


def poll_task(
    base_url: str,
    task_id: str,
    headers: Dict[str, str],
    *,
    poll_path: str,
    interval: float = 3.0,
    timeout: float = 900.0,
    success_status: Tuple[str, ...] = ("succeeded",),
    failure_status: Tuple[str, ...] = ("failed", "cancelled"),
) -> Dict[str, Any]:
    requests = _lazy_import("requests")
    start = time.time()
    url = urljoin(base_url, f"{poll_path}/{task_id}")
    while True:
        try:
            response = requests.get(url, headers=headers, timeout=30)
        except requests.RequestException as exc:  # type: ignore[attr-defined]
            raise RuntimeError(f"Failed to poll task status: {exc}") from exc

        if response.status_code >= 400:
            try:
                error_payload = response.json()
            except ValueError:
                error_payload = response.text
            raise RuntimeError(f"Polling failed with HTTP {response.status_code}: {error_payload}")

        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError("Task status response is not JSON.") from exc

        status = str(body.get("status", "")).lower()
        if status in success_status:
            return body
        if status in failure_status:
            raise RuntimeError(f"ByteDance task failed with status {status}: {body}")

        if time.time() - start > timeout:
            raise TimeoutError(f"Timed out waiting for task {task_id} to finish.")

        time.sleep(interval)


def concat_tensors(urls: Iterable[str]):
    torch = _lazy_import("torch")
    tensors = [tensor_from_url(url) for url in urls]
    if not tensors:
        raise RuntimeError("No image URLs returned from ByteDance API.")
    return torch.cat(tensors)


def count_images(tensor) -> int:
    if tensor is None:
        return 0
    return tensor.size(0) if tensor.ndim == 4 else 1


def ensure_single_image(tensor) -> None:
    count = count_images(tensor)
    if count != 1:
        raise ValueError(f"Exactly one input image is required, received {count}.")


def validate_aspect_ratio_range(
    tensor,
    min_ratio: Tuple[int, int],
    max_ratio: Tuple[int, int],
    *,
    strict: bool = True,
) -> float:
    ensure_single_image(tensor)
    image = tensor[0] if tensor.ndim == 4 else tensor
    if image.ndim == 3:
        _, h, w = image.shape
    else:
        h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image dimensions: {w}x{h}")
    a1, b1 = min_ratio
    a2, b2 = max_ratio
    if a1 <= 0 or b1 <= 0 or a2 <= 0 or b2 <= 0:
        raise ValueError("Ratios must be positive, e.g. (1, 4) or (4, 1).")
    lo = a1 / b1
    hi = a2 / b2
    if lo > hi:
        lo, hi = hi, lo
        a1, b1, a2, b2 = a2, b2, a1, b1
    ratio = w / h
    allowed = (lo < ratio < hi) if strict else (lo <= ratio <= hi)
    if not allowed:
        op = "<" if strict else "≤"
        raise ValueError(
            f"Image aspect ratio {ratio:.6g} outside allowed range: {a1}:{b1} {op} ratio {op} {a2}:{b2}."
        )
    return ratio



def download_video_output(url: str, timeout: int = 600):
    requests = _lazy_import("requests")
    imageio = _lazy_import("imageio.v2")
    torch = _lazy_import("torch")
    np = _lazy_import("numpy")

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    with imageio.get_reader(io.BytesIO(response.content), format="ffmpeg") as reader:  # type: ignore[attr-defined]
        meta = reader.get_meta_data() or {}
        fps = float(meta.get("fps", 24.0))
        frames = []
        for frame in reader:
            arr = np.asarray(frame).astype("float32") / 255.0
            frames.append(torch.from_numpy(arr))

    if not frames:
        raise RuntimeError("No frames decoded from downloaded video.")

    video_tensor = torch.stack(frames, dim=0)
    return {"frames": video_tensor, "fps": fps, "audio": None}


def validate_image_dimensions(tensor, *, min_width: int, min_height: int, max_width: int, max_height: int) -> None:
    torch = _lazy_import("torch")
    if tensor is None:
        raise ValueError("Image tensor is required.")
    images = tensor if tensor.ndim == 4 else tensor.unsqueeze(0)
    for img in images:
        _, h, w = img.shape if img.ndim == 3 else (0, img.shape[0], img.shape[1])
        if w < min_width or h < min_height or w > max_width or h > max_height:
            raise ValueError(
                f"Image dimensions {w}x{h} outside supported range {min_width}-{max_width} x {min_height}-{max_height}."
            )


def ensure_image_count_between(tensor, minimum: int, maximum: int) -> None:
    count = count_images(tensor)
    if count < minimum or count > maximum:
        raise ValueError(f"Expected between {minimum} and {maximum} images, received {count}.")
