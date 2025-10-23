# MyNodes\_Win — Legacy ByteDance Nodes for Windows Portable Builds

This package re‑implements the ByteDance custom nodes using the *classic* ComfyUI
interface (`NODE_CLASS_MAPPINGS`, `INPUT_TYPES`, etc.). It is intended for Windows
users who stay on ComfyUI v0.3.64 or other builds that do not ship the new
`comfy_api` extension system.

## Installation

1. Copy this folder into your ComfyUI installation so the path becomes:
   ```
   <ComfyUI root>/custom_nodes/MyNodes_Win/
   ```
   Ensure there is no extra nested directory level.

2. Install runtime dependencies **inside the same Python environment ComfyUI uses**:
- If you plan to use video nodes, ensure `imageio` and `imageio-ffmpeg` are installed so the node can decode returned videos.

   - Portable build (`python_embeded` present):
     ```cmd
     python_embeded\python.exe -m pip install requests Pillow numpy imageio imageio-ffmpeg
     ```
   - Virtual environment (`.venv` present):
     ```cmd
     .\.venv\Scripts\python.exe -m pip install requests Pillow numpy imageio imageio-ffmpeg
     ```
   - System Python: run `python -m pip install requests Pillow numpy imageio imageio-ffmpeg`.

3. Restart ComfyUI. The node will appear in the node list under
   `MyNodes / image / ByteDance Image (Custom - Legacy)`.

## Available Node

| Node ID                     | Display Name                            | Category        | Description                                     |
| --------------------------- | --------------------------------------- | --------------- | ----------------------------------------------- |
| `ByteDanceImageCustomWin`   | ByteDance Image (Custom - Legacy)       | `MyNodes/image` | Text-to-image generation via ByteDance API      |

The inputs mirror the original custom node: Base URL, API Key, Model/Endpoint ID,
prompt, size preset/custom dimensions, seed, guidance scale, and watermark toggle.

## Troubleshooting

- **Node does not show up**: check the ComfyUI console. Import errors (e.g. missing
  `requests` or `Pillow`) will be printed there. Install the missing package in the
  ComfyUI environment and restart.
- **“Failed to reach ByteDance API”**: confirm `Custom Base URL` / `API Key` /
  `Model` are correct and reachable from the machine.
- **Need more nodes (video, editing, Seed models)**: follow the same pattern and
  add new legacy-style classes to this folder. The modern implementation in
  `MyNodes/` can be used as a reference for request payloads and validation logic.

Feel free to extend the package and add additional nodes as required.
