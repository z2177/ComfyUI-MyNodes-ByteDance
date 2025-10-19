# ComfyUI ByteDance 自定义节点说明

本目录提供一组可直接配置的 ByteDance/BytePlus API 节点，支持在 ComfyUI 中填写自定义 Base URL、API Key、模型/Endpoint ID，快速完成在线推理接入。

## 目录结构

- `nodes_bytedance_custom.py`：节点主实现，涵盖图像与视频生成逻辑。
- `__init__.py`：导出 `comfy_entrypoint`，供 ComfyUI 自动加载。
- `INTRO.md`：面向业务使用者的快速上手说明。
- `CHANGELOG.md`：版本变更记录。

> 使用方式：将 `BytedanceCustom` 文件夹放入 ComfyUI 的 `custom_nodes/MyNodes/` 下，并确保 `custom_nodes/MyNodes/__init__.py` 会加载该扩展。

## 节点列表

所有节点位于前端的 `MyNodes` -> `image` / `video` 分类下：

| 节点名称 | 功能说明 |
| --- | --- |
| `ByteDance Image (Custom)` | 文生图，支持尺寸、种子、Guidance、Watermark 控制 |
| `ByteDance Image Edit (Custom)` | 图生图/编辑，上传图片并给出指令 |
| `ByteDance Seedream 4 (Custom)` | Seedream 4 文生图/图生图，高分辨率与分组生成支持 |
| `ByteDance Text to Video (Custom)` | 文生视频，可配置分辨率、时长、种子、镜头固定等 |
| `ByteDance Image to Video (Custom)` | 单图生成视频，用于 First Frame 场景 |
| `ByteDance First-Last-Frame to Video (Custom)` | 首尾帧驱动的视频生成 |
| `ByteDance Reference Images to Video (Custom)` | 多参考图生成视频，适合角色/风格保持 |

更多参数与日志策略说明参见 `INTRO.md`。

## 快速使用步骤

1. **填写连接信息（必填）**
   - `Custom Base URL`：例如 `https://ark.cn-beijing.volces.com/api/v3`
   - `Custom API Key`：ByteDance 控制台生成的密钥，将作为 Bearer Token 使用
   - `Custom Model`：模型或 Endpoint ID，如 `ep-2025XXXX...`
2. **配置业务参数**：按照官方节点习惯调整 Prompt、分辨率、时长、参考图等。
3. **连接预览/保存节点**：视频输出接 `PreviewVideo`/`SaveVideo`，图片输出接 `PreviewImage`/`SaveImage`。
4. **运行并观察日志**：命令行会输出提交/轮询/异常信息，便于排查。

## 常见问题

- **401 Unauthorized**：确认 API Key 与 Base URL 匹配，平台是否已启用对应 Endpoint。
- **404 Not Found**：通常是 Base URL 缺少路径（如 `/api/v3`），或 Endpoint ID 填写错误。
- **旧工作流参数错位**：新版节点把连接信息放在最前面，如出现 `invalid literal for int()` 等报错，请重新拖入节点或按新顺序填写。
- **节点不显示**：确保 `custom_nodes/MyNodes/__init__.py` 已导出 ByteDance 扩展，修改后需重启 ComfyUI。

## 版本记录

详见 `CHANGELOG.md`，其中列出了与官方 `comfy_api_nodes/nodes_bytedance.py` 的差异、日志增强和分类调整等内容。

