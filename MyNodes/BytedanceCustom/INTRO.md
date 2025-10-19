# MyNodes ByteDance 自定义节点简介

本文档快速介绍 `MyNodes ByteDance` 系列节点的能力与使用方式，帮助你在 ComfyUI 中接入 ByteDance/BytePlus 的在线推理服务。

## 核心能力

- **灵活的连接配置**：通过 `Custom Base URL / API Key / Custom Model` 即可对接任意自建 Endpoint，无需修改代码。
- **图像 & 视频全流程覆盖**：包含文生图、图生图、Seedream 4、文生视频、图/首尾帧/多参考图生成视频等常用场景。
- **内置日志与错误提示**：所有节点在提交、轮询、失败时都会输出关键日志，便于排查授权、参数或网络问题。
- **直接上传数据**：统一使用 Data URI 方式发送图片/视频帧，避免依赖 ComfyUI 的文件存储接口，适合纯线上部署。

## 节点列表

所有节点位于 `MyNodes` -> `ByteDance` 分类：

| 节点名称 | 功能说明 |
| --- | --- |
| `ByteDance Image (Custom)` | 文生图生成，支持尺寸、随机种子、Guidance、Watermark 控制 |
| `ByteDance Image Edit (Custom)` | 图生图/编辑，上传一张图片并给出指令 |
| `ByteDance Seedream 4 (Custom)` | Seedream 4 高分辨率生成，可选多图参考、分组生成等高级参数 |
| `ByteDance Text to Video (Custom)` | 纯文本生成视频，支持分辨率、时长、镜头固定、种子等控制项 |
| `ByteDance Image to Video (Custom)` | 单图生成视频，用于 First Frame 场景 |
| `ByteDance First-Last-Frame to Video (Custom)` | 首尾帧驱动的视频生成 |
| `ByteDance Reference Images to Video (Custom)` | 多参考图生成，用于角色/场景保持 |

## 使用步骤

1. **填写连接信息（必填）**
   - `Custom Base URL`：如 `https://ark.cn-beijing.volces.com/api/v3`
   - `Custom API Key`：对应 Endpoint 的密钥（以 Bearer Token 下发）
   - `Custom Model`：在 ByteDance 控制台获取的模型或 Endpoint ID，例如 `ep-20250827141608-b2l69`
2. **配置节点参数**：根据不同节点补充 Prompt、分辨率、时长、参考图等业务参数。
3. **预览/保存**：将输出连到 `PreviewImage`/`PreviewVideo` 或 `SaveImage`/`SaveVideo` 节点，即可在界面查看结果。

## 日志与排错

- 所有节点在执行时会打印 `Submitting video task...`、`Executing image generation...` 等调试信息，失败时会输出原始异常文本。
- 遇到 `401 Unauthorized`：确认 API Key、Base URL 是否匹配；ByteDance 平台通常需要在控制台启用对应 Endpoint。
- 遇到 `404 Not Found`：Base URL 或 Endpoint 路径错误；检查是否包含 `/api/v3` 等完整前缀。
- 遇到 `invalid literal for int()`：旧工作流输入顺序未更新。重新拖入节点或按新版顺序填写（连接信息位于最前）。

## 版本记录

- 2025-10-19
  - 首次发布 MyNodes 版本，支持自定义连接参数、统一日志、Data URI 上传等功能。
  - 将节点分类切换至 `MyNodes`，便于与官方节点区分。
  - 详见 `CHANGELOG.md` 获取完整改动列表。

如需扩展更多 ByteDance 接口，可参考 `nodes_bytedance_custom.py` 中的模式实现新的节点类。欢迎在当前目录提交补充文档或示例工作流。 
