# ComfyUI Doubao Seed 1.6 自定义节点说明

本目录提供一套基于火山方舟 Responses API 的 Doubao Seed 1.6 集成节点，支持 256k 上下文、深度思考模式与缓存能力，可在 ComfyUI 工作流中快速调用字节的文本/多模态模型。

## 目录结构
- `__init__.py`：将该节点包注册到 ComfyUI。
- `nodes_bytedance_seed.py`：核心实现，封装请求参数验证、API 调用与结果处理。
- `README.md`：使用说明与参考信息（本文档）。
- `CHANGELOG.md`：版本变更记录。

## 节点概览
### ByteDance Seed 1.6 (Custom)
统一节点覆盖 Chat / Lite / Flash / Vision 四种模型，通过 `Model Preset` 下拉框或 `Custom Model` 输入切换。内部自动拼装 Responses API 调用，输出以下内容：
1. **Assistant Text**：主回复文本。
2. **Response ID**：可回填到下一次调用的 `Previous Response ID`，实现多轮对话或树状分支。
3. **Reasoning Summary**：若模型返回思维链摘要，会在此输出。
4. **Raw Response JSON**：完整的 Responses API JSON，便于调试或提取结构化结果。

### 预设模型
| 预设名称 | 默认模型 ID | 说明 |
| --- | --- | --- |
| Seed 1.6 (Chat) | `doubao-seed-1-6-251015` | 通用深度思考聊天模型 |
| Seed 1.6 Lite | `doubao-seed-1-6-lite-251015` | 性价比版本，适合大规模问答 |
| Seed 1.6 Flash | `doubao-seed-1-6-flash-250828` | 低时延快速响应 |
| Seed 1.6 Vision | `doubao-seed-1-6-vision-250815` | 支持文本+图像输入的一体化模型 |

如需最新版本或自建 Endpoint，可在 `Custom Model` 中填入模型 ID 或 `ep-xxxx`。

## 输入参数
- **连接配置**
  - `Custom Base URL`：默认 `https://ark.cn-beijing.volces.com/api/v3`。
  - `Custom API Key`：火山方舟控制台获取的 API Key。
  - `Model Preset`：选择上述预设之一。
  - `Custom Model` *(可选)*：覆盖预设模型 ID。
- **推理内容**
  - `System Prompt` / `User Prompt`：文本消息，至少提供其中一个或输入图像。
  - `Primary Image` *(可选)*：ComfyUI Image Tensor，会自动转为 `input_image`。
- **上下文控制**
  - `Previous Response ID`：连接历史轮次或缓存的 response。
  - `Thinking Mode`：`model_default` / `enabled` / `disabled` / `auto`（若模型不支持 `auto` 会返回 API 错误，便于排查）。
  - `Store History`：是否在 Responses 侧保存本轮结果（默认开启）。
  - `Use Cache`：是否写入缓存，提高重复提示的命中率。
- **采样与截断**
  - `Max Output Tokens`、`Temperature`、`Top P`、`Top K`、`Stop Sequences`、`Metadata JSON` 等，与 Responses API 对应字段一致；留空即使用默认值。

## 使用建议
- **可选图像**：`Primary Image` 默认可为空，Vision 预设提供图像可获得多模态理解；纯文本任务保持为空即可。
- **模型权限**：若报错 `InvalidEndpointOrModel.NotFound`，请确认账号是否开通对应模型或改填自定义 Endpoint。
- **上下文管理**：结合 `Response ID`、`Store History`、`Use Cache` 能实现树状分支、窗口截断与低成本复用。
- **思考模式**：`auto` 会原样传给 API，某些模型可能返回 `Unsupported thinking type`，根据实际需求切换或忽略。
- **节点更新**：升级到新版后建议在工作流中重新放置节点，以同步最新输入签名。

## 参考文档
- Doubao Seed 1.6：<https://www.volcengine.com/docs/82379/1593702>
- Doubao Seed 1.6 Flash：<https://www.volcengine.com/docs/82379/1593704>
- Doubao Seed 1.6 Lite：<https://www.volcengine.com/docs/82379/1874969>
- Doubao Seed 1.6 Vision：<https://www.volcengine.com/docs/82379/1799865>
- Responses API 教程：<https://www.volcengine.com/docs/82379/1585128>

如需进一步扩展（工具调用、批量推理等），可参考上述官方文档或在 `nodes_bytedance_seed.py` 基础上自定义扩展。
