# 更新日志

## 2025-10-19

- 以官方 `comfy_api_nodes/nodes_bytedance.py` 为基础，抽取并改写为可自定义 Base URL / API Key / 模型 ID 的节点。
- 新增 `_sanitize_base_url`、`_build_auth_kwargs`、`_prepare_request_context` 等工具函数，统一校验连接参数。
- 上传流程改为使用 Data URI，支持直接将图片/帧发送给 ByteDance 接口，无需依赖 ComfyUI 的存储 API。
- 增强日志：提交任务、轮询、错误处理均输出详细日志，方便排错。
- 视频生成节点统一使用 `_compose_video_prompt` 生成控制指令，并兼容可选种子/镜头固定等参数。
- 节点分类归入 `MyNodes`，名称统一以 “ByteDance … (Custom)” 展示，便于与官方节点区分。
- 更新 README / INTRO 文档，说明连接方式、使用步骤与常见问题。

