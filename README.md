# ComfyUI MyNodes - ByteDance 节点合集

这个轻量化仓库仅包含两个自定义节点包：

- `MyNodes/BytedanceCustom/`：ByteDance Seedance/Seedream等 自定义节点，可配置 Base URL、API Key、模型/Endpoint ID。
- `MyNodes/BytedanceCustom_SeedModel/`：ByteDance Seed 1.6系列模型的节点扩展，可按需加载。

## 使用方式

1. 将整个 `MyNodes/` 目录拷贝到 ComfyUI 的 `custom_nodes/` 目录下：
   ```bash
   cp -R MyNodes /path/to/ComfyUI/custom_nodes/
   ```
2. 确保 ComfyUI 的 `custom_nodes/MyNodes/__init__.py` 已导出 `comfy_entrypoint`。
3. 重启 ComfyUI，节点会出现在前端的 `MyNodes` -> `image` / `video` 分类中，名称为 `ByteDance … (Custom)`。
4. 按 `MyNodes/BytedanceCustom/README.md` / `INTRO.md` 的说明填写 Base URL、API Key、模型 ID 等参数即可使用。

更多细节（参数说明、日志策略、常见问题）可查看 `MyNodes/BytedanceCustom` 目录内的文档。

