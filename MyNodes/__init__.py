from comfy_api.latest import ComfyExtension, IO

from .BytedanceCustom.nodes_bytedance_custom import comfy_entrypoint as bytedance_entrypoint
from .BytedanceCustom_SeedModel.nodes_bytedance_seed import comfy_entrypoint as seed_entrypoint


class MyNodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        nodes: list[type[IO.ComfyNode]] = []

        bytedance_extension = await bytedance_entrypoint()
        nodes.extend(await bytedance_extension.get_node_list())

        seed_extension = await seed_entrypoint()
        nodes.extend(await seed_extension.get_node_list())

        return nodes


async def comfy_entrypoint() -> ComfyExtension:
    return MyNodesExtension()


__all__ = ["comfy_entrypoint"]
