"""CineStyle custom node package with automatic V3 node discovery."""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType

from comfy_api.latest import ComfyExtension, io


_LOGGER = logging.getLogger(__name__)
_PACKAGE_DIR = Path(__file__).resolve().parent
_NODE_MODULES: list[ModuleType] = []
_V3_ENTRYPOINTS = []


def _load_node_modules() -> None:
    """Import every Python node module under ``py`` in stable filename order."""
    node_dir = _PACKAGE_DIR / "py"
    for path in sorted(node_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module_name = f"{__name__}._py_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception:
            _LOGGER.exception("Failed to load CineStyle node module %s", path)
            continue
        _NODE_MODULES.append(module)
        entrypoint = getattr(module, "comfy_entrypoint", None)
        if callable(entrypoint):
            _V3_ENTRYPOINTS.append(entrypoint)


class _ScannedExtension(ComfyExtension):
    """Aggregate extensions exported by the modules discovered in ``py``."""

    def __init__(self, entrypoints):
        self._entrypoints = list(entrypoints)
        self._extensions: list[ComfyExtension] = []

    async def on_load(self) -> None:
        from server import PromptServer

        server_instance = getattr(PromptServer, "instance", None)
        preview_cache = sys.modules.get(f"{__name__}._py_preview_cache")
        if preview_cache is not None:
            register_routes = getattr(preview_cache, "register_wait_input_cache_routes", None)
            if callable(register_routes):
                register_routes(server_instance)
        self._extensions = []
        for entrypoint in self._entrypoints:
            extension = await entrypoint()
            if not isinstance(extension, ComfyExtension):
                raise TypeError(f"{entrypoint} did not return a ComfyExtension")
            await extension.on_load()
            self._extensions.append(extension)

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        nodes: list[type[io.ComfyNode]] = []
        for extension in self._extensions:
            nodes.extend(await extension.get_node_list())
        return nodes


_load_node_modules()


async def comfy_entrypoint() -> _ScannedExtension:
    return _ScannedExtension(_V3_ENTRYPOINTS)


WEB_DIRECTORY = "./web"
