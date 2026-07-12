from pathlib import Path
import importlib

class PluginManager:
    def __init__(self):
        self.loaded = {}

    def discover(self):
        for file in Path("plugins").glob("*.py"):
            if file.stem in ("__init__", "manager"):
                continue
            self.loaded[file.stem] = importlib.import_module(
                f"plugins.{file.stem}"
            )

    def start(self):
        for name, module in self.loaded.items():
            plugin_name = getattr(module, "NAME", name)
            print(f"Plugin loaded: {plugin_name}")
