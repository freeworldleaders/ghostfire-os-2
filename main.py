from config.settings import SETTINGS
from runtime.engine import RuntimeEngine
from router.router import CommandRouter
from agents.registry import AgentRegistry
from plugins.manager import PluginManager

print(f"{SETTINGS['app_name']} {SETTINGS['version']}")

RuntimeEngine().start()

router = CommandRouter()
router.execute("BOOT")

registry = AgentRegistry()
registry.register("Commander")
registry.register("Guardian")
registry.start_all()

plugins = PluginManager()
plugins.discover()
plugins.start()
