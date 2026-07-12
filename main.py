from config.settings import SETTINGS
from core.eventbus import EventBus
from runtime.engine import RuntimeEngine
from router.router import CommandRouter
from agents.registry import AgentRegistry
from plugins.manager import PluginManager

event_bus = EventBus()

event_bus.emit(
    "ghostfire.boot.started",
    {
        "app_name": SETTINGS["app_name"],
        "version": SETTINGS["version"],
    },
    raise_exceptions=False,
)

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

event_bus.emit(
    "ghostfire.boot.completed",
    {
        "runtime": "online",
        "router": "BOOT",
        "agents": ["Commander", "Guardian"],
        "plugins": "started",
    },
    raise_exceptions=False,
)
