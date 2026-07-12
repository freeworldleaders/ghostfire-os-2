from config.settings import SETTINGS
from core.eventbus import EventBus
from core.scheduler import Scheduler
from runtime.engine import RuntimeEngine
from router.router import CommandRouter
from agents.registry import AgentRegistry
from plugins.manager import PluginManager

event_bus = EventBus()
scheduler = Scheduler(event_bus=event_bus)

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

scheduler.schedule_once(
    "ghostfire.scheduler.bootstrap",
    0,
    lambda: event_bus.emit(
        "ghostfire.scheduler.ready",
        {"status": "online"},
        raise_exceptions=False,
    ),
)
scheduler.run_pending()

print("Scheduler online")

event_bus.emit(
    "ghostfire.boot.completed",
    {
        "runtime": "online",
        "router": "BOOT",
        "agents": ["Commander", "Guardian"],
        "plugins": "started",
        "scheduler": "online",
    },
    raise_exceptions=False,
)
