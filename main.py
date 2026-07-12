import os
from pathlib import Path

from config.settings import SETTINGS
from core.eventbus import EventBus
from core.logging import GhostFireLogger
from core.scheduler import Scheduler
from core.service_manager import ServiceManager
from runtime.engine import RuntimeEngine
from router.router import CommandRouter
from agents.registry import AgentRegistry
from plugins.manager import PluginManager

event_bus = EventBus()

log_root = Path(
    os.environ.get(
        "GHOSTFIRE_LOG_ROOT",
        str(Path.home() / ".ghostfire" / "logs"),
    )
)

logger = GhostFireLogger(
    name="ghostfire.runtime",
    log_path=log_root / "ghostfire-os.jsonl",
    context={
        "app_name": SETTINGS["app_name"],
        "version": SETTINGS["version"],
    },
)

logger.attach_event_bus(event_bus)

scheduler = Scheduler(event_bus=event_bus)
service_manager = ServiceManager(event_bus=event_bus)

event_bus.emit(
    "ghostfire.boot.started",
    {
        "app_name": SETTINGS["app_name"],
        "version": SETTINGS["version"],
    },
    raise_exceptions=False,
)

print(f"{SETTINGS['app_name']} {SETTINGS['version']}")

runtime = RuntimeEngine()
router = CommandRouter()

registry = AgentRegistry()
registry.register("Commander")
registry.register("Guardian")

plugins = PluginManager()


def start_plugins() -> None:
    plugins.discover()
    plugins.start()


service_manager.register(
    "runtime",
    runtime.start,
)

service_manager.register(
    "router",
    lambda: router.execute("BOOT"),
    dependencies=("runtime",),
)

service_manager.register(
    "agents",
    registry.start_all,
    dependencies=("runtime",),
)

service_manager.register(
    "plugins",
    start_plugins,
    dependencies=("runtime",),
)

service_manager.register(
    "scheduler",
    lambda: scheduler.start(poll_interval=0.05),
    stop=lambda: (
        scheduler.stop(timeout=1.0)
        if scheduler.is_running
        else False
    ),
    dependencies=("runtime",),
    health=lambda: scheduler.is_running,
)

scheduler.schedule_once(
    "ghostfire.scheduler.bootstrap",
    0,
    lambda: event_bus.emit(
        "ghostfire.scheduler.ready",
        {"status": "online"},
        raise_exceptions=False,
    ),
)

service_manager.start_all()
scheduler.run_pending()

print("Scheduler online")

logger.info(
    "ghostfire.logging.ready",
    log_path=str(logger.log_path),
)

print("Logging online")
print("Service manager online")

event_bus.emit(
    "ghostfire.boot.completed",
    {
        "runtime": "online",
        "router": "BOOT",
        "agents": ["Commander", "Guardian"],
        "plugins": "started",
        "scheduler": "online",
        "logging": "online",
        "service_manager": "online",
    },
    raise_exceptions=False,
)

logger.info(
    "ghostfire.runtime.ready",
    status="online",
)
