import os
from pathlib import Path

from api.rest import RestApiServer
from cli.dashboard import TerminalDashboard
from config.settings import load_configuration
from core.eventbus import EventBus
from core.logging import GhostFireLogger
from core.scheduler import Scheduler
from core.service_manager import ServiceManager
from runtime.engine import RuntimeEngine
from router.router import CommandRouter
from agents.registry import AgentRegistry
from plugins.manager import PluginManager

event_bus = EventBus()

configuration = load_configuration(event_bus=event_bus)
settings = configuration.as_dict()

configured_log_root = settings["logging"]["root"]

log_root = Path(
    configured_log_root
    or os.environ.get(
        "GHOSTFIRE_LOG_ROOT",
        str(Path.home() / ".ghostfire" / "logs"),
    )
)

logger = GhostFireLogger(
    name="ghostfire.runtime",
    log_path=log_root / "ghostfire-os.jsonl",
    max_bytes=settings["logging"]["max_bytes"],
    backup_count=settings["logging"]["backup_count"],
    context={
        "app_name": settings["app_name"],
        "version": settings["version"],
        "configuration_revision": configuration.revision,
    },
)

logger.attach_event_bus(event_bus)

event_bus.emit(
    "ghostfire.configuration.active",
    {
        "revision": configuration.revision,
        "sources": list(configuration.sources),
    },
    raise_exceptions=False,
)

scheduler = Scheduler(event_bus=event_bus)
service_manager = ServiceManager(event_bus=event_bus)

event_bus.emit(
    "ghostfire.boot.started",
    {
        "app_name": settings["app_name"],
        "version": settings["version"],
    },
    raise_exceptions=False,
)

print(f"{settings['app_name']} {settings['version']}")
print("Configuration loaded")

runtime = RuntimeEngine()
router = CommandRouter()

registry = AgentRegistry()
registry.register("Commander")
registry.register("Guardian")

plugins = PluginManager()
dashboard = None
rest_api = None

if settings["rest_api"]["enabled"]:
    rest_api = RestApiServer(
        app_name=settings["app_name"],
        version=settings["version"],
        configuration_revision=configuration.revision,
        configuration_sources=configuration.sources,
        configuration=configuration.redacted(),
        service_manager=service_manager,
        scheduler=scheduler,
        host=settings["rest_api"]["host"],
        port=settings["rest_api"]["port"],
        auth_token=settings["rest_api"]["auth_token"],
        dashboard_provider=lambda: (
            dashboard.as_dict()
            if dashboard is not None
            else None
        ),
        event_bus=event_bus,
        request_timeout=settings[
            "rest_api"
        ]["request_timeout"],
    )


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
    lambda: scheduler.start(
        poll_interval=settings["scheduler"]["poll_interval"],
    ),
    stop=lambda: (
        scheduler.stop(
            timeout=settings[
                "service_manager"
            ]["scheduler_stop_timeout"]
        )
        if scheduler.is_running
        else False
    ),
    dependencies=("runtime",),
    health=lambda: scheduler.is_running,
)

if rest_api is not None:
    service_manager.register(
        "rest_api",
        rest_api.start,
        stop=rest_api.stop,
        dependencies=("runtime", "scheduler"),
        health=rest_api.is_running,
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

if rest_api is not None:
    print(f"REST API online: {rest_api.base_url}")
else:
    print("REST API disabled")

logger.info(
    "ghostfire.logging.ready",
    log_path=str(logger.log_path),
)

print("Logging online")
print("Service manager online")

if settings["terminal_dashboard"]["enabled"]:
    dashboard = TerminalDashboard(
        app_name=settings["app_name"],
        version=settings["version"],
        configuration_revision=configuration.revision,
        configuration_sources=configuration.sources,
        service_manager=service_manager,
        scheduler=scheduler,
        log_path=logger.log_path,
        event_bus=event_bus,
        width=settings["terminal_dashboard"]["width"],
        color=settings["terminal_dashboard"]["color"],
    )
    dashboard.display(
        check_health=settings[
            "terminal_dashboard"
        ]["show_health"],
    )
    print("Terminal dashboard online")
else:
    print("Terminal dashboard disabled")

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
        "configuration": "loaded",
        "configuration_revision": configuration.revision,
        "terminal_dashboard": (
            "online"
            if dashboard is not None
            else "disabled"
        ),
        "rest_api": (
            "online"
            if rest_api is not None
            else "disabled"
        ),
    },
    raise_exceptions=False,
)

logger.info(
    "ghostfire.runtime.ready",
    status="online",
)
