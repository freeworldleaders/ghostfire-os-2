import os
from pathlib import Path

from api.rest import RestApiServer
from api.websocket import WebSocketCommandServer
from cli.dashboard import TerminalDashboard
from config.settings import load_configuration
from core.eventbus import EventBus
from core.logging import GhostFireLogger
from core.scheduler import Scheduler
from core.service_manager import ServiceManager
from runtime.engine import RuntimeEngine
from router.router import CommandRouter
from agents.orchestrator import AgentTaskOrchestrator
from agents.policy import (
    AgentExecutionPolicy,
    PolicyAction,
    PolicyEffect,
)
from agents.registry import AgentRegistry
from agents.tools import AgentToolRegistry
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

execution_policy = AgentExecutionPolicy(
    event_bus=event_bus,
    history_limit=settings[
        "agent_execution_policy"
    ]["history_limit"],
    default_effect=settings[
        "agent_execution_policy"
    ]["default_effect"],
)

execution_policy.register_rule(
    "allow-agent-role-tasks",
    PolicyEffect.ALLOW,
    actions=(PolicyAction.AGENT_TASK,),
    roles=("orchestrator", "safety"),
    resources=("*",),
    modes=("execute",),
    priority=100,
    reason="registered Kingdom agent roles may execute tasks",
)
execution_policy.register_rule(
    "require-owner-approval-for-mutations",
    PolicyEffect.REQUIRE_APPROVAL,
    actions=(PolicyAction.TOOL_INVOCATION,),
    roles=("orchestrator", "safety"),
    resources=("*",),
    modes=("mutating",),
    priority=200,
    reason="mutating agent tools require owner approval",
)
execution_policy.register_rule(
    "allow-read-only-agent-tools",
    PolicyEffect.ALLOW,
    actions=(PolicyAction.TOOL_INVOCATION,),
    roles=("orchestrator", "safety"),
    resources=("*",),
    modes=("read_only",),
    priority=100,
    reason="registered agent roles may use read-only tools",
)

tool_registry = AgentToolRegistry(
    event_bus=event_bus,
    execution_policy=execution_policy,
    history_limit=settings["agent_tools"]["history_limit"],
    allow_mutating=settings["agent_tools"]["allow_mutating"],
)

registry = AgentRegistry(
    event_bus=event_bus,
    tool_registry=tool_registry,
    execution_policy=execution_policy,
    history_limit=settings["ai_agents"]["history_limit"],
    memory_limit=settings["ai_agents"]["memory_limit"],
)

tool_registry.register(
    "ghostfire.echo",
    lambda message: {"message": message},
    description="Return one message without side effects.",
    parameters={"message": str},
    required=("message",),
    allowed_roles=("orchestrator",),
)
tool_registry.register(
    "ghostfire.agent_status",
    lambda: registry.snapshot(),
    description="Return the current AI agent registry snapshot.",
    allowed_roles=("orchestrator", "safety"),
)

registry.register(
    "Commander",
    role="orchestrator",
    capabilities=("orchestrate", "command", "status"),
    allowed_tools=(
        "ghostfire.echo",
        "ghostfire.agent_status",
    ),
)
registry.register(
    "Guardian",
    role="safety",
    capabilities=("validate", "guard", "status"),
    allowed_tools=("ghostfire.agent_status",),
)

orchestrator = AgentTaskOrchestrator(
    registry,
    event_bus=event_bus,
    history_limit=settings[
        "agent_orchestrator"
    ]["history_limit"],
    max_tasks=settings["agent_orchestrator"]["max_tasks"],
)

plugins = PluginManager()
dashboard = None
rest_api = None
websocket_command_server = None

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


def execute_websocket_command(command: str):
    result = router.execute(command)

    return {
        "command": command,
        "result": result,
    }


def websocket_status():
    statuses = service_manager.list_statuses()

    return {
        "app_name": settings["app_name"],
        "version": settings["version"],
        "configuration_revision": configuration.revision,
        "scheduler_running": scheduler.is_running,
        "agents": registry.snapshot(),
        "agent_tools": tool_registry.snapshot(),
        "agent_execution_policy": execution_policy.snapshot(),
        "agent_orchestrator": orchestrator.snapshot(),
        "services": [
            {
                "name": status.name,
                "state": status.state.value,
                "dependencies": list(status.dependencies),
                "last_error": status.last_error,
            }
            for status in statuses
        ],
    }


if settings["websocket_command_server"]["enabled"]:
    websocket_command_server = WebSocketCommandServer(
        command_handler=execute_websocket_command,
        status_provider=websocket_status,
        host=settings["websocket_command_server"]["host"],
        port=settings["websocket_command_server"]["port"],
        auth_token=settings[
            "websocket_command_server"
        ]["auth_token"],
        allowed_commands=settings[
            "websocket_command_server"
        ]["allowed_commands"],
        path=settings["websocket_command_server"]["path"],
        max_message_bytes=settings[
            "websocket_command_server"
        ]["max_message_bytes"],
        idle_timeout=settings[
            "websocket_command_server"
        ]["idle_timeout"],
        event_bus=event_bus,
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
    "execution_policy",
    execution_policy.start,
    stop=execution_policy.stop,
    dependencies=("runtime",),
    health=execution_policy.health,
)

service_manager.register(
    "agent_tools",
    tool_registry.start,
    stop=tool_registry.stop,
    dependencies=("runtime", "execution_policy"),
    health=tool_registry.health,
)

service_manager.register(
    "agents",
    registry.start_all,
    stop=registry.stop_all,
    dependencies=("runtime", "agent_tools"),
    health=registry.health,
)

service_manager.register(
    "agent_orchestrator",
    orchestrator.start,
    stop=orchestrator.stop,
    dependencies=("agents",),
    health=orchestrator.health,
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

if websocket_command_server is not None:
    service_manager.register(
        "websocket_command_server",
        websocket_command_server.start,
        stop=websocket_command_server.stop,
        dependencies=("runtime", "router", "scheduler"),
        health=websocket_command_server.is_running,
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
print("Agent execution policy online")
print("Agent tool registry online")
print("AI agent framework online")
print("Agent task orchestrator online")

if rest_api is not None:
    print(f"REST API online: {rest_api.base_url}")
else:
    print("REST API disabled")

if websocket_command_server is not None:
    print(
        "WebSocket command server online: "
        f"{websocket_command_server.base_url}"
    )
else:
    print("WebSocket command server disabled")

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
        "agents": [
            agent.name
            for agent in registry.list_agents()
        ],
        "agent_tools": "online",
        "agent_execution_policy": "online",
        "agent_orchestrator": "online",
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
        "websocket_command_server": (
            "online"
            if websocket_command_server is not None
            else "disabled"
        ),
    },
    raise_exceptions=False,
)

logger.info(
    "ghostfire.runtime.ready",
    status="online",
)
