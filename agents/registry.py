"""Thread-safe AI agent registry and capability router."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from threading import RLock
from typing import Any

from agents.framework import (
    Agent,
    AgentCapabilityError,
    AgentHandler,
    AgentRegistrationError,
    AgentState,
    AgentTask,
)
from agents.policy import AgentExecutionPolicy, PolicyAction
from agents.tools import AgentToolRegistry
from core.eventbus import EventBus


class AgentRegistry:
    """Register, lifecycle-manage, and route work to AI agents."""

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        tool_registry: AgentToolRegistry | None = None,
        execution_policy: AgentExecutionPolicy | None = None,
        history_limit: int = 100,
        memory_limit: int = 100,
    ) -> None:
        if event_bus is not None and not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus or None")

        if (
            tool_registry is not None
            and not isinstance(tool_registry, AgentToolRegistry)
        ):
            raise TypeError(
                "tool_registry must be an AgentToolRegistry or None"
            )

        if (
            execution_policy is not None
            and not isinstance(
                execution_policy,
                AgentExecutionPolicy,
            )
        ):
            raise TypeError(
                "execution_policy must be an "
                "AgentExecutionPolicy or None"
            )

        if (
            isinstance(history_limit, bool)
            or not isinstance(history_limit, int)
            or history_limit < 1
        ):
            raise ValueError("history_limit must be a positive integer")

        if (
            isinstance(memory_limit, bool)
            or not isinstance(memory_limit, int)
            or memory_limit < 1
        ):
            raise ValueError("memory_limit must be a positive integer")

        self._event_bus = event_bus
        self._tool_registry = tool_registry
        self._execution_policy = execution_policy
        self._history_limit = history_limit
        self._memory_limit = memory_limit
        self._lock = RLock()
        self._agents: dict[str, Agent] = {}
        self.agents = self._agents

    def register(
        self,
        name: str,
        *,
        role: str = "general",
        capabilities: Iterable[str] = ("status",),
        handler: AgentHandler | None = None,
        allowed_tools: Iterable[str] = (),
    ) -> Agent:
        """Register and return one agent."""

        if isinstance(allowed_tools, str):
            raise TypeError(
                "allowed_tools must be an iterable of names"
            )

        normalized_allowed_tools = tuple(allowed_tools)

        if (
            self._tool_registry is None
            and normalized_allowed_tools
        ):
            raise AgentRegistrationError(
                "allowed_tools require a configured tool registry"
            )

        tool_client = (
            self._tool_registry.client(
                agent_name=name,
                agent_role=role,
                allowed_tools=normalized_allowed_tools,
            )
            if self._tool_registry is not None
            else None
        )

        agent = Agent(
            name,
            role=role,
            capabilities=capabilities,
            handler=handler,
            tool_client=tool_client,
            event_bus=self._event_bus,
            history_limit=self._history_limit,
            memory_limit=self._memory_limit,
        )

        with self._lock:
            if agent.name in self._agents:
                raise AgentRegistrationError(
                    f"agent already registered: {agent.name}"
                )

            self._agents[agent.name] = agent
            snapshot = agent.snapshot()

        self._publish(
            "ghostfire.agent.registered",
            snapshot,
        )
        return agent

    def unregister(self, name: str) -> Agent:
        """Remove and return a stopped agent."""

        normalized = self._normalize_name(name)

        with self._lock:
            agent = self._require_locked(normalized)

            if agent.state in {
                AgentState.ONLINE,
                AgentState.BUSY,
                AgentState.STARTING,
                AgentState.STOPPING,
            }:
                raise AgentRegistrationError(
                    "online agents cannot be unregistered"
                )

            self._agents.pop(normalized)

        self._publish(
            "ghostfire.agent.unregistered",
            agent.snapshot(),
        )
        return agent

    def get(self, name: str) -> Agent:
        """Return one registered agent."""

        normalized = self._normalize_name(name)

        with self._lock:
            return self._require_locked(normalized)

    def list_agents(self) -> tuple[Agent, ...]:
        """Return agents in registration order."""

        with self._lock:
            return tuple(self._agents.values())

    def start_all(self) -> tuple[str, ...]:
        """Start every registered agent."""

        started: list[str] = []

        for agent in self.list_agents():
            if agent.start():
                started.append(agent.name)

        return tuple(started)

    def stop_all(self) -> tuple[str, ...]:
        """Stop every registered agent in reverse order."""

        stopped: list[str] = []

        for agent in reversed(self.list_agents()):
            if agent.stop():
                stopped.append(agent.name)

        return tuple(stopped)

    def health(self) -> bool:
        """Return whether all registered agents are healthy."""

        agents = self.list_agents()
        return bool(agents) and all(
            agent.health()
            for agent in agents
        )

    def dispatch(
        self,
        capability: str,
        payload: Mapping[str, Any] | None = None,
        *,
        preferred_agent: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        """Route one task to a healthy capable agent."""

        task = AgentTask.create(
            capability,
            payload,
            metadata=metadata,
        )
        agent = self._select_agent(
            task.capability,
            preferred_agent=preferred_agent,
        )

        if self._execution_policy is not None:
            self._execution_policy.authorize(
                action=PolicyAction.AGENT_TASK,
                agent_name=agent.name,
                agent_role=agent.role,
                resource=task.capability,
                mode="execute",
                attributes={
                    "payload_keys": tuple(sorted(task.payload)),
                    "metadata_keys": tuple(sorted(task.metadata)),
                },
            )

        self._publish(
            "ghostfire.agent.task.routed",
            {
                "agent": agent.name,
                "role": agent.role,
                "task": task.as_dict(),
            },
        )

        return agent.execute(task)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe registry snapshot."""

        agents = self.list_agents()
        snapshots = [
            agent.snapshot()
            for agent in agents
        ]

        return {
            "count": len(snapshots),
            "healthy": bool(snapshots) and all(
                item["healthy"]
                for item in snapshots
            ),
            "agents": snapshots,
            "tool_registry_attached": (
                self._tool_registry is not None
            ),
            "execution_policy_attached": (
                self._execution_policy is not None
            ),
        }

    def _select_agent(
        self,
        capability: str,
        *,
        preferred_agent: str | None,
    ) -> Agent:
        if preferred_agent is not None:
            agent = self.get(preferred_agent)

            if not agent.supports(capability):
                raise AgentCapabilityError(
                    f"preferred agent {agent.name!r} does not "
                    f"support {capability!r}"
                )

            if not agent.health():
                raise AgentCapabilityError(
                    f"preferred agent {agent.name!r} is not healthy"
                )

            return agent

        for agent in self.list_agents():
            if agent.supports(capability) and agent.health():
                return agent

        raise AgentCapabilityError(
            f"no healthy agent supports capability {capability!r}"
        )

    def _require_locked(self, name: str) -> Agent:
        try:
            return self._agents[name]
        except KeyError as exc:
            raise AgentRegistrationError(
                f"agent not registered: {name}"
            ) from exc

    @staticmethod
    def _normalize_name(name: str) -> str:
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        normalized = name.strip()

        if not normalized:
            raise ValueError("name cannot be empty")

        return normalized

    def _publish(
        self,
        event_name: str,
        payload: Mapping[str, Any],
    ) -> None:
        if self._event_bus is None:
            return

        self._event_bus.emit(
            event_name,
            deepcopy(dict(payload)),
            raise_exceptions=False,
        )
