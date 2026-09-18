"""Protocol definitions for the EmbodiedAgentBench framework.

Protocols define the interfaces that components must implement,
enabling dependency injection and easy testing with mocks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from src.core.types import (
    Action,
    Observation,
    Percept,
    Plan,
    ReflectionResult,
    SubTask,
    TaskGoal,
)


@runtime_checkable
class PerceptionModule(Protocol):
    """Module that processes raw sensor data into structured percepts."""

    def perceive(self, raw_input: Any) -> List[Percept]:
        """Process raw sensor input into percepts."""
        ...

    def fuse(self, percepts: List[Percept]) -> Percept:
        """Fuse multiple percepts into a unified representation."""
        ...


@runtime_checkable
class Planner(Protocol):
    """Module that decomposes goals into subtasks and creates plans."""

    def create_plan(self, goal: TaskGoal, observation: Observation) -> Plan:
        """Create a plan to achieve the given goal."""
        ...

    def replan(self, failed_subtask: SubTask, observation: Observation) -> Optional[Plan]:
        """Create a new plan when the current one fails."""
        ...

    def should_replan(self, plan: Plan, observation: Observation) -> bool:
        """Determine if the current plan needs to be replaced."""
        ...


@runtime_checkable
class ActionExecutor(Protocol):
    """Module that executes actions in the environment."""

    def execute(self, action: Action) -> Dict[str, Any]:
        """Execute an action and return the result."""
        ...

    def validate_action(self, action: Action) -> bool:
        """Check if an action is valid in the current context."""
        ...

    def get_available_actions(self) -> List[str]:
        """Return list of action types this executor supports."""
        ...


@runtime_checkable
class Reflector(Protocol):
    """Module that reflects on execution results and provides insights."""

    def reflect(
        self,
        observation: Observation,
        action_result: Dict[str, Any],
    ) -> ReflectionResult:
        """Analyze execution result and provide reflection."""
        ...

    def assess_confidence(self, observation: Observation) -> float:
        """Assess confidence in the current plan execution."""
        ...

    def should_retry(self, observation: Observation, action_result: Dict[str, Any]) -> bool:
        """Determine if an action should be retried."""
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """Persistent storage for agent memories and experiences."""

    def store(self, key: str, value: Any) -> None:
        """Store a value in memory."""
        ...

    def retrieve(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from memory."""
        ...

    def store_experience(
        self,
        observation: Observation,
        action: Action,
        result: Dict[str, Any],
    ) -> None:
        """Store a complete experience tuple."""
        ...

    def retrieve_relevant(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve experiences relevant to a query."""
        ...

    def clear(self) -> None:
        """Clear all stored memories."""
        ...


@runtime_checkable
class ToolRegistry(Protocol):
    """Registry for agent tools/functions."""

    def register(self, name: str, func: Any, description: str) -> None:
        """Register a new tool."""
        ...

    def get_tool(self, name: str) -> Any:
        """Get a registered tool by name."""
        ...

    def list_tools(self) -> List[Dict[str, str]]:
        """List all registered tools with descriptions."""
        ...

    def call_tool(self, name: str, **kwargs: Any) -> Any:
        """Call a registered tool with arguments."""
        ...


@runtime_checkable
class MultiAgentCoordinator(Protocol):
    """Coordinator for multi-agent collaboration."""

    def register_agent(self, agent_id: str, role: str) -> None:
        """Register an agent with a specific role."""
        ...

    def distribute_task(self, task: SubTask) -> Dict[str, SubTask]:
        """Distribute a task to appropriate agents."""
        ...

    def aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from multiple agents."""
        ...

    def synchronize(self, agent_id: str, state: Dict[str, Any]) -> None:
        """Synchronize state with other agents."""
        ...


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """Adapter for connecting to physical or simulated environments."""

    def reset(self) -> Observation:
        """Reset the environment to initial state."""
        ...

    def step(self, action: Action) -> Observation:
        """Execute an action and return the new observation."""
        ...

    def get_observation(self) -> Observation:
        """Get current observation without taking an action."""
        ...

    def is_done(self) -> bool:
        """Check if the episode is complete."""
        ...

    def get_reward(self) -> float:
        """Get the current reward value."""
        ...
