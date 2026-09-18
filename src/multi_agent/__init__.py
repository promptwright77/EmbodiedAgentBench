"""Multi-agent module for coordinated agent collaboration."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.core.protocols import MultiAgentCoordinator
from src.core.types import AgentMessage, SubTask, TaskGoal


class SimpleAgent:
    """A simple agent with a specific role."""

    def __init__(self, agent_id: str, role: str, capabilities: List[str]) -> None:
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self._state: Dict[str, Any] = {}
        self._message_queue: List[AgentMessage] = []

    def receive_message(self, message: AgentMessage) -> None:
        """Receive a message."""
        self._message_queue.append(message)

    def send_message(
        self,
        receiver: str,
        content: Dict[str, Any],
        message_type: str,
    ) -> AgentMessage:
        """Send a message to another agent."""
        return AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            content=content,
            message_type=message_type,
            timestamp=time.time(),
        )

    def update_state(self, key: str, value: Any) -> None:
        """Update the agent's internal state."""
        self._state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get the agent's internal state."""
        return self._state.get(key, default)

    def process_queue(self) -> List[AgentMessage]:
        """Process all queued messages and return responses."""
        responses = []
        while self._message_queue:
            message = self._message_queue.pop(0)
            response = self._process_message(message)
            if response:
                responses.append(response)
        return responses

    def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process a single message and return a response if needed."""
        # Default processing - subclasses can override
        return None


class PerceptionAgent(SimpleAgent):
    """Agent specialized in perception tasks."""

    def __init__(self, agent_id: str = "perception_agent") -> None:
        super().__init__(
            agent_id=agent_id,
            role="perception",
            capabilities=["visual_processing", "sensor_fusion", "object_detection"],
        )


class PlanningAgent(SimpleAgent):
    """Agent specialized in task planning."""

    def __init__(self, agent_id: str = "planning_agent") -> None:
        super().__init__(
            agent_id=agent_id,
            role="planning",
            capabilities=["task_decomposition", "plan_creation", "replanning"],
        )


class ExecutionAgent(SimpleAgent):
    """Agent specialized in action execution."""

    def __init__(self, agent_id: str = "execution_agent") -> None:
        super().__init__(
            agent_id=agent_id,
            role="execution",
            capabilities=["action_execution", "tool_use", "control"],
        )


class ReflectionAgent(SimpleAgent):
    """Agent specialized in self-reflection."""

    def __init__(self, agent_id: str = "reflection_agent") -> None:
        super().__init__(
            agent_id=agent_id,
            role="reflection",
            capabilities=["self_assessment", "confidence_estimation", "meta_cognition"],
        )


class AgentTeam:
    """A team of agents that collaborate on tasks."""

    def __init__(self) -> None:
        self._agents: Dict[str, SimpleAgent] = {}
        self._shared_state: Dict[str, Any] = {}

    def register_agent(self, agent: SimpleAgent) -> None:
        """Register an agent with the team."""
        self._agents[agent.agent_id] = agent

    def get_agent(self, agent_id: str) -> Optional[SimpleAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents."""
        return [
            {
                "agent_id": agent.agent_id,
                "role": agent.role,
                "capabilities": agent.capabilities,
            }
            for agent in self._agents.values()
        ]

    def broadcast(
        self,
        sender_id: str,
        content: Dict[str, Any],
        message_type: str,
    ) -> None:
        """Broadcast a message to all agents except the sender."""
        sender = self._agents.get(sender_id)
        if sender is None:
            return

        for agent_id, agent in self._agents.items():
            if agent_id != sender_id:
                message = sender.send_message(agent_id, content, message_type)
                agent.receive_message(message)

    def send_message(
        self,
        sender_id: str,
        receiver_id: str,
        content: Dict[str, Any],
        message_type: str,
    ) -> bool:
        """Send a message from one agent to another."""
        sender = self._agents.get(sender_id)
        receiver = self._agents.get(receiver_id)

        if sender is None or receiver is None:
            return False

        message = sender.send_message(receiver_id, content, message_type)
        receiver.receive_message(message)
        return True

    def update_shared_state(self, key: str, value: Any) -> None:
        """Update shared state accessible by all agents."""
        self._shared_state[key] = value

    def get_shared_state(self, key: str, default: Any = None) -> Any:
        """Get shared state."""
        return self._shared_state.get(key, default)

    def synchronize(self) -> None:
        """Synchronize all agents by processing their message queues."""
        for agent in self._agents.values():
            agent.process_queue()


class SimpleMultiAgentCoordinator:
    """Simple coordinator for multi-agent collaboration."""

    def __init__(self) -> None:
        self._agent_roles: Dict[str, str] = {}
        self._agent_states: Dict[str, Dict[str, Any]] = {}

    def register_agent(self, agent_id: str, role: str) -> None:
        """Register an agent with a specific role."""
        self._agent_roles[agent_id] = role
        self._agent_states[agent_id] = {}

    def distribute_task(self, task: SubTask) -> Dict[str, SubTask]:
        """Distribute a task to appropriate agents based on roles."""
        # Simple distribution: assign to agent with matching role
        distributed = {}

        # For now, assign the whole task to the first matching agent
        for agent_id, role in self._agent_roles.items():
            if role in task.description.lower() or "execute" in task.description.lower():
                distributed[agent_id] = task
                break

        if not distributed:
            # Default: assign to first registered agent
            if self._agent_roles:
                first_agent = next(iter(self._agent_roles.keys()))
                distributed[first_agent] = task

        return distributed

    def aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from multiple agents."""
        aggregated = {
            "success": all(r.get("success", False) for r in results.values()),
            "partial_success": any(r.get("success", False) for r in results.values()),
            "agent_count": len(results),
            "results": results,
        }
        return aggregated

    def synchronize(self, agent_id: str, state: Dict[str, Any]) -> None:
        """Synchronize state with other agents."""
        self._agent_states[agent_id] = state

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states from all agents."""
        return self._agent_states.copy()
