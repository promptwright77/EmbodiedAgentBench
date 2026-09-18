"""Planning module for task decomposition and plan management."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from src.core.protocols import Planner
from src.core.types import Observation, Plan, SubTask, TaskGoal, TaskStatus


class LLMTaskPlanner:
    """LLM-based task planner using LangGraph for orchestration."""

    def __init__(
        self,
        llm: Any,
        max_subtasks: int = 10,
        temperature: float = 0.0,
    ) -> None:
        self.llm = llm
        self.max_subtasks = max_subtasks
        self.temperature = temperature
        self._plan_history: List[Plan] = []

    def create_plan(self, goal: TaskGoal, observation: Observation) -> Plan:
        """Create a plan by decomposing the goal into subtasks."""
        prompt = self._build_planning_prompt(goal, observation)

        try:
            response = self.llm.invoke(prompt)
            subtasks = self._parse_subtasks_from_response(response, goal.id)
        except Exception:
            # Fallback to simple decomposition
            subtasks = self._fallback_decomposition(goal)

        plan = Plan(
            id=str(uuid.uuid4()),
            goal_id=goal.id,
            subtasks=subtasks,
            created_at=time.time(),
        )
        self._plan_history.append(plan)
        return plan

    def replan(self, failed_subtask: SubTask, observation: Observation) -> Optional[Plan]:
        """Create a new plan when the current one fails."""
        # Analyze failure and create revised plan
        prompt = f"""The following subtask failed: {failed_subtask.description}
Error: {failed_subtask.error}

Current observation: {observation.get_text_summary()}

Create a revised plan that:
1. Addresses the failure reason
2. Is achievable given current state
3. Breaks down into specific, retriable subtasks

Format your response as a JSON list of subtask descriptions."""

        try:
            response = self.llm.invoke(prompt)
            subtasks = self._parse_subtasks_from_response(response, observation.goal.id)
            if subtasks:
                return Plan(
                    id=str(uuid.uuid4()),
                    goal_id=observation.goal.id,
                    subtasks=subtasks,
                    created_at=time.time(),
                    metadata={"replan_from": failed_subtask.id},
                )
        except Exception:
            pass
        return None

    def should_replan(self, plan: Plan, observation: Observation) -> bool:
        """Determine if the current plan needs to be replaced."""
        # Replan if too many subtasks have failed
        failed_count = sum(1 for s in plan.subtasks if s.status == TaskStatus.FAILED)
        total_count = len(plan.subtasks)

        if total_count > 0 and failed_count / total_count > 0.5:
            return True

        # Replan if we're stuck (no progress in multiple steps)
        if observation.elapsed_steps > 5 and not observation.completed_subtasks:
            return True

        return False

    def _build_planning_prompt(self, goal: TaskGoal, observation: Observation) -> str:
        """Build the prompt for LLM-based planning."""
        context = observation.get_text_summary() if observation.elapsed_steps > 0 else "Initial state"
        constraints = ", ".join(f"{k}={v}" for k, v in goal.constraints.items())

        return f"""You are a task planner for a robot agent. Decompose the following task into specific subtasks.

Task: {goal.description}
Constraints: {constraints or "None"}
Current state: {context}

Break down this task into {self.max_subtasks} or fewer specific, ordered subtasks.
Each subtask should be:
- Atomic (can be completed in one step)
- Observable (you can tell when it's done)
- Relevant to the main task

Format your response as a JSON list of subtask descriptions, ordered sequentially.
Example: ["First do X", "Then do Y", "Finally do Z"]
"""

    def _parse_subtasks_from_response(self, response: Any, goal_id: str) -> List[SubTask]:
        """Parse subtasks from LLM response."""
        subtasks = []

        try:
            # Try to parse as JSON
            content = response.content if hasattr(response, "content") else str(response)
            import json

            # Extract JSON from response
            import re
            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                items = json.loads(json_match.group())
                for i, desc in enumerate(items[: self.max_subtasks]):
                    subtasks.append(
                        SubTask(
                            id=f"{goal_id}-subtask-{i + 1}",
                            description=desc,
                            status=TaskStatus.PENDING,
                        )
                    )
        except Exception:
            pass

        return subtasks

    def _fallback_decomposition(self, goal: TaskGoal) -> List[SubTask]:
        """Fallback task decomposition when LLM is unavailable."""
        return [
            SubTask(
                id=f"{goal.id}-subtask-1",
                description=f"Attempt to: {goal.description}",
                status=TaskStatus.PENDING,
            )
        ]


class DeterministicPlanner:
    """A deterministic planner for testing without LLM."""

    def __init__(self) -> None:
        self._plan_templates: Dict[str, List[str]] = {
            "pick_and_place": [
                "approach the target object",
                "grasp the target object",
                "lift the object",
                "move to the target location",
                "place the object at the target",
                "release the object",
            ],
            "default": [
                "observe the environment",
                "analyze current state",
                "execute task step",
                "verify execution result",
            ],
        }

    def create_plan(self, goal: TaskGoal, observation: Observation) -> Plan:
        """Create a plan using predefined templates."""
        # Detect task type from goal description
        goal_lower = goal.description.lower()
        template_key = "default"

        if "pick" in goal_lower and "place" in goal_lower:
            template_key = "pick_and_place"
        elif "grasp" in goal_lower or "grab" in goal_lower:
            template_key = "pick_and_place"

        template = self._plan_templates.get(template_key, self._plan_templates["default"])

        subtasks = [
            SubTask(
                id=f"{goal.id}-subtask-{i + 1}",
                description=desc,
                status=TaskStatus.PENDING,
            )
            for i, desc in enumerate(template)
        ]

        return Plan(
            id=str(uuid.uuid4()),
            goal_id=goal.id,
            subtasks=subtasks,
            created_at=time.time(),
        )

    def replan(self, failed_subtask: SubTask, observation: Observation) -> Optional[Plan]:
        """Create a simplified plan after failure."""
        return self.create_plan(observation.goal, observation)

    def should_replan(self, plan: Plan, observation: Observation) -> bool:
        """Replan if more than half of completed subtasks failed."""
        failed_count = sum(1 for s in plan.subtasks if s.status == TaskStatus.FAILED)
        return failed_count > len(plan.subtasks) / 2


class HierarchicalPlanner:
    """Hierarchical task planner with multiple abstraction levels."""

    def __init__(self, llm: Optional[Any] = None) -> None:
        self.llm = llm
        self._coarse_planner = DeterministicPlanner()
        self._fine_planner = LLMTaskPlanner(llm) if llm else DeterministicPlanner()

    def create_plan(
        self,
        goal: TaskGoal,
        observation: Observation,
        granularity: str = "coarse",
    ) -> Plan:
        """Create a plan at the specified granularity level."""
        if granularity == "fine" and self._fine_planner:
            return self._fine_planner.create_plan(goal, observation)
        return self._coarse_planner.create_plan(goal, observation)

    def replan(self, failed_subtask: SubTask, observation: Observation) -> Optional[Plan]:
        """Replan at the same granularity level."""
        return self._coarse_planner.replan(failed_subtask, observation)

    def should_replan(self, plan: Plan, observation: Observation) -> bool:
        """Determine if replanning is needed."""
        return self._coarse_planner.should_replan(plan, observation)
