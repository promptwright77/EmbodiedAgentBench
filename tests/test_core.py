"""Unit tests for the core module."""

import pytest
import uuid

from src.core.types import (
    Action,
    ActionStatus,
    Observation,
    Percept,
    PerceptionSource,
    Plan,
    ReflectionResult,
    SubTask,
    TaskGoal,
    TaskStatus,
)
from src.core.protocols import (
    ActionExecutor,
    MemoryStore,
    PerceptionModule,
    Planner,
    Reflector,
)


class TestTaskGoal:
    """Tests for TaskGoal dataclass."""

    def test_create_task_goal(self):
        goal = TaskGoal(
            id="test-1",
            description="Pick up the cube",
            constraints={"precision": "high"},
            max_steps=10,
        )
        assert goal.id == "test-1"
        assert goal.description == "Pick up the cube"
        assert goal.constraints["precision"] == "high"
        assert goal.max_steps == 10

    def test_task_goal_str(self):
        goal = TaskGoal(id="test-1", description="Test task")
        assert "test-1" in str(goal)
        assert "Test task" in str(goal)


class TestSubTask:
    """Tests for SubTask dataclass."""

    def test_create_subtask(self):
        subtask = SubTask(id="sub-1", description="Grasp the cube")
        assert subtask.id == "sub-1"
        assert subtask.status == TaskStatus.PENDING
        assert subtask.attempts == 0

    def test_subtask_is_retriable(self):
        subtask = SubTask(
            id="sub-1",
            description="Grasp the cube",
            status=TaskStatus.FAILED,
            attempts=1,
            max_attempts=3,
        )
        assert subtask.is_retriable() is True

        subtask.attempts = 3
        assert subtask.is_retriable() is False

        subtask.status = TaskStatus.COMPLETED
        assert subtask.is_retriable() is False


class TestAction:
    """Tests for Action dataclass."""

    def test_create_action(self):
        action = Action(
            id="action-1",
            type="move_to",
            parameters={"target": [1, 2, 3]},
        )
        assert action.id == "action-1"
        assert action.type == "move_to"
        assert action.status == ActionStatus.PENDING

    def test_action_to_dict(self):
        action = Action(id="action-1", type="grasp", parameters={"target": "cube"})
        data = action.to_dict()
        assert data["id"] == "action-1"
        assert data["type"] == "grasp"
        assert data["status"] == "PENDING"


class TestObservation:
    """Tests for Observation dataclass."""

    def test_create_observation(self):
        goal = TaskGoal(id="goal-1", description="Test")
        obs = Observation(goal=goal, elapsed_steps=5)
        assert obs.goal.id == "goal-1"
        assert obs.elapsed_steps == 5
        assert len(obs.percepts) == 0

    def test_observation_get_text_summary(self):
        goal = TaskGoal(id="goal-1", description="Test task")
        percept = Percept(
            source=PerceptionSource.TEXT,
            data="Robot sees a cube",
            timestamp=0.0,
        )
        obs = Observation(goal=goal, percepts=[percept], elapsed_steps=3)
        summary = obs.get_text_summary()
        assert "Step 3" in summary
        assert "cube" in summary


class TestPlan:
    """Tests for Plan dataclass."""

    def test_create_plan(self):
        subtasks = [
            SubTask(id="sub-1", description="Step 1"),
            SubTask(id="sub-2", description="Step 2"),
        ]
        plan = Plan(id="plan-1", goal_id="goal-1", subtasks=subtasks)
        assert len(plan.subtasks) == 2
        assert plan.current_subtask_index == 0

    def test_plan_get_current_subtask(self):
        subtasks = [SubTask(id="sub-1", description="Step 1")]
        plan = Plan(id="plan-1", goal_id="goal-1", subtasks=subtasks)
        current = plan.get_current_subtask()
        assert current is not None
        assert current.id == "sub-1"

    def test_plan_advance(self):
        subtasks = [
            SubTask(id="sub-1", description="Step 1"),
            SubTask(id="sub-2", description="Step 2"),
        ]
        plan = Plan(id="plan-1", goal_id="goal-1", subtasks=subtasks)
        assert plan.advance() is True
        assert plan.current_subtask_index == 1
        assert plan.advance() is False

    def test_plan_is_complete(self):
        subtasks = [
            SubTask(id="sub-1", description="Step 1", status=TaskStatus.COMPLETED),
            SubTask(id="sub-2", description="Step 2", status=TaskStatus.COMPLETED),
        ]
        plan = Plan(id="plan-1", goal_id="goal-1", subtasks=subtasks)
        assert plan.is_complete() is True

        subtasks[1].status = TaskStatus.FAILED
        assert plan.is_complete() is False


class TestReflectionResult:
    """Tests for ReflectionResult dataclass."""

    def test_create_reflection_result(self):
        result = ReflectionResult(
            assessment="on_track",
            confidence=0.8,
            issues_identified=["Issue 1"],
            suggested_adjustments=["Adjustment 1"],
        )
        assert result.assessment == "on_track"
        assert result.confidence == 0.8
        assert len(result.issues_identified) == 1

    def test_reflection_result_defaults(self):
        result = ReflectionResult(assessment="ok", confidence=0.5)
        assert result.retry_recommended is False
        assert result.replan_recommended is False
        assert result.abort_recommended is False
