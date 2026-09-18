#!/usr/bin/env python3
"""Demo script showcasing the EmbodiedAgentBench framework."""

from __future__ import annotations

import argparse
import time
import uuid

from src.core.types import Action, Observation, Percept, PerceptionSource, TaskGoal, TaskStatus
from src.evaluation import EvaluationMode, EvaluationMetrics, EvaluationRunner
from src.execution import MockActionExecutor
from src.memory import InMemoryStore
from src.multi_agent import AgentTeam, ExecutionAgent, PerceptionAgent, PlanningAgent, ReflectionAgent
from src.perception import MultiModalPerceiver, SimulatorPerceiver, TextPerceiver
from src.planning import DeterministicPlanner, HierarchicalPlanner
from src.reflection import SelfReflector


def create_sample_goal() -> TaskGoal:
    """Create a sample task goal for demonstration."""
    return TaskGoal(
        id=str(uuid.uuid4()),
        description="Pick up the red cube and place it in the blue bin",
        constraints={"precision": "high", "max_time": 60},
        max_steps=20,
    )


def create_sample_observation(goal: TaskGoal, step: int = 0) -> Observation:
    """Create a sample observation for demonstration."""
    percept = Percept(
        source=PerceptionSource.SIMULATOR,
        data={"robot_position": [0.1, 0.2, 0.3], "objects": ["red_cube", "blue_bin"]},
        timestamp=time.time(),
        confidence=0.95,
        metadata={"description": "Simulator state"},
    )

    return Observation(
        goal=goal,
        percepts=[percept],
        elapsed_steps=step,
        completed_subtasks=[],
        failed_subtasks=[],
    )


def run_mock_evaluation() -> EvaluationMetrics:
    """Run a mock evaluation demonstrating the framework."""
    print("=" * 60)
    print("Running MOCK Evaluation Mode")
    print("=" * 60)

    # Create components
    planner = DeterministicPlanner()
    executor = MockActionExecutor(failure_rate=0.1)
    reflector = SelfReflector()
    memory = InMemoryStore()

    # Create goal and observation
    goal = create_sample_goal()
    observation = create_sample_observation(goal)

    print(f"\nTask: {goal.description}")
    print(f"Constraints: {goal.constraints}")

    # Create plan
    plan = planner.create_plan(goal, observation)
    print(f"\nPlan created with {len(plan.subtasks)} subtasks:")
    for i, subtask in enumerate(plan.subtasks):
        print(f"  {i + 1}. {subtask.description}")

    # Execute plan
    print("\nExecuting plan...")
    for i, subtask in enumerate(plan.subtasks):
        subtask.status = TaskStatus.IN_PROGRESS
        print(f"\n  Step {i + 1}: {subtask.description}")

        # Create and execute action
        action = Action(
            id=str(uuid.uuid4()),
            type="mock_action",
            parameters={"subtask_id": subtask.id},
        )

        result = executor.execute(action)

        if result["success"]:
            subtask.status = TaskStatus.COMPLETED
            observation.completed_subtasks.append(subtask.id)
            print(f"    [OK] Success: {result}")
        else:
            subtask.status = TaskStatus.FAILED
            observation.failed_subtasks.append(subtask.id)
            print(f"    [FAIL] Failed: {result.get('error', 'Unknown error')}")

        # Store experience
        memory.store_experience(observation, action, result)

        # Reflect
        reflection = reflector.reflect(observation, result)
        print(f"    Reflection: {reflection.assessment} (confidence: {reflection.confidence:.2f})")

    # Summary
    success_rate = len(observation.completed_subtasks) / len(plan.subtasks)
    print(f"\n{'=' * 60}")
    print(f"Evaluation Complete")
    print(f"{'=' * 60}")
    print(f"  Completed: {len(observation.completed_subtasks)}/{len(plan.subtasks)} subtasks")
    print(f"  Failed: {len(observation.failed_subtasks)}/{len(plan.subtasks)} subtasks")
    print(f"  Success Rate: {success_rate * 100:.1f}%")

    metrics = EvaluationMetrics(
        evaluation_mode="MOCK",
        task_success_rate=success_rate,
        task_steps_taken=len(plan.subtasks),
        evaluation_note=(
            "Mock evaluation: Results are deterministic and do not reflect "
            "real robot performance. No physics simulation or actual sensing."
        ),
    )

    return metrics


def run_multi_agent_demo() -> None:
    """Demonstrate multi-agent collaboration."""
    print("\n" + "=" * 60)
    print("Running Multi-Agent Collaboration Demo")
    print("=" * 60)

    # Create agent team
    team = AgentTeam()

    # Register specialized agents
    perception_agent = PerceptionAgent()
    planning_agent = PlanningAgent()
    execution_agent = ExecutionAgent()
    reflection_agent = ReflectionAgent()

    team.register_agent(perception_agent)
    team.register_agent(planning_agent)
    team.register_agent(execution_agent)
    team.register_agent(reflection_agent)

    print("\nRegistered Agents:")
    for agent_info in team.list_agents():
        print(f"  - {agent_info['agent_id']} ({agent_info['role']})")
        print(f"    Capabilities: {', '.join(agent_info['capabilities'])}")

    # Demonstrate message passing
    print("\nDemonstrating message passing...")
    team.send_message(
        sender_id="planning_agent",
        receiver_id="execution_agent",
        content={"task": "Execute pick and place", "priority": "high"},
        message_type="task_assignment",
    )

    team.synchronize()
    print("  ✓ Messages processed successfully")

    # Demonstrate shared state
    print("\nDemonstrating shared state...")
    team.update_shared_state("current_task", "Pick and place operation")
    team.update_shared_state("progress", 0.5)
    print(f"  Current task: {team.get_shared_state('current_task')}")
    print(f"  Progress: {team.get_shared_state('progress') * 100:.0f}%")


def run_reflection_demo() -> None:
    """Demonstrate reflection capabilities."""
    print("\n" + "=" * 60)
    print("Running Reflection Demo")
    print("=" * 60)

    reflector = SelfReflector()
    memory = InMemoryStore()

    # Scenario 1: On track
    print("\nScenario 1: Agent is on track")
    goal = create_sample_goal()
    observation = create_sample_observation(goal, step=2)
    observation.completed_subtasks = ["subtask-1", "subtask-2"]

    result = {"success": True, "output": "Action completed"}
    reflection = reflector.reflect(observation, result)

    print(f"  Assessment: {reflection.assessment}")
    print(f"  Confidence: {reflection.confidence:.2f}")
    print(f"  Issues: {reflection.issues_identified or 'None'}")
    print(f"  Adjustments: {reflection.suggested_adjustments or 'None'}")

    # Scenario 2: Struggling
    print("\nScenario 2: Agent is struggling")
    observation2 = create_sample_observation(goal, step=5)
    observation2.failed_subtasks = ["subtask-1", "subtask-2", "subtask-3"]

    result2 = {"success": False, "error": "Action timed out"}
    reflection2 = reflector.reflect(observation2, result2)

    print(f"  Assessment: {reflection2.assessment}")
    print(f"  Confidence: {reflection2.confidence:.2f}")
    print(f"  Issues: {reflection2.issues_identified}")
    print(f"  Retry recommended: {reflection2.retry_recommended}")
    print(f"  Replan recommended: {reflection2.replan_recommended}")


def main() -> None:
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(description="EmbodiedAgentBench Demo")
    parser.add_argument(
        "--mode",
        choices=["all", "mock", "multi-agent", "reflection"],
        default="all",
        help="Demo mode to run",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("EmbodiedAgentBench Framework Demo")
    print("=" * 60)
    print("A comprehensive embodied AI agent framework with")
    print("perception-planning-action-reflection闭环")
    print("=" * 60)

    if args.mode in ["all", "mock"]:
        metrics = run_mock_evaluation()
        print(f"\nEvaluation Metrics:")
        print(f"  Mode: {metrics.evaluation_mode}")
        print(f"  Success Rate: {metrics.task_success_rate * 100:.1f}%")
        print(f"  Steps Taken: {metrics.task_steps_taken}")
        print(f"  Note: {metrics.evaluation_note}")

    if args.mode in ["all", "multi-agent"]:
        run_multi_agent_demo()

    if args.mode in ["all", "reflection"]:
        run_reflection_demo()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
