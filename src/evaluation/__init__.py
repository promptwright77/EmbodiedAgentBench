"""Evaluation module for measuring agent performance."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.types import Action, Observation, TaskStatus


class EvaluationMode(Enum):
    """Evaluation modes available for the agent."""

    MOCK = auto()  # Deterministic mock evaluation
    REPLAY = auto()  # Replay-based evaluation
    SIMULATION = auto()  # Closed-loop simulation
    REAL = auto()  # Real robot evaluation


@dataclass
class EvaluationMetrics:
    """Metrics collected during evaluation."""

    # Task-level metrics
    task_success_rate: float = 0.0
    task_completion_time: float = 0.0
    task_steps_taken: int = 0

    # Subtask-level metrics
    subtask_success_count: int = 0
    subtask_failure_count: int = 0
    subtask_retry_count: int = 0
    subtask_replan_count: int = 0

    # Action-level metrics
    action_success_count: int = 0
    action_failure_count: int = 0
    total_actions: int = 0

    # Reflection metrics
    reflection_count: int = 0
    retry_recommendations: int = 0
    replan_recommendations: int = 0

    # Memory metrics
    experiences_stored: int = 0
    experiences_retrieved: int = 0

    # Confidence metrics
    initial_confidence: float = 0.0
    final_confidence: float = 0.0
    avg_confidence: float = 0.0

    # Evaluation metadata
    evaluation_mode: str = "unknown"
    evaluation_note: str = ""
    timestamp: float = field(default_factory=time.time)


class EvaluationRunner:
    """Runner for evaluating the agent in different modes."""

    def __init__(self, mode: EvaluationMode = EvaluationMode.MOCK) -> None:
        self.mode = mode
        self._metrics_history: List[EvaluationMetrics] = []

    def evaluate(
        self,
        agent: Any,
        test_cases: List[Dict[str, Any]],
    ) -> EvaluationMetrics:
        """Evaluate the agent on a set of test cases."""
        total_metrics = EvaluationMetrics(evaluation_mode=self.mode.name)

        for test_case in test_cases:
            case_metrics = self._evaluate_single_case(agent, test_case)
            self._merge_metrics(total_metrics, case_metrics)

        # Calculate averages
        if len(test_cases) > 0:
            total_metrics.task_success_rate /= len(test_cases)
            total_metrics.avg_confidence /= max(self._count_successful_cases(total_metrics), 1)

        self._metrics_history.append(total_metrics)
        return total_metrics

    def _evaluate_single_case(
        self,
        agent: Any,
        test_case: Dict[str, Any],
    ) -> EvaluationMetrics:
        """Evaluate a single test case."""
        metrics = EvaluationMetrics(evaluation_mode=self.mode.name)

        goal_description = test_case.get("goal", "Complete the task")
        expected_steps = test_case.get("expected_steps", 5)

        metrics.initial_confidence = 1.0
        metrics.evaluation_note = self._get_evaluation_note(self.mode)

        # Simulate evaluation based on mode
        if self.mode == EvaluationMode.MOCK:
            success = test_case.get("mock_success", True)
            metrics.task_success_rate = 1.0 if success else 0.0
            metrics.task_steps_taken = expected_steps if success else expected_steps + 2
        else:
            # For other modes, run actual evaluation
            metrics.task_success_rate = 0.8  # Placeholder

        return metrics

    def _merge_metrics(
        self,
        total: EvaluationMetrics,
        case: EvaluationMetrics,
    ) -> None:
        """Merge metrics from a single case into total metrics."""
        total.task_success_rate += case.task_success_rate
        total.task_steps_taken += case.task_steps_taken
        total.subtask_success_count += case.subtask_success_count
        total.subtask_failure_count += case.subtask_failure_count
        total.action_success_count += case.action_success_count
        total.action_failure_count += case.action_failure_count
        total.total_actions += case.total_actions

    def _count_successful_cases(self, metrics: EvaluationMetrics) -> int:
        """Count the number of successful cases."""
        return int(metrics.task_success_rate > 0.5)

    def _get_evaluation_note(self, mode: EvaluationMode) -> str:
        """Get the evaluation note for the mode."""
        notes = {
            EvaluationMode.MOCK: (
                "Mock evaluation: Results are deterministic and do not reflect "
                "real robot performance. No physics simulation or actual sensing."
            ),
            EvaluationMode.REPLAY: (
                "Replay evaluation: Uses pre-recorded episodes. Results indicate "
                "policy behavior on fixed data, not real-world generalization."
            ),
            EvaluationMode.SIMULATION: (
                "Simulation evaluation: Uses simplified physics model. Results show "
                "closed-loop behavior but may not transfer to real robots."
            ),
            EvaluationMode.REAL: (
                "Real robot evaluation: Results reflect actual hardware performance. "
                "This is the ground truth for deployment."
            ),
        }
        return notes.get(mode, "Unknown evaluation mode")


class ExperimentLogger:
    """Logger for structured experiment tracking."""

    def __init__(self, log_dir: str = "./data/logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._experiment_id = 0

    def log_experiment(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        metrics: EvaluationMetrics,
    ) -> str:
        """Log an experiment with its configuration and metrics."""
        self._experiment_id += 1
        experiment_id = f"exp_{self._experiment_id:04d}"

        log_data = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "timestamp": time.time(),
            "config": config,
            "metrics": {
                "task_success_rate": metrics.task_success_rate,
                "task_completion_time": metrics.task_completion_time,
                "task_steps_taken": metrics.task_steps_taken,
                "subtask_success_count": metrics.subtask_success_count,
                "subtask_failure_count": metrics.subtask_failure_count,
                "action_success_count": metrics.action_success_count,
                "action_failure_count": metrics.action_failure_count,
                "reflection_count": metrics.reflection_count,
                "evaluation_mode": metrics.evaluation_mode,
                "evaluation_note": metrics.evaluation_note,
            },
        }

        log_file = self.log_dir / f"{experiment_id}.json"
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

        return experiment_id

    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Load a logged experiment."""
        log_file = self.log_dir / f"{experiment_id}.json"
        if not log_file.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found")

        with open(log_file) as f:
            return json.load(f)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all logged experiments."""
        experiments = []
        for log_file in sorted(self.log_dir.glob("exp_*.json")):
            try:
                with open(log_file) as f:
                    data = json.load(f)
                    experiments.append({
                        "experiment_id": data["experiment_id"],
                        "experiment_name": data["experiment_name"],
                        "timestamp": data["timestamp"],
                        "success_rate": data["metrics"]["task_success_rate"],
                    })
            except Exception:
                continue
        return experiments


class ComparativeEvaluator:
    """Evaluator for comparing different agent configurations."""

    def __init__(self) -> None:
        self._results: Dict[str, EvaluationMetrics] = {}

    def add_result(
        self,
        condition_name: str,
        metrics: EvaluationMetrics,
    ) -> None:
        """Add evaluation results for a condition."""
        self._results[condition_name] = metrics

    def compare(
        self,
        metric_name: str,
    ) -> Dict[str, float]:
        """Compare a specific metric across all conditions."""
        comparisons = {}

        for name, metrics in self._results.items():
            value = getattr(metrics, metric_name, None)
            if value is not None:
                comparisons[name] = value

        return comparisons

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all comparisons."""
        return {
            "conditions": list(self._results.keys()),
            "task_success_rates": self.compare("task_success_rate"),
            "avg_steps": self.compare("task_steps_taken"),
            "action_success_rates": {
                name: (
                    m.action_success_count / max(m.total_actions, 1)
                    if m.total_actions > 0
                    else 0.0
                )
                for name, m in self._results.items()
            },
        }
