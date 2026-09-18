"""Core domain types for the EmbodiedAgentBench framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
    """Status of a task in the agent loop."""

    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()
    REPLANNING = auto()


class ActionStatus(Enum):
    """Status of an action execution."""

    PENDING = auto()
    EXECUTING = auto()
    SUCCESS = auto()
    FAILURE = auto()
    TIMEOUT = auto()


class PerceptionSource(Enum):
    """Source of perceptual input."""

    CAMERA = auto()
    LIDAR = auto()
    TACTILE = auto()
    AUDIO = auto()
    TEXT = auto()
    VISION_LANGUAGE = auto()
    SIMULATOR = auto()


@dataclass(frozen=True)
class TaskGoal:
    """A high-level task goal for the embodied agent."""

    id: str
    description: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    max_steps: int = 20
    success_criteria: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Task({self.id}): {self.description}"


@dataclass
class SubTask:
    """A subtask decomposed from the main task goal."""

    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    parent_id: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def is_retriable(self) -> bool:
        return self.attempts < self.max_attempts and self.status in (
            TaskStatus.FAILED,
            TaskStatus.RETRYING,
        )


@dataclass
class Percept:
    """A perceptual observation from the environment."""

    source: PerceptionSource
    data: Any  # Can be image, point cloud, text, etc.
    timestamp: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_text_description(self) -> str:
        """Extract text description from the percept."""
        if self.source == PerceptionSource.TEXT:
            return str(self.data)
        return f"[{self.source.name}] {self.metadata.get('description', 'Perceptual input')}"


@dataclass
class Observation:
    """Complete observation at a point in time."""

    goal: TaskGoal
    percepts: List[Percept] = field(default_factory=list)
    elapsed_steps: int = 0
    completed_subtasks: List[str] = field(default_factory=list)
    failed_subtasks: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def get_text_summary(self) -> str:
        """Get a text summary of the observation."""
        percept_summaries = [p.get_text_description() for p in self.percepts]
        return f"Step {self.elapsed_steps}: {'; '.join(percept_summaries)}"


@dataclass
class Action:
    """An action to be executed by the agent."""

    id: str
    type: str  # e.g., "move_to", "grasp", "place", "open_gripper"
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: ActionStatus = ActionStatus.PENDING
    execution_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "parameters": self.parameters,
            "status": self.status.name,
        }


@dataclass
class AgentMessage:
    """Message passed between agents or components."""

    sender: str
    receiver: str
    content: Dict[str, Any]
    message_type: str  # e.g., "observation", "plan", "result", "反思"
    timestamp: float
    reply_to: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of executing an action or subtask."""

    success: bool
    output: Any = None
    error: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    reflection_summary: Optional[str] = None


@dataclass
class ReflectionResult:
    """Result of the reflection process."""

    assessment: str  # e.g., "on_track", "struggling", "failed"
    confidence: float  # 0.0 to 1.0
    issues_identified: List[str] = field(default_factory=list)
    suggested_adjustments: List[str] = field(default_factory=list)
    retry_recommended: bool = False
    replan_recommended: bool = False
    abort_recommended: bool = False


@dataclass
class Plan:
    """A plan consisting of ordered subtasks."""

    id: str
    goal_id: str
    subtasks: List[SubTask] = field(default_factory=list)
    current_subtask_index: int = 0
    created_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_current_subtask(self) -> Optional[SubTask]:
        if 0 <= self.current_subtask_index < len(self.subtasks):
            return self.subtasks[self.current_subtask_index]
        return None

    def advance(self) -> bool:
        """Advance to the next subtask. Returns True if there are more subtasks."""
        if self.current_subtask_index < len(self.subtasks) - 1:
            self.current_subtask_index += 1
            return True
        return False

    def is_complete(self) -> bool:
        return all(s.status == TaskStatus.COMPLETED for s in self.subtasks)

    def has_failures(self) -> bool:
        return any(s.status == TaskStatus.FAILED for s in self.subtasks)
