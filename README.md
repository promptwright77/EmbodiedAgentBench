# EmbodiedAgentBench

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Orchestration-1C3C3C.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive **embodied AI agent framework** implementing the **perception-planning-action-reflection闭环** (closed loop) for robotics and simulation environments.

## Overview

EmbodiedAgentBench provides a complete framework for building embodied AI agents that can:

- **Perceive** the environment through multi-modal sensors (vision, language, proprioception)
- **Plan** task decomposition using LLM-based or deterministic planners
- **Act** by executing actions in simulated or real environments
- **Reflect** on execution results for self-assessment and adaptive behavior

This framework is designed for researchers and engineers working on robotics, manipulation tasks, and agent-based AI systems.

## Key Features

- **Modular Architecture**: Clean separation of concerns with Protocol-based interfaces
- **Multi-Modal Perception**: Fuses inputs from camera, lidar, tactile sensors, and text
- **Flexible Planning**: Supports both LLM-based and deterministic task planners
- **Tool Use System**: Extensible registry for agent tools and functions
- **Reflection Mechanism**: Self-assessment and confidence evaluation
- **Long-term Memory**: Persistent experience storage with SQLite backend
- **Multi-Agent Collaboration**: Team-based architecture for coordinated agents
- **Layered Evaluation**: Mock → Replay → Simulation → Real evaluation pipeline

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EmbodiedAgentBench                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Perception  │→ │  Planning   │→ │  Execution  │          │
│  │   Module    │  │   Module    │  │   Module    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│         ↑                                    ↓              │
│         │            ┌─────────────┐      ┌─────────────┐   │
│         └────────────│  Reflection │←─────│   Memory    │   │
│                      │   Module    │      │   Module    │   │
│                      └─────────────┘      └─────────────┘   │
│         ┌─────────────────────────────────────────────┐     │
│         │           Multi-Agent Coordinator            │     │
│         └─────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/he-buyiyang/EmbodiedAgentBench.git
cd EmbodiedAgentBench

# Install with uv (recommended)
uv sync

# Install with pip
pip install -e .
```

## Quick Start

```python
from src.core.types import TaskGoal, Observation
from src.planning import DeterministicPlanner
from src.execution import MockActionExecutor
from src.reflection import SelfReflector
from src.memory import InMemoryStore

# Create components
planner = DeterministicPlanner()
executor = MockActionExecutor()
reflector = SelfReflector()
memory = InMemoryStore()

# Define a task
goal = TaskGoal(
    id="task-1",
    description="Pick up the red cube and place it in the blue bin",
    max_steps=20,
)

# Create observation and plan
observation = Observation(goal=goal)
plan = planner.create_plan(goal, observation)

# Execute and reflect
for subtask in plan.subtasks:
    action = {"type": "grasp", "target": "red_cube"}
    result = executor.execute(action)
    memory.store_experience(observation, action, result)
    reflection = reflector.reflect(observation, result)
```

## Running the Demo

```bash
# Run all demos
make run-demo

# Run specific demo modes
uv run python scripts/run_demo.py --mode mock
uv run python scripts/run_demo.py --mode multi-agent
uv run python scripts/run_demo.py --mode reflection
```

## Project Structure

```
EmbodiedAgentBench/
├── src/
│   ├── core/           # Core types and protocols
│   ├── perception/     # Multi-modal perception modules
│   ├── planning/       # Task planning and decomposition
│   ├── execution/      # Action execution and tools
│   ├── reflection/     # Self-reflection and assessment
│   ├── memory/         # Persistent memory and experience
│   ├── multi_agent/    # Multi-agent collaboration
│   └── evaluation/     # Evaluation framework
├── tests/              # Unit tests
├── scripts/            # Demo and utility scripts
├── docs/               # Documentation
└── configs/            # Configuration files
```

## Evaluation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Mock** | Deterministic evaluation with mock executors | Fast testing, CI |
| **Replay** | Evaluate on pre-recorded episodes | Benchmarking |
| **Simulation** | Closed-loop with simplified physics | Algorithm development |
| **Real** | Real robot evaluation | Deployment validation |

## Documentation

- [Detailed Explanation](detailed-explanation.md) - Comprehensive project documentation
- [Architecture Guide](docs/architecture.md) - System architecture details
- [API Reference](docs/api.md) - Module and function documentation

## Evaluation Framework

The framework includes a comprehensive evaluation system:

```python
from src.evaluation import EvaluationRunner, EvaluationMode

runner = EvaluationRunner(mode=EvaluationMode.MOCK)
metrics = runner.evaluate(agent, test_cases)

print(f"Success Rate: {metrics.task_success_rate * 100:.1f}%")
print(f"Note: {metrics.evaluation_note}")
```

## Multi-Agent Collaboration

```python
from src.multi_agent import AgentTeam, PerceptionAgent, PlanningAgent

team = AgentTeam()
team.register_agent(PerceptionAgent())
team.register_agent(PlanningAgent())

team.send_message(
    sender_id="planning_agent",
    receiver_id="perception_agent",
    content={"task": "Analyze scene"},
    message_type="request",
)
team.synchronize()
```

## Testing

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=term-missing
```

## Dependencies

- **Core**: `pydantic>=2.9.0`, `numpy>=1.26.0`
- **Agent**: `langgraph>=0.2.0`, `langchain-core>=0.3.0`
- **Evaluation**: `pandas>=2.2.0`, `matplotlib>=3.9.0`
- **Dev**: `pytest>=8.3.0`, `ruff>=0.6.0`, `mypy>=1.13.0`

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**he-buyiyang** - MartensonCordona20@outlook.com

## Acknowledgments

This project draws inspiration from:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration
- [LeRobot](https://github.com/huggingface/lerobot) - Robotics toolkit
- Academic research on embodied AI and agent systems
