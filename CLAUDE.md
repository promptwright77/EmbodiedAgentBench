# EmbodiedAgentBench

## Project Overview

EmbodiedAgentBench is a comprehensive embodied AI agent framework implementing the **perception-planning-action-reflection closed loop** for robotics and simulation environments.

## Architecture

```
src/
├── core/           # Types and Protocols (typed interfaces)
├── perception/     # Multi-modal sensory processing
├── planning/       # Task decomposition and planning
├── execution/      # Action execution and tools
├── reflection/     # Self-assessment and meta-cognition
├── memory/         # Persistent experience storage
├── multi_agent/    # Multi-agent collaboration
└── evaluation/     # Layered evaluation framework
```

## Key Commands

```bash
make setup          # Install dependencies
make check          # Run linter and type checker
make test           # Run all tests
make run-demo       # Run demonstration
```

## Design Principles

1. **Protocol-based interfaces** - All components implement typed Protocols for dependency injection
2. **No dict boundaries** - All data flows through Pydantic-validated dataclasses
3. **Evaluation honesty** - Every metric includes what it does NOT prove
4. **Layered evaluation** - Mock → Replay → Simulation → Real

## Evaluation Modes

| Mode | Purpose |
|------|---------|
| MOCK | Fast deterministic testing |
| REPLAY | Pre-recorded episode evaluation |
| SIMULATION | Closed-loop with simplified physics |
| REAL | Real robot deployment |
