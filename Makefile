.PHONY: setup check test test-unit clean run-demo run-evaluation lint format

setup:
	uv sync

setup-dev:
	uv sync --extra dev

setup-agent:
	uv sync --extra agent

setup-evaluation:
	uv sync --extra evaluation

check:
	uv run ruff check src/
	uv run mypy src/

test-unit:
	uv run pytest tests/unit/ -v

test:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

test-verbose:
	uv run pytest tests/ -v -s

run-demo:
	uv run python scripts/run_demo.py

run-evaluation:
	uv run python scripts/run_evaluation.py

run-agent:
	uv run python scripts/run_agent.py --goal "pick up the red cube and place it in the blue bin"

lint:
	uv run ruff check src/ --fix

format:
	uv run ruff format src/

clean:
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
