"""Memory module for persistent storage and experience replay."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.protocols import MemoryStore
from src.core.types import Action, Observation


class InMemoryStore:
    """In-memory storage implementation."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}
        self._experience_store: List[Dict[str, Any]] = []

    def store(self, key: str, value: Any) -> None:
        """Store a value in memory."""
        self._store[key] = value

    def retrieve(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from memory."""
        return self._store.get(key, default)

    def store_experience(
        self,
        observation: Observation,
        action: Action,
        result: Dict[str, Any],
    ) -> None:
        """Store a complete experience tuple."""
        experience = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "goal_description": observation.goal.description,
            "elapsed_steps": observation.elapsed_steps,
            "action_type": action.type,
            "action_parameters": action.parameters,
            "action_result": result,
            "success": result.get("success", False),
        }
        self._experience_store.append(experience)

    def retrieve_relevant(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve experiences relevant to a query."""
        query_lower = query.lower()
        relevant = [
            exp
            for exp in self._experience_store
            if query_lower in exp.get("goal_description", "").lower()
        ]
        return relevant[-limit:] if len(relevant) > limit else relevant

    def clear(self) -> None:
        """Clear all stored memories."""
        self._store.clear()
        self._experience_store.clear()


class SQLiteMemoryStore:
    """SQLite-based persistent memory store."""

    def __init__(self, db_path: str = ".embodied_agent_memory.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                goal_description TEXT NOT NULL,
                elapsed_steps INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                action_parameters TEXT NOT NULL,
                action_result TEXT NOT NULL,
                success INTEGER NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiences_goal
            ON experiences(goal_description)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiences_timestamp
            ON experiences(timestamp)
        """)

        conn.commit()
        conn.close()

    def store(self, key: str, value: Any) -> None:
        """Store a value in memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        json_value = json.dumps(value)
        now = time.time()

        cursor.execute(
            """
            INSERT OR REPLACE INTO memories (key, value, created_at, updated_at)
            VALUES (?, ?, COALESCE((SELECT created_at FROM memories WHERE key = ?), ?), ?)
            """,
            (key, json_value, key, now, now),
        )

        conn.commit()
        conn.close()

    def retrieve(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT value FROM memories WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return default

        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return row[0]

    def store_experience(
        self,
        observation: Observation,
        action: Action,
        result: Dict[str, Any],
    ) -> None:
        """Store a complete experience tuple."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO experiences
            (id, timestamp, goal_description, elapsed_steps, action_type,
             action_parameters, action_result, success)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                time.time(),
                observation.goal.description,
                observation.elapsed_steps,
                action.type,
                json.dumps(action.parameters),
                json.dumps(result),
                1 if result.get("success", False) else 0,
            ),
        )

        conn.commit()
        conn.close()

    def retrieve_relevant(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve experiences relevant to a query."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM experiences
            WHERE goal_description LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (f"%{query}%", limit),
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "goal_description": row[2],
                "elapsed_steps": row[3],
                "action_type": row[4],
                "action_parameters": json.loads(row[5]),
                "action_result": json.loads(row[6]),
                "success": bool(row[7]),
            }
            for row in rows
        ]

    def retrieve_similar(
        self,
        action_type: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve experiences with similar action types."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM experiences
            WHERE action_type = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (action_type, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "goal_description": row[2],
                "elapsed_steps": row[3],
                "action_type": row[4],
                "action_parameters": json.loads(row[5]),
                "action_result": json.loads(row[6]),
                "success": bool(row[7]),
            }
            for row in rows
        ]

    def clear(self) -> None:
        """Clear all stored memories."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories")
        cursor.execute("DELETE FROM experiences")
        conn.commit()
        conn.close()


class ExperienceBuffer:
    """Buffer for managing agent experiences with prioritization."""

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._buffer: List[Dict[str, Any]] = []

    def add(
        self,
        observation: Observation,
        action: Action,
        result: Dict[str, Any],
    ) -> None:
        """Add an experience to the buffer."""
        experience = {
            "observation": observation,
            "action": action,
            "result": result,
            "timestamp": time.time(),
            "priority": self._calculate_priority(observation, result),
        }
        self._buffer.append(experience)

        # Evict lowest priority if over capacity
        if len(self._buffer) > self.max_size:
            self._buffer.sort(key=lambda x: x["priority"])
            self._buffer.pop(0)

    def sample(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Sample a batch of experiences, prioritizing high-value ones."""
        if len(self._buffer) <= batch_size:
            return self._buffer.copy()

        # Prioritize experiences with failures (for learning)
        prioritized = sorted(self._buffer, key=lambda x: x["priority"], reverse=True)
        return prioritized[:batch_size]

    def _calculate_priority(
        self,
        observation: Observation,
        result: Dict[str, Any],
    ) -> float:
        """Calculate priority for an experience."""
        priority = 1.0

        # Higher priority for failures (learning opportunity)
        if not result.get("success", True):
            priority += 2.0

        # Higher priority for completions
        if observation.completed_subtasks:
            priority += 1.0

        # Higher priority for recent experiences
        recency = min(observation.elapsed_steps / observation.goal.max_steps, 1.0)
        priority += recency

        return priority

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the experience buffer."""
        if not self._buffer:
            return {"total": 0, "success_rate": 0.0, "avg_priority": 0.0}

        successes = sum(1 for e in self._buffer if e["result"].get("success", False))
        avg_priority = sum(e["priority"] for e in self._buffer) / len(self._buffer)

        return {
            "total": len(self._buffer),
            "success_rate": successes / len(self._buffer),
            "avg_priority": avg_priority,
        }
