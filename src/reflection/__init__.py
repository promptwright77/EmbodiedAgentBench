"""Reflection module for self-assessment and meta-cognition."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.core.protocols import Reflector
from src.core.types import Action, Observation, ReflectionResult, TaskStatus


class SelfReflector:
    """Self-reflection module that assesses execution progress."""

    def __init__(self, llm: Optional[Any] = None) -> None:
        self.llm = llm
        self._reflection_history: List[ReflectionResult] = []

    def reflect(
        self,
        observation: Observation,
        action_result: Dict[str, Any],
    ) -> ReflectionResult:
        """Analyze execution result and provide reflection."""
        # Assess the current situation
        assessment = self._assess_situation(observation, action_result)
        confidence = self.assess_confidence(observation)
        issues = self._identify_issues(observation, action_result)
        adjustments = self._suggest_adjustments(observation, action_result, issues)

        retry_recommended = self.should_retry(observation, action_result)
        replan_recommended = len(issues) > 2 or confidence < 0.5
        abort_recommended = confidence < 0.2 and len(issues) > 3

        result = ReflectionResult(
            assessment=assessment,
            confidence=confidence,
            issues_identified=issues,
            suggested_adjustments=adjustments,
            retry_recommended=retry_recommended,
            replan_recommended=replan_recommended,
            abort_recommended=abort_recommended,
        )

        self._reflection_history.append(result)
        return result

    def assess_confidence(self, observation: Observation) -> float:
        """Assess confidence in the current plan execution."""
        # Base confidence on progress
        if not observation.completed_subtasks and observation.elapsed_steps > 3:
            return 0.3

        completion_rate = len(observation.completed_subtasks) / max(
            len(observation.completed_subtasks) + len(observation.failed_subtasks), 1
        )

        # Decay confidence over time without progress
        time_penalty = min(observation.elapsed_steps * 0.02, 0.3)

        confidence = completion_rate - time_penalty
        return max(0.0, min(1.0, confidence))

    def should_retry(self, observation: Observation, action_result: Dict[str, Any]) -> bool:
        """Determine if an action should be retried."""
        if not action_result.get("success", True):
            # Check if we've made progress recently
            if observation.elapsed_steps > 0 and observation.elapsed_steps % 2 == 0:
                if not observation.completed_subtasks:
                    return True
            return len(observation.failed_subtasks) < 3
        return False

    def _assess_situation(
        self,
        observation: Observation,
        action_result: Dict[str, Any],
    ) -> str:
        """Assess the current situation."""
        confidence = self.assess_confidence(observation)
        success = action_result.get("success", True)

        if confidence >= 0.7 and success:
            return "on_track"
        elif confidence >= 0.4:
            return "struggling"
        else:
            return "failed"

    def _identify_issues(
        self,
        observation: Observation,
        action_result: Dict[str, Any],
    ) -> List[str]:
        """Identify issues in the current execution."""
        issues = []

        # Check for repeated failures
        if len(observation.failed_subtasks) > 2:
            issues.append("Multiple subtasks have failed")

        # Check for lack of progress
        if observation.elapsed_steps > 5 and len(observation.completed_subtasks) == 0:
            issues.append("No progress made in recent steps")

        # Check for action failure
        if not action_result.get("success", True):
            issues.append(f"Action failed: {action_result.get('error', 'Unknown error')}")

        # Check for step limit
        if observation.elapsed_steps >= observation.goal.max_steps - 2:
            issues.append("Approaching step limit")

        return issues

    def _suggest_adjustments(
        self,
        observation: Observation,
        action_result: Dict[str, Any],
        issues: List[str],
    ) -> List[str]:
        """Suggest adjustments based on identified issues."""
        adjustments = []

        if "Multiple subtasks have failed" in issues:
            adjustments.append("Consider a completely different approach")

        if "No progress made in recent steps" in issues:
            adjustments.append("Break down the current subtask into smaller steps")

        if "Action failed" in str(issues):
            adjustments.append("Retry with different parameters or skip this step")

        if "Approaching step limit" in issues:
            adjustments.append("Prioritize completing the most critical subtasks")

        return adjustments


class LLMReflector:
    """LLM-powered reflection module for deeper analysis."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self._base_reflector = SelfReflector(llm)

    def reflect(
        self,
        observation: Observation,
        action_result: Dict[str, Any],
    ) -> ReflectionResult:
        """Use LLM to generate reflection."""
        # First get base reflection
        base_result = self._base_reflector.reflect(observation, action_result)

        # Enhance with LLM analysis
        prompt = f"""Analyze the following execution and provide refined reflection:

Observation: {observation.get_text_summary()}
Action Result: {action_result}
Issues Found: {base_result.issues_identified}
Suggested Adjustments: {base_result.suggested_adjustments}

Based on this analysis:
1. Is the current approach working? (on_track/struggling/failed)
2. What specific issues are most critical?
3. What adjustments would you recommend?

Respond with a JSON object containing: assessment, confidence (0-1), issues_identified, suggested_adjustments."""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            import json
            import re

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                llm_analysis = json.loads(json_match.group())
                return ReflectionResult(
                    assessment=llm_analysis.get("assessment", base_result.assessment),
                    confidence=llm_analysis.get("confidence", base_result.confidence),
                    issues_identified=llm_analysis.get(
                        "issues_identified", base_result.issues_identified
                    ),
                    suggested_adjustments=llm_analysis.get(
                        "suggested_adjustments", base_result.suggested_adjustments
                    ),
                    retry_recommended=base_result.retry_recommended,
                    replan_recommended=base_result.replan_recommended,
                    abort_recommended=base_result.abort_recommended,
                )
        except Exception:
            pass

        return base_result

    def assess_confidence(self, observation: Observation) -> float:
        return self._base_reflector.assess_confidence(observation)

    def should_retry(self, observation: Observation, action_result: Dict[str, Any]) -> bool:
        return self._base_reflector.should_retry(observation, action_result)


class ReflectionLogger:
    """Logger for tracking reflection history."""

    def __init__(self) -> None:
        self._logs: List[Dict[str, Any]] = []

    def log(self, observation: Observation, reflection: ReflectionResult) -> None:
        """Log a reflection event."""
        self._logs.append({
            "timestamp": time.time(),
            "observation": {
                "goal": observation.goal.description,
                "elapsed_steps": observation.elapsed_steps,
                "completed_subtasks": observation.completed_subtasks,
                "failed_subtasks": observation.failed_subtasks,
            },
            "reflection": {
                "assessment": reflection.assessment,
                "confidence": reflection.confidence,
                "issues_identified": reflection.issues_identified,
                "suggested_adjustments": reflection.suggested_adjustments,
            },
        })

    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all logged reflections."""
        return self._logs.copy()

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the n most recent reflections."""
        return self._logs[-n:] if len(self._logs) > n else self._logs.copy()
