"""Perception module for multi-modal sensory processing."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.core.protocols import PerceptionModule
from src.core.types import Percept, PerceptionSource


class MultiModalPerceiver:
    """Multi-modal perception module that fuses inputs from different sensors."""

    def __init__(self) -> None:
        self._sources: Dict[PerceptionSource, PerceptionModule] = {}

    def register_source(
        self,
        source: PerceptionSource,
        module: PerceptionModule,
    ) -> None:
        """Register a perception module for a specific source."""
        self._sources[source] = module

    def perceive(self, raw_inputs: Dict[PerceptionSource, Any]) -> List[Percept]:
        """Process raw inputs from multiple sources into percepts."""
        percepts = []
        for source, raw_input in raw_inputs.items():
            if source in self._sources:
                source_percepts = self._sources[source].perceive(raw_input)
                percepts.extend(source_percepts)
            else:
                # Default: wrap raw input as a percept
                percepts.append(
                    Percept(
                        source=source,
                        data=raw_input,
                        timestamp=time.time(),
                        confidence=1.0,
                    )
                )
        return percepts

    def fuse(self, percepts: List[Percept]) -> Percept:
        """Fuse multiple percepts into a unified representation."""
        if not percepts:
            raise ValueError("Cannot fuse empty percepts list")
        if len(percepts) == 1:
            return percepts[0]

        # Simple fusion: concatenate text descriptions
        fused_text = " | ".join(p.get_text_description() for p in percepts)
        avg_confidence = sum(p.confidence for p in percepts) / len(percepts)

        return Percept(
            source=PerceptionSource.VISION_LANGUAGE,
            data=fused_text,
            timestamp=time.time(),
            confidence=avg_confidence,
            metadata={
                "fused_sources": [p.source.name for p in percepts],
                "percept_count": len(percepts),
            },
        )


class TextPerceiver:
    """Simple text-based perception module."""

    def perceive(self, raw_input: str) -> List[Percept]:
        """Process text input into percepts."""
        return [
            Percept(
                source=PerceptionSource.TEXT,
                data=raw_input,
                timestamp=time.time(),
                confidence=1.0,
            )
        ]

    def fuse(self, percepts: List[Percept]) -> Percept:
        """Fuse text percepts."""
        fused_text = " ".join(p.data for p in percepts if p.source == PerceptionSource.TEXT)
        return percepts[0].model_copy(update={"data": fused_text})


class VisionLanguagePerceiver:
    """Vision-language perception module for processing image + text."""

    def __init__(self, model_name: str = "clip-vit-base-patch32") -> None:
        self.model_name = model_name
        self._initialized = False

    def perceive(self, raw_input: Dict[str, Any]) -> List[Percept]:
        """Process image and optional text into percepts."""
        # raw_input: {"image": np.ndarray, "text": Optional[str]}
        image = raw_input.get("image")
        text = raw_input.get("text", "")

        percepts = []

        if image is not None:
            percepts.append(
                Percept(
                    source=PerceptionSource.VISION_LANGUAGE,
                    data={
                        "image_shape": image.shape if hasattr(image, "shape") else "unknown",
                        "text": text,
                    },
                    timestamp=time.time(),
                    confidence=0.9,
                    metadata={"description": f"Vision input: {text}"},
                )
            )

        if text:
            percepts.append(
                Percept(
                    source=PerceptionSource.TEXT,
                    data=text,
                    timestamp=time.time(),
                    confidence=1.0,
                )
            )

        return percepts

    def fuse(self, percepts: List[Percept]) -> Percept:
        """Fuse vision and language percepts."""
        vl_percepts = [p for p in percepts if p.source == PerceptionSource.VISION_LANGUAGE]
        text_percepts = [p for p in percepts if p.source == PerceptionSource.TEXT]

        fused_data = {
            "vision": vl_percepts[0].data if vl_percepts else None,
            "text": " ".join(p.data for p in text_percepts),
        }

        return Percept(
            source=PerceptionSource.VISION_LANGUAGE,
            data=fused_data,
            timestamp=time.time(),
            confidence=sum(p.confidence for p in percepts) / len(percepts) if percepts else 0.0,
            metadata={"fused": True},
        )


class SimulatorPerceiver:
    """Perception module for simulator environment."""

    def perceive(self, raw_input: Dict[str, Any]) -> List[Percept]:
        """Process simulator state into percepts."""
        # raw_input: {"state": dict, "objects": list, "robot_pos": tuple}
        state = raw_input.get("state", {})
        objects = raw_input.get("objects", [])
        robot_pos = raw_input.get("robot_pos")

        percepts = []

        # State percept
        percepts.append(
            Percept(
                source=PerceptionSource.SIMULATOR,
                data=state,
                timestamp=time.time(),
                confidence=1.0,
                metadata={"description": f"Simulator state: {state}"},
            )
        )

        # Objects percept
        if objects:
            object_descriptions = [
                f"{obj.get('type', 'unknown')} at {obj.get('position', 'unknown')}"
                for obj in objects
            ]
            percepts.append(
                Percept(
                    source=PerceptionSource.SIMULATOR,
                    data={"objects": objects},
                    timestamp=time.time(),
                    confidence=1.0,
                    metadata={"description": "; ".join(object_descriptions)},
                )
            )

        # Robot position percept
        if robot_pos is not None:
            percepts.append(
                Percept(
                    source=PerceptionSource.SIMULATOR,
                    data={"robot_position": robot_pos},
                    timestamp=time.time(),
                    confidence=1.0,
                    metadata={"description": f"Robot at {robot_pos}"},
                )
            )

        return percepts

    def fuse(self, percepts: List[Percept]) -> Percept:
        """Fuse simulator percepts."""
        fused_state = {}
        fused_objects = []
        fused_robot_pos = None

        for p in percepts:
            if p.source == PerceptionSource.SIMULATOR:
                data = p.data
                if isinstance(data, dict):
                    if "state" in data:
                        fused_state.update(data["state"])
                    if "objects" in data:
                        fused_objects.extend(data["objects"])
                    if "robot_position" in data:
                        fused_robot_pos = data["robot_position"]

        return Percept(
            source=PerceptionSource.SIMULATOR,
            data={
                "state": fused_state,
                "objects": fused_objects,
                "robot_position": fused_robot_pos,
            },
            timestamp=time.time(),
            confidence=1.0,
            metadata={"fused": True},
        )
