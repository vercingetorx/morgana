from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Protocol, Type

from .core_backend import CoreBackend
from .insightface_backend import InsightFaceBackend


class BackendProtocol(Protocol):
    """
    Minimal interface that all backends must implement so that the CLI can
    drive them in a backend-agnostic way.
    """

    def build_identity(self, source_paths: List[Path]) -> dict[str, object]:
        ...

    def swap_image(
        self,
        source_paths: List[Path],
        target_path: Path,
        output_path: Path,
        identity: Optional[dict[str, object]] = None,
    ) -> None:
        ...

    def swap_video(
        self,
        source_paths: List[Path],
        video_path: Path,
        output_path: Path,
        identity: Optional[dict[str, object]] = None,
    ) -> None:
        ...


BackendType = Type[BackendProtocol]

BACKENDS: Dict[str, BackendType] = {
    "core": CoreBackend,
    "insightface": InsightFaceBackend,
}

__all__ = ["BACKENDS", "BackendProtocol", "CoreBackend", "InsightFaceBackend"]
