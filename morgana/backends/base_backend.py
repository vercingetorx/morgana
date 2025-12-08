from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np

from morgana.core.pipeline import FaceSwapPipeline


class BaseBackend(ABC):
    """
    Shared backend implementation that owns a FaceSwapPipeline and exposes
    the minimal interface the CLI needs:

      - build_identity(...)
      - swap_image(...)
      - swap_video(...)

    Concrete backends (core / insightface) supply only the analyzer +
    model wiring by implementing _build_pipeline; all higher-level behavior
    (identity building, swapping) is shared here.
    """

    def __init__(
        self,
        *,
        device: str,
        swapper_backend: str,
        swapper_path: Path,
        restorer_name: str,
        identity_mode: str,
        restorer_visibility: float,
        mask_backend: str,
        face_index: Optional[int],
        max_faces: Optional[int],
        gate_identity,
        gate_enabled: bool,
        gate_threshold: float,
        refine_landmarks: bool,
    ) -> None:
        self.device = device
        self.swapper_backend = swapper_backend
        self.swapper_path = swapper_path
        self.restorer_name = restorer_name
        self.identity_mode = identity_mode
        self.restorer_visibility = restorer_visibility
        self.mask_backend = mask_backend
        self.face_index = face_index
        self.max_faces = max_faces
        self.gate_identity = gate_identity
        self.gate_enabled = gate_enabled
        self.gate_threshold = gate_threshold
        self.refine_landmarks = refine_landmarks

        self.pipeline = self._build_pipeline()

    @abstractmethod
    def _build_pipeline(self) -> FaceSwapPipeline:
        """Construct and return a FaceSwapPipeline for this backend."""
        raise NotImplementedError

    def build_identity(self, source_paths: List[Path]) -> dict[str, np.ndarray]:
        if self.identity_mode == "pose":
            return self.pipeline.compute_pose_identities_from_paths(source_paths)
        return self.pipeline.compute_identity_from_paths(source_paths)

    def swap_image(
        self,
        source_paths: List[Path],
        target_path: Path,
        output_path: Path,
        identity: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        self.pipeline.swap_image_file(source_paths, target_path, output_path, identity=identity)

    def swap_video(
        self,
        source_paths: List[Path],
        video_path: Path,
        output_path: Path,
        identity: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        self.pipeline.swap_video_file(source_paths, video_path, output_path, identity=identity)

