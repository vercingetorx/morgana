from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np


@dataclass
class Face:
    """
    Basic face container used across detectors/embedders/swappers.
    bbox: [x1, y1, x2, y2] in pixel space.
    landmarks: (5, 2) array of keypoints ordered as left-eye, right-eye, nose, left-mouth, right-mouth.
    landmarks_106: optional (106, 2) dense landmark set (e.g. buffalo_l 2d106det),
                   kept separate so that 5-point landmarks remain compatible with ArcFace/INSwapper.
    """

    bbox: np.ndarray
    landmarks: Optional[np.ndarray]
    landmarks_106: Optional[np.ndarray] = None
    score: float = 0.0
    embedding: Optional[np.ndarray] = None

    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) * 0.5, (self.bbox[1] + self.bbox[3]) * 0.5])


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def sort_and_select_faces(
    faces: List[Face],
    face_index: Optional[int] = None,
    max_faces: Optional[int] = None,
) -> List[Face]:
    """
    Deterministically select which faces to swap in a frame.

    Faces are sorted by bounding-box area, largest first. If face_index is
    not None, only that single face (by index in the sorted list) is kept.
    Otherwise, up to max_faces faces are kept from the head of the list.
    """
    if not faces:
        return []

    # Sort by bbox area descending.
    sorted_faces = sorted(
        faces,
        key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
        reverse=True,
    )

    if face_index is not None:
        if 0 <= face_index < len(sorted_faces):
            return [sorted_faces[face_index]]
        return []

    if max_faces is not None and max_faces >= 0:
        return sorted_faces[:max_faces]

    return sorted_faces
