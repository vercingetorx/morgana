from __future__ import annotations

from typing import List

import numpy as np

from morgana.core.types import Face


class BuffaloAnalyzer:
    """
    Minimal "analysis" wrapper for the buffalo_l detector + ArcFace embedder.

    Given an image, it:
      1. Runs the detector to produce Face objects (bbox, landmarks, score).
      2. For each face with landmarks, runs the embedder to attach a 512-d
         identity vector in Face.embedding.

    This mirrors InsightFace's FaceAnalysis shape, but uses our local ONNX
    models instead of any external dependency.
    """

    def __init__(self, detector, embedder) -> None:
        self.detector = detector
        self.embedder = embedder

    def analyze(self, image: np.ndarray) -> List[Face]:
        faces: List[Face] = self.detector.detect(image)
        for face in faces:
            if face.landmarks is None:
                continue
            if getattr(face, "embedding", None) is None:
                self.embedder.embed(image, face)
        return faces

