from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from morgana.core.alignment import estimate_affine, warp_face
from morgana.core.types import Face, resolve_path
from morgana.utils.model_loader import build_session, ensure_unzipped


class ArcFaceEmbedder:
    """ONNX ArcFace embedder using the supplied buffalo_l backbone."""

    def __init__(self, model_path: str | Path, providers: Optional[List[str]] = None):
        self.model_path = resolve_path(model_path)
        if self.model_path.suffix == ".zip":
            self.model_path = prepare_buffalo_models(self.model_path)
        self.session = build_session(self.model_path, providers=providers)
        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        input_shape = input_cfg.shape
        # ArcFaceONNX uses input_mean/input_std inferred from the graph;
        # for buffalo_l w600k_r50 this is typically 127.5 / 127.5.
        self.input_mean = 127.5
        self.input_std = 127.5
        # Infer input spatial size from the ONNX input tensor.
        if len(input_shape) == 4 and isinstance(input_shape[2], int) and input_shape[2] > 0:
            self.input_size = int(input_shape[2])
        else:
            self.input_size = 112

    def preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess crop for ArcFace: mirror InsightFace ArcFaceONNX.
        Uses cv2.dnn.blobFromImages with (x - input_mean) / input_std and RGB ordering.
        """
        input_size = (self.input_size, self.input_size)
        blob = cv2.dnn.blobFromImages(
            [face_img],
            scalefactor=1.0 / self.input_std,
            size=input_size,
            mean=(self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        return blob

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        if face.landmarks is None:
            raise ValueError("Landmarks required for ArcFace embedding")
        M = estimate_affine(face.landmarks, image_size=self.input_size)
        aligned = warp_face(image, M, image_size=self.input_size)
        blob = self.preprocess(aligned)
        feat = self.session.run(None, {self.input_name: blob})[0][0]
        norm = np.linalg.norm(feat) + 1e-12
        emb = feat / norm
        # Store embedding on the Face object for downstream reuse (e.g. identity gating),
        # mirroring InsightFace FaceAnalysis behavior.
        try:
            face.embedding = emb
        except Exception:
            pass
        return emb


def prepare_buffalo_models(zip_path: Path) -> Path:
    """Extract the buffalo_l bundle and return the ArcFace path."""
    target_dir = zip_path.parent
    ensure_unzipped(zip_path, target_dir)
    arcface_path = target_dir / "w600k_r50.onnx"
    if not arcface_path.exists():
        raise FileNotFoundError(f"ArcFace model missing after extraction: {arcface_path}")
    return arcface_path
