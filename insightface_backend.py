from __future__ import annotations

from pathlib import Path
from typing import List
import contextlib
import os
import sys
import warnings
import cv2
import numpy as np
try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        desc = kwargs.get("desc")
        if desc:
            print(f"{desc}...", flush=True)
        return iterable

from morgana import config
from morgana.utils import image as image_utils
from morgana.utils import video as video_utils
from morgana.restoration.ultra_sharp import UltraSharpRestorer
from morgana.core.types import Face as CoreFace
from morgana.swappers.hyperswap import HyperswapONNX
from morgana.swappers.inswapper import InSwapperONNX
from morgana.swappers.reswapper import ReSwapperONNX
from morgana.detection.landmark106 import Landmark106Refiner
from morgana.core.pipeline import FaceSwapPipeline
from .base_backend import BaseBackend


@contextlib.contextmanager
def _suppress_insightface_output():
    """
    Temporarily silence stdout/stderr for noisy InsightFace/ONNXRuntime logs.
    This avoids provider warnings and model discovery prints.
    """
    old_out, old_err = sys.stdout, sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull  # type: ignore[assignment]
            sys.stderr = devnull  # type: ignore[assignment]
            yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def _import_insightface():
    try:
        import insightface  # type: ignore
        from insightface.app import FaceAnalysis  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "InsightFace backend requested but 'insightface' is not installed. "
            "Install it with: pip install insightface"
        ) from exc
    # Silence common ONNXRuntime provider warnings emitted by InsightFace.
    warnings.filterwarnings(
        "ignore",
        message="Specified provider 'CUDAExecutionProvider' is not in available provider names.*",
        category=UserWarning,
    )
    return insightface, FaceAnalysis


def _make_app(device: str):
    """
    Build and prepare an InsightFace FaceAnalysis app configured for this
    project:

    - Uses the 'buffalo_l' model pack under models_root().parent.
    - Selects CPU (ctx_id = -1) or GPU 0 (ctx_id = 0) based on the device
      string ('cpu' / 'cuda').
    - Calls app.prepare(det_size=(640, 640)) with noisy stdout/stderr
      temporarily suppressed.

    This is the only InsightFace-specific entry point; everything downstream
    uses InsightFaceAnalyzer + FaceSwapPipeline just like the core backend.
    """
    insightface, FaceAnalysis = _import_insightface()
    # Root directory should contain a 'models' subfolder with buffalo_l.zip.
    root = config.models_root().parent
    with _suppress_insightface_output():
        app = FaceAnalysis(name="buffalo_l", root=str(root))
        # ctx_id: -1 = CPU, >=0 = GPU index
        ctx_id = -1
        if device and device.startswith("cuda"):
            ctx_id = 0
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def _insight_face_to_core(face) -> CoreFace | None:
    """
    Convert an InsightFace Face object into our lightweight CoreFace so it can
    be consumed by HyperswapONNX, which expects 5-point landmarks in the
    standard ArcFace order (left-eye, right-eye, nose, left-mouth, right-mouth).
    """
    try:
        bbox = np.asarray(getattr(face, "bbox"), dtype=np.float32)
    except Exception:
        return None

    lm = None
    if hasattr(face, "landmark_5") and getattr(face, "landmark_5") is not None:
        lm = np.asarray(face.landmark_5, dtype=np.float32)
    elif hasattr(face, "kps") and getattr(face, "kps") is not None:
        lm = np.asarray(face.kps, dtype=np.float32)
    if lm is None or lm.shape != (5, 2):
        return None

    score = float(getattr(face, "det_score", 1.0))

    emb = None
    if hasattr(face, "normed_embedding") and getattr(face, "normed_embedding") is not None:
        emb = np.asarray(face.normed_embedding, dtype=np.float32)
    elif hasattr(face, "embedding") and getattr(face, "embedding") is not None:
        e = np.asarray(face.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(e) + 1e-12)
        emb = e / norm

    return CoreFace(bbox=bbox, landmarks=lm, landmarks_106=None, score=score, embedding=emb)


class InsightFaceAnalyzer:
    """
    Analyzer wrapper around InsightFace FaceAnalysis that produces CoreFace
    objects compatible with the core pipeline. This allows us to reuse the
    same identity-building logic across backends by delegating to
    FaceSwapPipeline.compute_* methods.
    """

    def __init__(self, app, refine_landmarks: bool = False) -> None:
        self.app = app
        self.refiner: Landmark106Refiner | None = None
        if refine_landmarks:
            model_path_106 = config.models_root() / "buffalo_l" / "2d106det.onnx"
            if model_path_106.exists():
                self.refiner = Landmark106Refiner(model_path_106)

    def analyze(self, image: np.ndarray) -> list[CoreFace]:
        faces = self.app.get(image)
        core_faces: list[CoreFace] = []
        for f in faces:
            core_face = _insight_face_to_core(f)
            if core_face is not None:
                core_faces.append(core_face)
        if self.refiner is not None and core_faces:
            self.refiner.refine_faces_in_image(image, core_faces)
        return core_faces


def build_insightface_pipeline(
    device: str,
    swapper_backend: str,
    swapper_path: Path,
    restorer_name: str,
    identity_mode: str,
    restorer_visibility: float,
    mask_backend: str,
    face_index: int | None = None,
    max_faces: int | None = None,
    gate_identity=None,
    gate_enabled: bool = False,
    gate_threshold: float = 0.25,
    refine_landmarks: bool = False,
) -> FaceSwapPipeline:
    """
    Construct a FaceSwapPipeline driven by InsightFace FaceAnalysis:
    - InsightFace provides detection + embeddings via InsightFaceAnalyzer.
    - Swapper is one of INSwapper/Hyperswap/ReSwapper.
    - Restorer (optional) is an UltraSharpRestorer.
    """
    app = _make_app(device=device)
    analyzer = InsightFaceAnalyzer(app, refine_landmarks=refine_landmarks)

    if swapper_backend == "hyperswap":
        swapper = HyperswapONNX(swapper_path)
    elif swapper_backend == "reswapper":
        swapper = ReSwapperONNX(swapper_path)
    else:
        # Default: INSwapper ONNX, same wrapper as the core backend.
        swapper = InSwapperONNX(swapper_path)

    if restorer_name != "none":
        restorer_model = config.restorer_model_path(restorer_name)
        restorer = UltraSharpRestorer(model_path=restorer_model)
    else:
        restorer = None

    return FaceSwapPipeline(
        swapper=swapper,
        restorer=restorer,
        restorer_visibility=restorer_visibility,
        identity_mode=identity_mode,
        mask_backend=mask_backend,
        device=device,
        analyzer=analyzer,
        face_index=face_index,
        max_faces=max_faces,
        gate_identity=gate_identity,
        gate_enabled=gate_enabled,
        gate_threshold=gate_threshold,
    )


class InsightFaceBackend(BaseBackend):
    """
    Backend implementation that wraps InsightFace FaceAnalysis but exposes the
    same high-level interface as the core backend for the CLI.
    """

    def _build_pipeline(self) -> FaceSwapPipeline:
        return build_insightface_pipeline(
            device=self.device,
            swapper_backend=self.swapper_backend,
            swapper_path=self.swapper_path,
            restorer_name=self.restorer_name,
            identity_mode=self.identity_mode,
            restorer_visibility=self.restorer_visibility,
            mask_backend=self.mask_backend,
            face_index=self.face_index,
            max_faces=self.max_faces,
            gate_identity=self.gate_identity,
            gate_enabled=self.gate_enabled,
            gate_threshold=self.gate_threshold,
            refine_landmarks=self.refine_landmarks,
        )
