from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from morgana import config
from morgana.core.pipeline import FaceSwapPipeline
from morgana.detection.buffalo import BuffaloLDetector
from morgana.detection.landmark106 import Landmark106Refiner
from morgana.embedding.arcface import ArcFaceEmbedder, prepare_buffalo_models
from morgana.detection.analyzer import BuffaloAnalyzer
from morgana.restoration.ultra_sharp import UltraSharpRestorer
from morgana.swappers.hyperswap import HyperswapONNX
from morgana.swappers.inswapper import InSwapperONNX
from morgana.swappers.reswapper import ReSwapperONNX
from .base_backend import BaseBackend


def build_core_pipeline(
    arcface_path: Path,
    swapper_path: Path,
    device: str,
    swapper_backend: str,
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
    Construct the core backend pipeline purely from local ONNX components:
    buffalo_l detector + ArcFace + chosen swapper (INSwapper/Hyperswap/ReSwapper)
    plus optional restoration. When refine_landmarks is True, run the optional
    106-point landmark refiner and derive 5-pt landmarks from it for alignment.
    """
    landmark_refiner = None
    if refine_landmarks:
        landmark_refiner = Landmark106Refiner(config.models_root() / "buffalo_l" / "2d106det.onnx")

    detector = BuffaloLDetector(config.detector_model_path(), device=device, landmark_refiner=landmark_refiner)
    embedder = ArcFaceEmbedder(arcface_path)
    analyzer = BuffaloAnalyzer(detector=detector, embedder=embedder)

    if swapper_backend == "hyperswap":
        swapper = HyperswapONNX(swapper_path)
    elif swapper_backend == "reswapper":
        swapper = ReSwapperONNX(swapper_path)
    else:
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


class CoreBackend(BaseBackend):
    """
    Backend implementation that uses only local ONNX components (buffalo_l +
    ArcFace + chosen swapper) wired through FaceSwapPipeline.
    """

    def _build_pipeline(self) -> FaceSwapPipeline:
        arcface_path = config.arcface_model_path()
        if arcface_path.suffix == ".zip":
            arcface_path = prepare_buffalo_models(arcface_path)
        return build_core_pipeline(
            arcface_path=arcface_path,
            swapper_path=self.swapper_path,
            device=self.device,
            swapper_backend=self.swapper_backend,
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
