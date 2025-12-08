from __future__ import annotations

import os
from pathlib import Path

from morgana.utils.model_download import ensure_model_file


def models_root() -> Path:
    env = os.environ.get("FACESWAP_MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parent
    # Default models directory lives at the project root: ./models
    return (here.parent / "models").resolve()


def default_detector_path() -> Path:
    # Buffalo_l detector (SCRFD det_10g.onnx)
    path = models_root() / "buffalo_l" / "det_10g.onnx"
    return ensure_model_file(path, models_root())


def default_arcface_path() -> Path:
    # Prefer the unpacked ArcFace ONNX (w600k_r50.onnx) that ships in
    # models/buffalo_l. Older layouts may still use buffalo_l.zip; callers
    # that want to support that can pass a .zip path directly to ArcFaceEmbedder.
    path = models_root() / "buffalo_l" / "w600k_r50.onnx"
    return ensure_model_file(path, models_root())


def default_swapper_path() -> Path:
    path = models_root() / "insightface" / "inswapper_128.onnx"
    return ensure_model_file(path, models_root())


def default_hyperswap_path() -> Path:
    """
    Default Hyperswap model path (non-NSFW variant).
    """
    path = models_root() / "hyperswap" / "hyperswap_1a_256.onnx"
    return ensure_model_file(path, models_root())


def hyperswap_model_path(name: str) -> Path:
    """
    Map a short Hyperswap model name to a concrete ONNX model.

    Known names (NSFW variants are intentionally excluded for now):
      - 1a  -> hyperswap_1a_256.onnx
      - 1b  -> hyperswap_1b_256.onnx
      - 1c  -> hyperswap_1c_256.onnx
    """
    base = models_root() / "hyperswap"
    mapping = {
        "1a": base / "hyperswap_1a_256.onnx",
        "1b": base / "hyperswap_1b_256.onnx",
        "1c": base / "hyperswap_1c_256.onnx",
    }
    if name not in mapping:
        raise ValueError(f"Unknown Hyperswap model '{name}'")
    return ensure_model_file(mapping[name], models_root())


def reswapper_model_path(name: str) -> Path:
    """
    Map a short ReSwapper model name to a concrete ONNX model under models/reswapper.

    Known names:
      - 128 -> reswapper_128.onnx
      - 256 -> reswapper_256.onnx
    """
    base = models_root() / "reswapper"
    mapping = {
        "128": base / "reswapper_128.onnx",
        "256": base / "reswapper_256.onnx",
    }
    if name not in mapping:
        raise ValueError(f"Unknown ReSwapper model '{name}'")
    return ensure_model_file(mapping[name], models_root())


def restorer_model_path(name: str) -> Path:
    """
    Map a short restorer name to a concrete ONNX model under facerestore.

    Known names:
      - gpen-512, gpen-1024, gpen-2048
      - gfpgan-14
      - restoreformer-pp
      - codeformer
      - ultra-sharp-x4 (generic image upscaler)
    """
    base = models_root()
    mapping = {
        "gpen-512": base / "facerestore" / "GPEN-BFR-512.onnx",
        "gpen-1024": base / "facerestore" / "GPEN-BFR-1024.onnx",
        "gpen-2048": base / "facerestore" / "GPEN-BFR-2048.onnx",
        "gfpgan-14": base / "facerestore" / "GFPGANv1.4.onnx",
        "restoreformer-pp": base / "facerestore" / "RestoreFormer_PP.onnx",
        "codeformer": base / "facerestore" / "CodeFormer_512.onnx",
        "ultra-sharp-x4": base / "facerestore" / "ultra_sharp_2_x4.onnx",
    }
    if name not in mapping:
        raise ValueError(f"Unknown restorer model '{name}'")
    return ensure_model_file(mapping[name], models_root())
