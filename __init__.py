"""Minimal modular face swap framework."""

import warnings

# Silence noisy, non-actionable version-check warnings from albumentations
# when running in offline or firewalled environments.
warnings.filterwarnings(
    "ignore",
    message="Error fetching version info.*",
    category=UserWarning,
    module="albumentations.check_version",
)

from morgana.core.pipeline import FaceSwapPipeline
from morgana.detection.buffalo import BuffaloLDetector
from morgana.embedding.arcface import ArcFaceEmbedder
from morgana.swappers.hyperswap import HyperswapONNX
from morgana.swappers.inswapper import InSwapperONNX

__all__ = [
    "FaceSwapPipeline",
    "BuffaloLDetector",
    "ArcFaceEmbedder",
    "InSwapperONNX",
    "HyperswapONNX",
]
