from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from morgana.restoration.base import BaseRestorer
from morgana.utils.model_loader import build_session


class UltraSharpRestorer(BaseRestorer):
    """
    Generic ONNX-based face restoration/upscaling wrapper.

    The underlying model is expected to take an RGB image normalized to
    [-1, 1] and output an RGB image normalized to [-1, 1]. Many face
    restorers expect a fixed spatial size (e.g. 512x512); we handle that
    by resizing the input to the model's expected size and then resizing
    the output to the requested scale of the original frame.
    """

    def __init__(self, model_path: Optional[Path] = None, providers: Optional[list[str]] = None):
        if model_path is None:
            raise ValueError("UltraSharpRestorer requires an explicit model_path")
        self.model_path = model_path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Restoration model not found at '{self.model_path}'")
        self.session = build_session(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        name_lower = self.model_path.name.lower()
        # Some models (e.g., ultra_sharp_x4) appear to be BGR-trained and
        # operate in [0, 1] rather than [-1, 1]. Treat those specially to
        # avoid purple/shifted colors.
        self._bgr_model = "ultra_sharp" in name_lower
        # Cache the model's expected spatial size if it is fixed.
        input_cfg = self.session.get_inputs()[0]
        shape = list(getattr(input_cfg, "shape", []))
        self._input_h: Optional[int] = None
        self._input_w: Optional[int] = None
        if len(shape) == 4:
            h, w = shape[2], shape[3]
            if isinstance(h, int) and h > 0 and isinstance(w, int) and w > 0:
                self._input_h = h
                self._input_w = w

    @staticmethod
    def _preprocess_rgb(image: np.ndarray) -> np.ndarray:
        # BGR uint8 -> RGB float32 in [-1, 1], NCHW
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        blob = rgb.transpose(2, 0, 1)[None, ...]
        return blob

    @staticmethod
    def _preprocess_bgr(image: np.ndarray) -> np.ndarray:
        # BGR uint8 -> BGR float32 in [-1, 1], NCHW (no channel swap)
        bgr = image.astype(np.float32) / 255.0
        bgr = (bgr - 0.5) / 0.5
        blob = bgr.transpose(2, 0, 1)[None, ...]
        return blob

    @staticmethod
    def _preprocess_bgr_01(image: np.ndarray) -> np.ndarray:
        # BGR uint8 -> BGR float32 in [0, 1], NCHW (no channel swap)
        bgr = image.astype(np.float32) / 255.0
        blob = bgr.transpose(2, 0, 1)[None, ...]
        return blob

    @staticmethod
    def _postprocess_rgb(out: np.ndarray) -> np.ndarray:
        # NCHW [-1, 1] -> BGR uint8
        img = out[0].transpose(1, 2, 0)
        img = (img * 0.5 + 0.5) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    @staticmethod
    def _postprocess_bgr(out: np.ndarray) -> np.ndarray:
        # NCHW [-1, 1] -> BGR uint8 (no channel swap)
        img = out[0].transpose(1, 2, 0)
        img = (img * 0.5 + 0.5) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def _postprocess_bgr_01(out: np.ndarray) -> np.ndarray:
        # NCHW [0, 1] -> BGR uint8 (no channel swap)
        img = out[0].transpose(1, 2, 0)
        img = np.clip(img, 0.0, 1.0) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def restore(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        # If the model expects a fixed spatial size, resize to that first.
        run_image = image
        if self._input_h is not None and self._input_w is not None:
            if h != self._input_h or w != self._input_w:
                run_image = cv2.resize(image, (self._input_w, self._input_h), interpolation=cv2.INTER_AREA)

        if self._bgr_model:
            # UltraSharp-style models are BGR in [0,1] based on probing;
            # use a fixed convention instead of guessing at runtime.
            inp = self._preprocess_bgr_01(run_image)
            out = self.session.run(None, {self.input_name: inp})[0]
            up = self._postprocess_bgr_01(out)
        else:
            inp = self._preprocess_rgb(run_image)
            out = self.session.run(None, {self.input_name: inp})[0]
            up = self._postprocess_rgb(out)

        # Always resize back to the original patch size; we only want to
        # enhance the face region, not change its size in the frame.
        if up.shape[0] != h or up.shape[1] != w:
            up = cv2.resize(up, (w, h), interpolation=cv2.INTER_CUBIC)

        return up
