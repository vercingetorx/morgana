from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import onnx
from onnx import numpy_helper

from morgana.core.alignment import estimate_affine, invert_affine, paste_back, warp_face
from morgana.core.types import Face, resolve_path
from morgana.utils.model_loader import build_session


class ReSwapperONNX:
    """
    Wrapper around ReSwapper ONNX models (128 or 256).

    IO (from provided shapes):
      inputs:
        - target: [1, 3, H, W] (aligned target face, H=W=128 or 256)
        - source: [1, 512]     (identity embedding)
      outputs:
        - output: [1, 3, H, W] (swapped face)
    """

    def __init__(self, model_path: str | Path, providers: Optional[List[str]] = None):
        self.model_path = resolve_path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ReSwapper model not found at '{self.model_path}'")
        self.session = build_session(self.model_path, providers=providers)

        inputs = self.session.get_inputs()
        if len(inputs) != 2:
            raise RuntimeError("Unexpected ReSwapper inputs")

        img_name = None
        id_name = None
        image_size = 128
        for inp in inputs:
            shape = list(inp.shape)
            if len(shape) == 4:
                img_name = inp.name
                if len(shape) >= 4 and isinstance(shape[2], int) and shape[2] > 0:
                    image_size = int(shape[2])
            elif len(shape) == 2 or (shape and shape[-1] == 512):
                id_name = inp.name
        if img_name is None or id_name is None:
            raise RuntimeError("Could not infer ReSwapper IO layout")

        self.image_input = img_name
        self.id_input = id_name
        self.output_name = self.session.get_outputs()[0].name
        self.image_size = image_size

        # Load the embedding mapping matrix from the ONNX graph, similar to
        # INSwapper's emap. ReSwapper models are expected to embed this
        # projection; if it is missing or malformed we fail fast instead of
        # silently feeding raw embeddings with different semantics.
        model = onnx.load(str(self.model_path))
        graph = model.graph
        if not graph.initializer:
            raise RuntimeError(
                "ReSwapper model does not contain any initializers; cannot extract embedding mapping matrix."
            )
        emap = numpy_helper.to_array(graph.initializer[-1]).astype(np.float32)
        if emap.ndim != 2 or 512 not in emap.shape:
            raise RuntimeError(
                f"ReSwapper embedding mapping must be a 2D matrix with a 512-dim axis; got shape {emap.shape!r}"
            )
        self.emap = emap

    @staticmethod
    def _preprocess(aligned: np.ndarray) -> np.ndarray:
        """
        Preprocess crop for ReSwapper: BGR uint8 -> NCHW float32 in [0, 1].
        ReSwapper is implemented in the InsightFace ecosystem, so we mirror
        INSwapper's normalization (input_mean=0, input_std=255).
        """
        blob = aligned.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None, ...]
        return blob

    @staticmethod
    def _postprocess(out: np.ndarray) -> np.ndarray:
        """NCHW float in [0,1] (BGR-like) -> BGR uint8."""
        out = np.clip(out, 0.0, 1.0)[0]  # CHW
        swapped = (out * 255.0).astype(np.uint8).transpose(1, 2, 0)  # HWC BGR
        return swapped

    def _map_identity(self, identity: np.ndarray) -> np.ndarray:
        """
        Project ArcFace embedding into ReSwapper latent using the emap matrix
        extracted from the ONNX graph. Mirrors INSwapper behavior.
        """
        emb = identity.astype(np.float32)
        emap = self.emap
        if emap.ndim == 2:
            if emap.shape[0] == emb.shape[0]:
                return emb @ emap
            if emap.shape[1] == emb.shape[0]:
                return emap.T @ emb
        return emb

    def swap(self, target_img: np.ndarray, face: Face, identity: np.ndarray) -> np.ndarray:
        if face.landmarks is None:
            raise ValueError("Landmarks required for ReSwapper")

        # Align target face to the model's native size (128 or 256) using the
        # same 5-point ArcFace-style landmarks as the other swappers.
        M = estimate_affine(face.landmarks, self.image_size)
        aligned = warp_face(target_img, M, self.image_size)

        target_blob = self._preprocess(aligned)
        id_vec = self._map_identity(identity)[None, :].astype(np.float32)

        out = self.session.run([self.output_name], {self.image_input: target_blob, self.id_input: id_vec})[0]
        swapped_face = self._postprocess(out)

        inv_M = invert_affine(M)
        return paste_back(target_img, swapped_face, inv_M, mask_blur=9)
