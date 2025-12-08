from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from morgana.core.alignment import invert_affine
from morgana.core.types import Face, resolve_path
from morgana.utils.model_loader import build_session


class HyperswapONNX:
    """
    Wrapper around the Hyperswap ONNX models.

    Expected IO (from user-provided inspection of hyperswap_1a_256.onnx):
      inputs:
        - source: [1, 512]        (identity embedding)
        - target: [1, 3, 256, 256] (aligned target face)
      outputs:
        - output: [1, 3, 256, 256] (swapped face)
        - mask:   [1, 1, 256, 256] (blending mask)
    """

    def __init__(self, model_path: str | Path, providers: Optional[List[str]] = None):
        self.model_path = resolve_path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Hyperswap model not found at '{self.model_path}'")
        self.session = build_session(self.model_path, providers=providers)
        self.image_size = 256

        # Infer input/output names by shape rather than relying on hard-coded strings.
        # We support two families of models:
        #   - hyperswap_1[a|b|c]_256.onnx: source [1, 512], target [1, 3, 256, 256]
        #   - edtalk_256.onnx (unsupported here): source [1, 1, 80, 16]
        src_name = None
        tgt_name = None
        for inp in self.session.get_inputs():
            shape = list(inp.shape)
            if len(shape) == 2 and shape[-1] == 512:
                src_name = inp.name
            elif len(shape) == 4 and shape[-3] == 3:
                tgt_name = inp.name
            elif len(shape) == 4 and shape[0] == 1 and shape[1] == 1 and shape[2] == 80 and shape[3] == 16:
                # edtalk-style source tensor: not supported.
                raise RuntimeError(
                    "Unsupported Hyperswap/edtalk model layout detected "
                    "(source [1,1,80,16]). This project only supports "
                    "hyperswap_1a/1b/1c models where the source input is [1,512]."
                )

        out_name = None
        mask_name = None
        for out in self.session.get_outputs():
            shape = list(out.shape)
            if len(shape) == 4 and shape[-3] == 3:
                out_name = out.name
            elif len(shape) == 4 and shape[-3] == 1:
                mask_name = out.name

        if src_name is None or tgt_name is None or out_name is None or mask_name is None:
            raise RuntimeError("Could not infer Hyperswap ONNX IO layout")

        self.src_input = src_name
        self.tgt_input = tgt_name
        self.output_name = out_name
        self.mask_name = mask_name

        # FFHQ-style 256x256 alignment template used by HyperSwap, matched
        # to the implementation in ComfyUI ReActor's run_hyperswap.
        self.std_landmarks_256 = np.array(
            [
                [84.87, 105.94],   # left eye
                [171.13, 105.94],  # right eye
                [128.00, 146.66],  # nose tip
                [96.95, 188.64],   # left mouth corner
                [159.05, 188.64],  # right mouth corner
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _preprocess_target(aligned: np.ndarray) -> np.ndarray:
        """BGR uint8 -> NCHW float32 in [-1, 1]."""
        blob = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = (blob - 127.5) / 127.5
        blob = blob.transpose(2, 0, 1)[None, ...]
        return blob

    @staticmethod
    def _postprocess_output(out: np.ndarray) -> np.ndarray:
        """NCHW float -> BGR uint8."""
        img = out[0].transpose(1, 2, 0)
        img = (img * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    @staticmethod
    def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
        """NCHW mask -> 2D float mask in [0,1]."""
        m = mask[0, 0].astype(np.float32)
        max_val = float(m.max()) if m.size > 0 else 1.0
        if max_val > 1.5:
            m = m / max_val  # normalize if in [0,255] or other scale
        m = np.clip(m, 0.0, 1.0)
        return m

    def swap(self, target_img: np.ndarray, face: Face, identity: np.ndarray) -> np.ndarray:
        if face.landmarks is None:
            raise ValueError("Landmarks required for Hyperswap")

        # Align target face to 256x256 using the FFHQ-style template HyperSwap
        # was trained on. This matches ReActor's run_hyperswap alignment and
        # produces correctly scaled faces in the crop instead of tiny faces.
        src_lm = face.landmarks.astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(src_lm, self.std_landmarks_256, method=cv2.LMEDS)
        if M is None:
            raise ValueError("Could not compute affine transform for Hyperswap")
        M = M.astype(np.float32)

        aligned = cv2.warpAffine(
            target_img,
            M,
            (self.image_size, self.image_size),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )

        target_blob = self._preprocess_target(aligned)
        identity_vec = identity[None, :].astype(np.float32)

        out_arr, mask_arr = self.session.run(
            [self.output_name, self.mask_name],
            {self.src_input: identity_vec, self.tgt_input: target_blob},
        )

        swapped_face = self._postprocess_output(out_arr)
        face_mask = self._postprocess_mask(mask_arr)

        # Warp swapped face and mask back to full frame
        inv_M = invert_affine(M)
        h, w = target_img.shape[:2]
        swapped_full = cv2.warpAffine(
            swapped_face, inv_M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        mask_full = cv2.warpAffine(
            face_mask, inv_M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

        # Optional blur for softer edges
        ksize = 9
        if ksize > 1:
            k = ksize + (ksize + 1) % 2
            mask_full = cv2.GaussianBlur(mask_full, (k, k), 0)

        mask_full = mask_full[..., None]
        out = swapped_full.astype(np.float32) * mask_full + target_img.astype(np.float32) * (1.0 - mask_full)
        return np.clip(out, 0, 255).astype(np.uint8)
