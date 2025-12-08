from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import onnx
from onnx import numpy_helper

from morgana.core.alignment import estimate_affine, invert_affine, warp_face
from morgana.core.types import Face, resolve_path
from morgana.utils.model_loader import build_session


class InSwapperONNX:
    """
    INSwapper wrapper that mirrors InsightFace's INSwapper behavior as closely as possible.
    """

    def __init__(self, model_path: str | Path, providers: Optional[List[str]] = None):
        self.model_path = resolve_path(model_path)

        # Load the ONNX model to extract the embedding mapping matrix (emap).
        model = onnx.load(str(self.model_path))
        graph = model.graph
        self.emap = numpy_helper.to_array(graph.initializer[-1]).astype(np.float32)

        # Input normalization parameters follow InsightFace: mean=0, std=255.
        self.input_mean = 0.0
        self.input_std = 255.0

        # Build the runtime session.
        self.session = build_session(self.model_path, providers=providers)
        inputs = self.session.get_inputs()
        if len(inputs) != 2:
            raise RuntimeError("Unexpected inswapper inputs")

        # Collect input/output names and infer spatial size from the image input.
        self.input_names: List[str] = [inp.name for inp in inputs]
        outputs = self.session.get_outputs()
        if len(outputs) != 1:
            raise RuntimeError("Unexpected inswapper outputs")
        self.output_names: List[str] = [outputs[0].name]

        input_shape = inputs[0].shape
        # input_shape: [N, C, H, W]; we infer (W, H) as input_size.
        if len(input_shape) != 4:
            raise RuntimeError(f"Unexpected inswapper input shape: {input_shape}")
        self.input_shape = input_shape
        self.image_size = int(input_shape[2])
        self.input_size = (int(input_shape[3]), int(input_shape[2]))

    def _forward(self, img_blob: np.ndarray, latent: np.ndarray) -> np.ndarray:
        """
        Run INSwapper forward pass on preprocessed image blob and latent embedding.
        img_blob: (1,3,H,W) float32
        latent: (1,D) float32
        """
        img_norm = (img_blob - self.input_mean) / self.input_std
        pred = self.session.run(self.output_names, {self.input_names[0]: img_norm, self.input_names[1]: latent})[0]
        return pred

    def _build_latent(self, identity: np.ndarray) -> np.ndarray:
        """
        Project ArcFace identity embedding into INSwapper latent space using emap,
        matching InsightFace behavior.
        """
        latent = identity.astype(np.float32).reshape(1, -1)
        latent = np.dot(latent, self.emap)
        norm = np.linalg.norm(latent) + 1e-12
        latent /= norm
        return latent

    def swap(self, target_img: np.ndarray, face: Face, identity: np.ndarray) -> np.ndarray:
        if face.landmarks is None:
            raise ValueError("Landmarks required for swapping")

        # Align target face to the INSwapper input size using ArcFace-style 5-pt landmarks.
        M = estimate_affine(face.landmarks, self.image_size)
        aimg = warp_face(target_img, M, self.image_size)

        # Preprocess aligned crop using the same OpenCV blob logic as InsightFace.
        blob = cv2.dnn.blobFromImage(
            aimg,
            scalefactor=1.0 / self.input_std,
            size=self.input_size,
            mean=(self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
            crop=False,
        )

        latent = self._build_latent(identity)
        pred = self.session.run(self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent})[0]

        # Convert output to BGR uint8 aligned crop.
        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255.0 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]

        # Paste back using InsightFace's mask construction logic.
        target = target_img
        fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
        fake_diff = np.abs(fake_diff).mean(axis=2)
        fake_diff[:2, :] = 0
        fake_diff[-2:, :] = 0
        fake_diff[:, :2] = 0
        fake_diff[:, -2:] = 0

        IM = invert_affine(M)
        h, w = target.shape[:2]
        img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)
        bgr_fake_full = cv2.warpAffine(bgr_fake, IM, (w, h), borderValue=0.0)
        img_white_full = cv2.warpAffine(img_white, IM, (w, h), borderValue=0.0)
        fake_diff_full = cv2.warpAffine(fake_diff, IM, (w, h), borderValue=0.0)

        img_white_full[img_white_full > 20] = 255
        fthresh = 10
        fake_diff_full[fake_diff_full < fthresh] = 0
        fake_diff_full[fake_diff_full >= fthresh] = 255

        img_mask = img_white_full
        mask_h_inds, mask_w_inds = np.where(img_mask == 255)
        if mask_h_inds.size == 0 or mask_w_inds.size == 0:
            # Fallback: simple hard paste if mask is degenerate.
            return bgr_fake_full

        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h * mask_w))

        k = max(mask_size // 10, 10)
        kernel = np.ones((k, k), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)

        kernel = np.ones((2, 2), np.uint8)
        fake_diff_full = cv2.dilate(fake_diff_full, kernel, iterations=1)

        k = max(mask_size // 20, 5)
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)

        k = 5
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        fake_diff_full = cv2.GaussianBlur(fake_diff_full, blur_size, 0)

        img_mask /= 255.0
        fake_diff_full /= 255.0

        img_mask = np.reshape(img_mask, (img_mask.shape[0], img_mask.shape[1], 1))
        fake_merged = img_mask * bgr_fake_full + (1.0 - img_mask) * target.astype(np.float32)
        fake_merged = np.clip(fake_merged, 0, 255).astype(np.uint8)
        return fake_merged
