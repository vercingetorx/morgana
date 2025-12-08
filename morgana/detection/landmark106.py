from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import onnx

from morgana.utils.model_loader import build_session
from morgana.utils.model_download import ensure_model_file


class Landmark106Refiner:
    """
    Lightweight 106-point landmark predictor using the InsightFace buffalo_l
    2d106det.onnx model. Input is a face crop; output is (106, 2) landmarks
    in crop coordinates.
    """

    def __init__(self, model_path: Path):
        """
        Initialize the 106-point landmark refiner from an ONNX model.

        We mirror InsightFace's Landmark model:
          - infer input_size from the ONNX input tensor,
          - infer input_mean / input_std by inspecting the first few nodes.
        """
        # Allow the model to be auto-downloaded when missing.
        model_path = ensure_model_file(model_path, model_path.parent.parent)
        self.session = build_session(model_path)
        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        input_shape = input_cfg.shape
        # Expect NCHW input; infer square spatial size from H.
        self.input_size = int(input_shape[2]) if len(input_shape) >= 4 else 192

        # Inspect the ONNX graph to decide normalization, like InsightFace.
        model = onnx.load(str(model_path))
        graph = model.graph
        find_sub = False
        find_mul = False
        for nid, node in enumerate(graph.node[:8]):
            if node.name.startswith("Sub") or node.name.startswith("_minus"):
                find_sub = True
            if node.name.startswith("Mul") or node.name.startswith("_mul"):
                find_mul = True
            if nid < 3 and node.name == "bn_data":
                find_sub = True
                find_mul = True
        if find_sub and find_mul:
            # MXNet-style model: mean/std baked into graph.
            self.input_mean = 0.0
            self.input_std = 1.0
        else:
            # Standard InsightFace ONNX: (x - 127.5) / 128.
            self.input_mean = 127.5
            self.input_std = 128.0

    def _forward_batch(self, aligned_rgbs: Sequence[np.ndarray]) -> np.ndarray:
        """
        Run 106-point detection on a batch of aligned RGB crops of size
        (input_size, input_size). Returns (N, 106, 2) in aligned coordinates.
        """
        if not aligned_rgbs:
            return np.empty((0, 106, 2), dtype=np.float32)

        S = self.input_size
        blobs: List[np.ndarray] = []
        for rgb in aligned_rgbs:
            if rgb.shape[0] != S or rgb.shape[1] != S:
                rgb = cv2.resize(rgb, (S, S), interpolation=cv2.INTER_LINEAR)
            blob = cv2.dnn.blobFromImage(
                rgb,
                scalefactor=1.0 / float(self.input_std),
                size=(S, S),
                mean=(self.input_mean, self.input_mean, self.input_mean),
                swapRB=False,
            )
            blobs.append(blob[0])

        batch = np.stack(blobs, axis=0)
        out = self.session.run(None, {self.input_name: batch})[0]  # (N, 212)
        pts = out.reshape(out.shape[0], -1, 2).astype(np.float32)
        # Model outputs approx [-1, 1]; map to [0, S] aligned coordinates.
        pts[:, :, 0:2] += 1.0
        pts[:, :, 0:2] *= (S / 2.0)
        return pts

    @staticmethod
    def to_five_points(pts106: np.ndarray) -> np.ndarray:
        """
        Approximate 5-point landmarks (left eye, right eye, nose, left mouth,
        right mouth) from 106-point output.

        Instead of trusting a single index per feature (which can be noisy on
        heavy makeup/paint), we take a small spatial neighbourhood around each
        canonical index and use the centroid as the landmark.
        """
        # Indices based on common 106-pt ordering (may vary slightly by model).
        left_eye_idx = 38
        right_eye_idx = 88
        nose_idx = 52
        left_mouth_idx = 76
        right_mouth_idx = 82
        anchor_indices = np.array(
            [left_eye_idx, right_eye_idx, nose_idx, left_mouth_idx, right_mouth_idx],
            dtype=np.int32,
        )

        pts = np.asarray(pts106, dtype=np.float32)
        if pts.shape[0] < anchor_indices.max() + 1:
            # Fallback: not enough points; just index directly where possible.
            valid = anchor_indices[anchor_indices < pts.shape[0]]
            return pts[valid, :]
        # Use a small neighbourhood around each anchor: take the K nearest
        # neighbours (including the anchor itself) in Euclidean distance and
        # average them to reduce local noise.
        K = 5
        refined: List[np.ndarray] = []
        for idx in anchor_indices:
            anchor = pts[idx]
            d = np.linalg.norm(pts - anchor[None, :], axis=1)
            nn_idx = np.argsort(d)[:K]
            refined.append(pts[nn_idx].mean(axis=0))
        return np.stack(refined, axis=0).astype(np.float32)

    def refine_faces_in_image(self, image: np.ndarray, faces: Sequence[object]) -> None:
        """
        Refine landmarks for a collection of faces in a single image.

        For each face with a valid bbox, we:
          - crop the bbox,
          - run 106-point prediction on the crop,
          - map 106-pt landmarks back to image coordinates,
          - derive 5-pt landmarks and attach both to the face object.

        Faces are expected to expose:
          - bbox: np.ndarray [x1, y1, x2, y2]
          - landmarks (optional): will be overwritten with 5-pt landmarks
          - landmarks_106 (optional): will be set to (106, 2) landmarks
        """
        if not faces:
            return

        ih, iw = image.shape[:2]
        aligned_rgbs: List[np.ndarray] = []
        Ms: List[np.ndarray] = []
        indices: List[int] = []

        for idx, face in enumerate(faces):
            bbox = getattr(face, "bbox", None)
            if bbox is None:
                continue
            bb = np.asarray(bbox, dtype=np.float32)
            x1, y1, x2, y2 = bb.astype(np.float32)
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            S = float(self.input_size)
            scale = S / (max(w, h) * 1.5)
            # Similarity transform matching InsightFace face_align.transform with rotation=0.
            cos_r, sin_r = 1.0, 0.0
            M = np.array(
                [[scale * cos_r, -scale * sin_r, 0.0], [scale * sin_r, scale * cos_r, 0.0]],
                dtype=np.float32,
            )
            M[0, 2] = S * 0.5 - cx * scale
            M[1, 2] = S * 0.5 - cy * scale

            aligned = cv2.warpAffine(image, M, (self.input_size, self.input_size), borderValue=0.0)
            if aligned.size == 0:
                continue
            indices.append(idx)
            Ms.append(M)
            aligned_rgbs.append(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))

        if not aligned_rgbs:
            return

        pts106_aligned = self._forward_batch(aligned_rgbs)  # (N, 106, 2) in aligned coords

        for j, face_idx in enumerate(indices):
            face = faces[face_idx]
            M = Ms[j]
            IM = cv2.invertAffineTransform(M)
            pts = pts106_aligned[j]
            ones = np.ones((pts.shape[0], 1), dtype=np.float32)
            pts_h = np.concatenate([pts, ones], axis=1)  # (106,3)
            pts_img = (IM @ pts_h.T).T.astype(np.float32)  # (106,2)

            # Attach refined 106-point landmarks.
            setattr(face, "landmarks_106", pts_img)
            # Also derive refined 5-point landmarks so that, when the refiner
            # is enabled, downstream alignment can opt into using them instead
            # of the detector's raw 5-pt output.
            five = self.to_five_points(pts_img)
            setattr(face, "landmarks", five)
