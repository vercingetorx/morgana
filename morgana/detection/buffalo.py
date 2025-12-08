from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from morgana.core.types import Face, resolve_path
from morgana.detection.landmark106 import Landmark106Refiner
from morgana.utils.model_loader import build_session, ensure_unzipped


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """
    Decode SCRFD distance predictions to bounding boxes.
    points: (N, 2) anchor centers [x, y]
    distance: (N, 4) distances to left, top, right, bottom in pixels.
    Returns (N, 4) [x1, y1, x2, y2].
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """
    Decode SCRFD distance predictions to keypoints.
    points: (N, 2) anchor centers [x, y]
    distance: (N, 10) distances for 5 (x, y) keypoints.
    Returns (N, 10) [x1,y1,x2,y2,...].
    """
    preds: List[np.ndarray] = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """
    Standard NMS over (N,4) boxes and (N,) scores.
    Returns indices of boxes to keep.
    """
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        # Compute IoU safely to avoid 0/0 warnings.
        iou = np.zeros_like(inter)
        valid = union > 0
        iou[valid] = inter[valid] / union[valid]

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


class BuffaloLDetector:
    """
    Pure ONNX face detector using buffalo_l's det_10g model (SCRFD-style).

    Returns bounding boxes + 5-point landmarks compatible with ArcFace/Hyperswap.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        conf: float = 0.4,
        nms_iou: float = 0.4,
        max_faces: int = 300,
        landmark_refiner: Optional[Landmark106Refiner] = None,
        providers: Optional[Sequence[str]] = None,
    ):
        self.model_path = self._resolve_model_path(resolve_path(model_path))
        self.device = device
        self.conf = conf
        self.nms_iou = nms_iou
        self.max_faces = max_faces
        self.landmark_refiner = landmark_refiner
        # When True, run the optional 106-point landmark refiner and derive 5-pt landmarks
        # from it for alignment. This adds extra compute per face.
        self.refine_landmarks: bool = landmark_refiner is not None
        self.providers = list(providers) if providers is not None else self._providers_for_device(device)

        self.session = build_session(self.model_path, providers=list(self.providers))
        self.input_name = self.session.get_inputs()[0].name
        ishape = self.session.get_inputs()[0].shape
        h = ishape[2] if isinstance(ishape[2], (int, np.integer)) and ishape[2] is not None else 640
        w = ishape[3] if isinstance(ishape[3], (int, np.integer)) and ishape[3] is not None else 640
        self.input_size: Tuple[int, int] = (int(w), int(h))  # (width, height)
        # SCRFD-specific configuration for det_10g: 3 feature levels, 2 anchors per location, with keypoints.
        self._feat_stride_fpn: List[int] = [8, 16, 32]
        self._num_anchors: int = 2
        self.use_kps: bool = True
        self._center_cache: dict[Tuple[int, int, int], np.ndarray] = {}

    @staticmethod
    def _providers_for_device(device: str) -> List[str]:
        dev = (device or "").lower()
        if dev.startswith("cuda") or dev.startswith("gpu"):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @staticmethod
    def _resolve_model_path(path: Path) -> Path:
        """
        Accept either det_10g.onnx directly or buffalo_l.zip and extract the detector.
        """
        if path.suffix == ".zip":
            extracted = ensure_unzipped(path, path.parent, members=None)
            # The buffalo_l bundle contains det_10g.onnx alongside ArcFace, etc.
            candidate = path.parent / "det_10g.onnx"
            if not candidate.exists():
                # Fallback: search extracted members.
                for p in extracted:
                    if p.name == "det_10g.onnx":
                        candidate = p
                        break
            if not candidate.exists():
                raise FileNotFoundError(f"det_10g.onnx not found in buffalo_l bundle: {path}")
            return candidate
        if not path.exists():
            raise FileNotFoundError(f"Detector model not found at {path}")
        if path.suffix.lower() != ".onnx":
            raise ValueError(f"Detector model must be an ONNX file or buffalo_l.zip; got {path}")
        return path

    def _postprocess(
        self,
        outputs: Sequence[np.ndarray],
        det_scale: float,
        input_wh: Tuple[int, int],
    ) -> List[Face]:
        """
        Decode SCRFD-style outputs into Face objects with NMS applied.
        Expects outputs ordered as [cls_8, cls_16, cls_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32].
        """
        input_w, input_h = input_wh
        input_height = input_h
        input_width = input_w

        scores_list: List[np.ndarray] = []
        bboxes_list: List[np.ndarray] = []
        kpss_list: List[np.ndarray] = []

        fmc = len(self._feat_stride_fpn)
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outputs[idx]
            bbox_preds = outputs[idx + fmc] * float(stride)
            kps_preds = outputs[idx + fmc * 2] * float(stride) if self.use_kps and len(outputs) >= fmc * 3 else None

            # Flatten predictions.
            scores = scores.reshape(-1).astype(np.float32)
            bbox_preds = bbox_preds.reshape(-1, 4).astype(np.float32)
            if kps_preds is not None:
                kps_preds = kps_preds.reshape(-1, 10).astype(np.float32)

            height = int(input_height // stride)
            width = int(input_width // stride)
            if height <= 0 or width <= 0:
                continue
            key = (height, width, int(stride))
            if key in self._center_cache:
                anchor_centers = self._center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * float(stride)).reshape(-1, 2)
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape(-1, 2)
                if len(self._center_cache) < 100:
                    self._center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= self.conf)[0]
            if pos_inds.size == 0:
                continue

            bboxes = _distance2bbox(anchor_centers, bbox_preds)
            pos_bboxes = bboxes[pos_inds]
            pos_scores = scores[pos_inds]
            bboxes_list.append(pos_bboxes)
            scores_list.append(pos_scores)

            if self.use_kps and kps_preds is not None:
                kpss = _distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        if not scores_list:
            return []

        scores = np.concatenate(scores_list, axis=0).astype(np.float32)
        bboxes = np.concatenate(bboxes_list, axis=0).astype(np.float32)
        kpss = np.concatenate(kpss_list, axis=0).astype(np.float32) if (self.use_kps and kpss_list) else None

        # Map back to original image coordinates.
        bboxes /= float(det_scale)
        if kpss is not None:
            kpss /= float(det_scale)

        # Sort by score and apply optional top-K before NMS.
        order = scores.argsort()[::-1]
        if self.max_faces > 0 and order.size > self.max_faces:
            order = order[: self.max_faces]
        bboxes = bboxes[order]
        scores = scores[order]
        if kpss is not None:
            kpss = kpss[order]

        keep = _nms(bboxes, scores, self.nms_iou)

        faces: List[Face] = []
        for idx in keep:
            bbox = bboxes[idx]
            lm = None
            if kpss is not None:
                lm = kpss[idx]
            faces.append(Face(bbox=bbox.astype(np.float32), landmarks=lm, score=float(scores[idx])))
        return faces

    def detect(self, image: np.ndarray) -> List[Face]:
        # Mirror InsightFace SCRFD preprocessing: resize with aspect preservation and letterbox.
        orig_h, orig_w = image.shape[:2]
        input_w, input_h = self.input_size
        im_ratio = float(orig_h) / float(orig_w)
        model_ratio = float(input_h) / float(input_w)
        if im_ratio > model_ratio:
            new_h = input_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = input_w
            new_h = int(new_w * im_ratio)
        det_scale = float(new_h) / float(orig_h)
        resized_img = cv2.resize(image, (new_w, new_h))
        det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized_img

        # Build blob: (BGR->RGB handled via swapRB) with (x-127.5)/128 norm.
        blob = cv2.dnn.blobFromImage(
            det_img,
            scalefactor=1.0 / 128.0,
            size=(input_w, input_h),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
        outputs = self.session.run(None, {self.input_name: blob})
        faces = self._postprocess(outputs, det_scale, (input_w, input_h))

        # NOTE: We deliberately do not synthesize fake 5-pt landmarks here.
        # Faces with bbox but no landmarks will be surfaced as-is; downstream
        # components decide whether to skip or error on such faces.

        # Optional 106-pt landmark refinement: when enabled and a refiner is provided,
        # run the buffalo_l 2d106det model on face crops and attach both landmarks_106
        # and refined 5-pt landmarks derived from it.
        if self.refine_landmarks and self.landmark_refiner is not None:  # pragma: no cover
            self.landmark_refiner.refine_faces_in_image(image, faces)

        return faces

    @staticmethod
    def _fallback_landmarks(bbox: np.ndarray) -> np.ndarray:
        """
        Simple geometric fallback if no landmarks are available.
        """
        x1, y1, x2, y2 = bbox.astype(np.float32)
        w, h = x2 - x1, y2 - y1
        return np.array(
            [
                [x1 + 0.3 * w, y1 + 0.35 * h],
                [x1 + 0.7 * w, y1 + 0.35 * h],
                [x1 + 0.5 * w, y1 + 0.55 * h],
                [x1 + 0.35 * w, y1 + 0.75 * h],
                [x1 + 0.65 * w, y1 + 0.75 * h],
            ],
            dtype=np.float32,
        )
