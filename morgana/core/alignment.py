from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np

STANDARD_5PTS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _arcface_template(image_size: int) -> np.ndarray:
    """
    Mirror InsightFace's face_align.estimate_norm behavior for ArcFace-style
    alignment.

    For 112x112, use the canonical template directly. For sizes that are
    multiples of 128 (e.g., 128x128 INSwapper input), use a horizontally
    shifted template as in InsightFace:

      - if image_size % 112 == 0:
            ratio = image_size / 112.0
            diff_x = 0
      - else (assume multiple of 128):
            ratio = image_size / 128.0
            diff_x = 8.0 * ratio

      dst = STANDARD_5PTS_112 * ratio
      dst[:, 0] += diff_x
    """
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    elif image_size % 128 == 0:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    else:
        # Fallback: scale the canonical template without horizontal shift.
        ratio = float(image_size) / 112.0
        diff_x = 0.0

    dst = STANDARD_5PTS_112 * ratio
    dst[:, 0] += diff_x
    return dst


def estimate_affine(landmarks: np.ndarray, image_size: int) -> np.ndarray:
    """Compute affine transform that warps the face onto the ArcFace template."""
    assert landmarks.shape == (5, 2), "Expected 5 landmarks"
    dst = _arcface_template(image_size)
    M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst, method=cv2.LMEDS)
    if M is None:
        raise ValueError("Could not compute affine transform for landmarks")
    return M.astype(np.float32)


def warp_face(image: np.ndarray, M: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.warpAffine(image, M, (image_size, image_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def invert_affine(M: np.ndarray) -> np.ndarray:
    full = np.vstack([M, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
    inv = np.linalg.inv(full)[0:2]
    return inv.astype(np.float32)


def paste_back(
    target_image: np.ndarray, swapped_face: np.ndarray, inv_M: np.ndarray, mask_blur: int = 11
) -> np.ndarray:
    """Warp swapped face back and blend with the original frame.

    Uses a soft elliptical mask to reduce visible box edges and
    BORDER_REPLICATE to avoid dark borders from padding.
    """
    h, w = target_image.shape[:2]
    # Warp swapped face back with replicated borders to avoid black edges.
    face_area = cv2.warpAffine(
        swapped_face, inv_M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    # Build a soft elliptical mask in crop space so we don't paste a hard box.
    ch, cw = swapped_face.shape[:2]
    yy, xx = np.ogrid[:ch, :cw]
    cy, cx = (ch - 1) / 2.0, (cw - 1) / 2.0
    ry, rx = ch * 0.45, cw * 0.45
    norm = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    mask_local = np.clip(1.0 - norm, 0.0, 1.0).astype(np.float32)

    mask = cv2.warpAffine(
        mask_local, inv_M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    if mask_blur > 1:
        ksize = mask_blur + (mask_blur + 1) % 2
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    mask = mask[..., None]
    output = face_area * mask + target_image.astype(np.float32) * (1.0 - mask)
    return np.clip(output, 0, 255).astype(np.uint8)


def expand_bbox(bbox: np.ndarray, scale: float, image_shape: Tuple[int, int]) -> np.ndarray:
    """Expand bbox by scale factor while keeping it inside image bounds."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    new_x1 = max(0, cx - w * 0.5)
    new_y1 = max(0, cy - h * 0.5)
    new_x2 = min(image_shape[1] - 1, cx + w * 0.5)
    new_y2 = min(image_shape[0] - 1, cy + h * 0.5)
    return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)
