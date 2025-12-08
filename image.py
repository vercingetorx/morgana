from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from morgana.core.types import resolve_path


def load_image(path: str | Path) -> np.ndarray:
    p = resolve_path(path)
    # Use a context manager so the file handle is closed immediately after
    # reading, even when processing many images.
    with Image.open(p) as img:
        img = img.convert("RGB")
        arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def save_image(path: str | Path, image: np.ndarray) -> None:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image.fromarray(img).save(p)


def is_video(path: str | Path) -> bool:
    ext = resolve_path(path).suffix.lower()
    return ext in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)
