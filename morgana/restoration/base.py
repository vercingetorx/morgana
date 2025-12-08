from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseRestorer(ABC):
    """Abstract interface for optional face/frame restoration or upscaling."""

    @abstractmethod
    def restore(self, image: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        """Return a restored/upscaled version of the input image."""
        raise NotImplementedError

