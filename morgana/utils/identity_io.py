from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from morgana.core.types import resolve_path


Identity = Dict[str, np.ndarray]


def save_identity(identity: Identity, path: str | Path, mode: str | None = None) -> None:
    """
    Save an identity vector or pose-aware identity dict to a .npz file.

    Single-vector identities are stored under the key "avg".
    Dict identities store each bucket ("avg", "front", "left", "right", ...)
    as a separate array. A "_type" metadata field distinguishes formats.
    """
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta_mode = np.array(mode) if mode is not None else np.array("")
    arrays = {k: v for k, v in identity.items()}
    arrays["_type"] = np.array("dict")
    arrays["_mode"] = meta_mode
    np.savez(p, **arrays)


def load_identity(path: str | Path) -> tuple[Identity, str | None]:
    """
    Load an identity previously saved with save_identity.
    Returns either a single np.ndarray or a dict[str, np.ndarray].
    """
    p = resolve_path(path)
    with np.load(p, allow_pickle=False) as npz:
        files = list(npz.files)
        if "_type" not in files:
            raise ValueError(f"Identity file '{p}' is missing required '_type' metadata")
        itype = str(npz["_type"])
        mode = str(npz["_mode"]) if "_mode" in files and str(npz["_mode"]) else None
        # Skip metadata keys when building the identity payload.
        keys = [k for k in files if not k.startswith("_")]
        if not keys:
            raise ValueError(f"Identity file '{p}' contains no identity arrays")
        if itype == "vector":
            # Legacy single-vector identities: normalize into dict form.
            if "avg" not in keys:
                raise ValueError(f"Identity file '{p}' missing 'avg' array for vector identity")
            return {"avg": npz["avg"]}, mode
        if itype == "dict":
            identity_dict: Identity = {k: npz[k] for k in keys}
            return identity_dict, mode
        raise ValueError(f"Identity file '{p}' has unsupported _type='{itype}'")
