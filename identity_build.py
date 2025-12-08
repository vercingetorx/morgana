from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def average_identity(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """
    Compute a normalized average identity vector with simple outlier rejection.

    We follow the same approach as the core backend: first compute a mean
    vector, use it as a center to filter out embeddings with low cosine
    similarity, then recompute the mean and normalize.
    """
    embs = [np.asarray(e, dtype=np.float32).ravel() for e in embeddings]
    if not embs:
        raise RuntimeError("No embeddings provided for average_identity")
    stacked = np.stack(embs, axis=0)
    mean = stacked.mean(axis=0)
    norm = np.linalg.norm(mean) + 1e-12
    center = mean / norm
    if stacked.shape[0] > 3:
        sims = stacked @ center / (np.linalg.norm(stacked, axis=1) + 1e-12)
        keep = sims >= 0.6
        if np.any(keep):
            stacked = stacked[keep]
    mean = stacked.mean(axis=0)
    norm = np.linalg.norm(mean) + 1e-12
    return mean / norm


def pose_aware_identities(
    embeddings: Sequence[np.ndarray],
    yaw_scores: Sequence[float],
    yaw_thresh: float = 0.35,
) -> Dict[str, np.ndarray]:
    """
    Build pose-aware identities: average embeddings per yaw bucket plus a global average.
    Buckets: 'front', 'left', 'right', and 'avg' as fallback.
    """
    if len(embeddings) != len(yaw_scores):
        raise ValueError("embeddings and yaw_scores must have the same length")
    embs = [np.asarray(e, dtype=np.float32).ravel() for e in embeddings]
    if not embs:
        raise RuntimeError("No embeddings provided for pose_aware_identities")

    buckets: Dict[str, List[np.ndarray]] = {"front": [], "left": [], "right": []}
    for emb, yaw in zip(embs, yaw_scores):
        if yaw < -yaw_thresh:
            buckets["left"].append(emb)
        elif yaw > yaw_thresh:
            buckets["right"].append(emb)
        else:
            buckets["front"].append(emb)

    def _mean_norm(embs_list: List[np.ndarray]) -> np.ndarray:
        m = np.mean(np.stack(embs_list, axis=0), axis=0)
        n = np.linalg.norm(m) + 1e-12
        return m / n

    identities: Dict[str, np.ndarray] = {}
    identities["avg"] = _mean_norm(embs)
    for key, lst in buckets.items():
        if lst:
            identities[key] = _mean_norm(lst)
    return identities


def blend_identity_dicts(
    identities: Sequence[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    Blend multiple person-level identity dicts (each with at least an 'avg'
    key) into a single synthetic identity. All input identities contribute
    equally; there are no per-person weights.

    Each dict is expected to contain 512-D unit-norm embeddings keyed by:
      - 'avg' (mandatory), and optionally
      - pose buckets such as 'front', 'left', 'right', etc.

    For each key present in at least one input dict, this function:
      - collects all vectors for that key,
      - computes their simple mean,
      - and re-normalizes to unit length.

    The result is a new identity dict usable with the existing pipelines and
    identity I/O helpers.
    """
    if not identities:
        raise ValueError("blend_identity_dicts requires at least one identity dict")

    # Collect all keys across identities, ignoring reserved metadata keys.
    all_keys: set[str] = set()
    for ident in identities:
        for k in ident.keys():
            if k.startswith("_"):
                continue
            all_keys.add(k)

    if "avg" not in all_keys:
        raise RuntimeError("Cannot blend identities: no 'avg' key found in any input identity dict")

    blended: Dict[str, np.ndarray] = {}
    for key in sorted(all_keys):
        vecs: List[np.ndarray] = []
        for ident in identities:
            if key in ident:
                v = np.asarray(ident[key], dtype=np.float32).ravel()
                vecs.append(v)
        if not vecs:
            continue
        stacked = np.stack(vecs, axis=0)
        mean = stacked.mean(axis=0)
        norm = float(np.linalg.norm(mean) + 1e-12)
        if norm <= 0.0:
            continue
        blended[key] = mean / norm

    if "avg" not in blended:
        raise RuntimeError("Blended identity is missing 'avg' after aggregation")
    return blended
