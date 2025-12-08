from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        desc = kwargs.get("desc")
        if desc:
            print(f"{desc}...", flush=True)
        return iterable

from morgana.core.alignment import expand_bbox
from morgana.core.types import Face, sort_and_select_faces
from morgana.utils import image as image_utils
from morgana.utils import video as video_utils
from morgana.utils.identity_build import average_identity, pose_aware_identities
from morgana.utils.sam_mask import build_sam_masker


def swap_faces_in_image(
    image: np.ndarray,
    faces: List[Face],
    *,
    swapper,
    identity: dict[str, np.ndarray],
    identity_mode: str,
    mask_backend: str,
    device: str,
    restorer=None,
    restorer_visibility: float = 1.0,
    face_index: int | None = None,
    max_faces: int | None = None,
    gate_identity=None,
    gate_enabled: bool = False,
    gate_threshold: float = 0.25,
) -> np.ndarray:
    """
    Shared core logic to apply identity-driven swaps to an image given a set
    of detected faces. This is used by the core pipeline and by backends
    that adapt external detectors into Face objects.
    """
    if not faces:
        return image
    output = image.copy()

    # Deterministic face selection, unless identity gating is enabled. When
    # gating is enabled, always consider all detected faces and let the gate
    # identity decide which ones to swap.
    if not gate_enabled:
        faces = sort_and_select_faces(faces, face_index=face_index, max_faces=max_faces)

    # Optional gating: build normalized reference vector once.
    gate_vec: Optional[np.ndarray] = None
    if gate_enabled:
        if gate_identity is None:
            raise RuntimeError("gate_enabled is True but no gate_identity was provided")
        raw = gate_identity
        gate_vec = raw.get("avg")
        if gate_vec is None:
            raise RuntimeError("Gate identity does not contain an 'avg' vector for gating")
        gate_vec = np.asarray(gate_vec, dtype=np.float32)
        norm = float(np.linalg.norm(gate_vec) + 1e-12)
        gate_vec = gate_vec / norm

    skipped_no_landmarks = 0
    for face in faces:
        if face.landmarks is None:
            skipped_no_landmarks += 1
            continue
        if gate_enabled:
            if face.embedding is None:
                raise RuntimeError("Gate enabled but face has no embedding attached")
            f = np.asarray(face.embedding, dtype=np.float32)
            fn = float(np.linalg.norm(f) + 1e-12)
            f = f / fn
            sim = float(np.dot(f, gate_vec))  # type: ignore[arg-type]
            if sim < gate_threshold:
                continue

        # Select identity for this face: single vector or pose-aware buckets.
        face_identity = identity.get("avg")
        if identity_mode == "pose":
            lm = face.landmarks
            left_eye, right_eye, nose = lm[0], lm[1], lm[2]
            mid_eye = (left_eye + right_eye) * 0.5
            eye_dx = np.linalg.norm(right_eye - left_eye) + 1e-6
            yaw_score = (nose[0] - mid_eye[0]) / eye_dx
            if yaw_score < -0.35 and "left" in identity:
                face_identity = identity["left"]
            elif yaw_score > 0.35 and "right" in identity:
                face_identity = identity["right"]
            elif "front" in identity:
                face_identity = identity["front"]

        output = swapper.swap(output, face, face_identity)

        if restorer is not None:
            # Restore only the swapped face region, not the whole frame.
            h, w = output.shape[:2]
            # Use a slightly expanded region for restoration blending to
            # capture context around the face, but keep SAM conditioning on
            # the original detector bbox (face.bbox). This mirrors the older
            # behavior where SAM was guided by the tight face box while the
            # restored patch could extend further.
            restore_bbox = expand_bbox(face.bbox, scale=1.3, image_shape=(h, w))
            x1, y1, x2, y2 = [int(v) for v in restore_bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            patch = output[y1:y2, x1:x2].copy()
            restored = restorer.restore(patch)
            if restored.shape[:2] != patch.shape[:2]:
                restored = cv2.resize(
                    restored, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_CUBIC
                )
            # Build blending mask according to the selected backend.
            ph, pw = patch.shape[:2]
            if mask_backend == "sam-vit-b":
                sam = build_sam_masker(device, "vit_b")
                full_mask = sam.mask_from_box(output, face.bbox)
                mask = full_mask[y1:y2, x1:x2].astype(np.float32)
            elif mask_backend == "sam-vit-l":
                sam = build_sam_masker(device, "vit_l")
                full_mask = sam.mask_from_box(output, face.bbox)
                mask = full_mask[y1:y2, x1:x2].astype(np.float32)
            elif mask_backend == "ellipse":
                mask = np.zeros((ph, pw), dtype=np.float32)
                center = (pw // 2, ph // 2)
                axes = (int(pw * 0.45), int(ph * 0.55))
                cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
            else:
                raise RuntimeError(f"Unknown mask-backend: {mask_backend!r}")

            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            mask = mask[..., None]
            v = np.clip(restorer_visibility, 0.0, 1.0)
            restored_vis = patch.astype(np.float32) * (1.0 - v) + restored.astype(np.float32) * v
            blended = restored_vis * mask + patch.astype(np.float32) * (1.0 - mask)
            output[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return output


class FaceSwapPipeline:
    """
    Pipeline ties together an analyzer, swapper, and optional restorer.

    For the core backend, the analyzer is the *only* entry point:
      analyzer.analyze(image) -> List[Face] with bbox/landmarks/score/embedding.

    All identity building and target swapping go through the analyzer; the
    older split of detector+embedder inside this pipeline is no longer used.
    """

    def __init__(
        self,
        swapper,
        restorer=None,
        restorer_visibility: float = 1.0,
        identity_mode: str = "average",
        mask_backend: str = "sam",
        device: str = "cpu",
        analyzer=None,
        face_index: int | None = None,
        max_faces: int | None = None,
        gate_identity=None,
        gate_enabled: bool = False,
        gate_threshold: float = 0.25,
    ):
        if analyzer is None:
            raise RuntimeError("FaceSwapPipeline requires an analyzer; detector/embedder ctor args have been removed.")
        self.swapper = swapper
        self.analyzer = analyzer
        self.restorer = restorer
        self.restorer_visibility = float(restorer_visibility)
        self.identity_mode = identity_mode
        self.mask_backend = mask_backend
        self.device = device
        self.face_index = face_index
        self.max_faces = max_faces
        self.gate_identity = gate_identity
        self.gate_enabled = bool(gate_enabled)
        self.gate_threshold = float(gate_threshold)

    @staticmethod
    def _pick_face(faces: List[Face]) -> Optional[Face]:
        if not faces:
            return None
        return max(faces, key=lambda f: f.score)

    @staticmethod
    def _pick_face_for_identity(faces: List[Face], image_shape: tuple[int, int]) -> Optional[Face]:
        """
        Pick a reasonably large, confident face for identity building.
        Very small or low-score detections are skipped instead of poisoning the embedding.
        """
        if not faces:
            return None
        face = max(faces, key=lambda f: f.score)
        h, w = image_shape
        x1, y1, x2, y2 = face.bbox
        area = max(0.0, (x2 - x1) * (y2 - y1))
        frac = area / float(max(1.0, w * h))
        if face.score < 0.4 or frac < 0.01:
            return None
        return face

    def compute_identity(self, source_image: np.ndarray) -> dict[str, np.ndarray]:
        if self.analyzer is None:
            raise RuntimeError("FaceSwapPipeline requires an analyzer; detector/embedder-only usage has been removed.")
        faces = self.analyzer.analyze(source_image)
        face = self._pick_face_for_identity(faces, source_image.shape[:2])
        if face is None:
            raise RuntimeError("No face detected in source image")
        if face.embedding is None:
            raise RuntimeError("Analyzer did not attach an embedding to the selected face")
        return face.embedding

    def compute_identity_from_paths(self, source_paths: List[str | Path]) -> dict[str, np.ndarray]:
        if self.analyzer is None:
            raise RuntimeError("FaceSwapPipeline requires an analyzer; detector/embedder-only usage has been removed.")
        embeddings: List[np.ndarray] = []
        iterator = source_paths
        if len(source_paths) > 1:
            iterator = tqdm(source_paths, desc="Building identity (core)", unit="img")
        for path in iterator:
            img = image_utils.load_image(path)
            faces = self.analyzer.analyze(img)
            face = self._pick_face_for_identity(faces, img.shape[:2])
            if face is None:
                continue
            if face.embedding is None:
                continue
            emb = face.embedding
            embeddings.append(emb)
        if not embeddings:
            raise RuntimeError("No faces detected in any source images")
        avg = average_identity(embeddings)
        return {"avg": avg}

    def compute_pose_identities_from_paths(self, source_paths: List[str | Path]) -> dict[str, np.ndarray]:
        """
        Build pose-aware identities: average embeddings per rough yaw bucket plus a global average.
        Buckets: 'front', 'left', 'right', and 'avg' as fallback.
        """
        yaw_scores: list[float] = []
        all_embs: list[np.ndarray] = []

        iterator = source_paths
        if len(source_paths) > 1:
            iterator = tqdm(source_paths, desc="Building identity (core, pose-aware)", unit="img")
        if self.analyzer is None:
            raise RuntimeError("FaceSwapPipeline requires an analyzer; detector/embedder-only usage has been removed.")

        for path in iterator:
            img = image_utils.load_image(path)
            faces = self.analyzer.analyze(img)
            face = self._pick_face_for_identity(faces, img.shape[:2])
            if face is None or face.landmarks is None:
                continue
            if face.embedding is None:
                continue
            emb = face.embedding
            all_embs.append(emb)

            # Rough yaw estimate from landmarks: nose offset from mid-eye center.
            lm = face.landmarks
            left_eye, right_eye, nose = lm[0], lm[1], lm[2]
            mid_eye = (left_eye + right_eye) * 0.5
            eye_dx = np.linalg.norm(right_eye - left_eye) + 1e-6
            yaw_score = (nose[0] - mid_eye[0]) / eye_dx
            yaw_scores.append(float(yaw_score))

        if not all_embs:
            raise RuntimeError("No faces detected in any source images for pose-aware identity")
        return pose_aware_identities(all_embs, yaw_scores)

    def swap_image_array(self, target_image: np.ndarray, identity: dict[str, np.ndarray]) -> np.ndarray:
        if self.analyzer is None:
            raise RuntimeError("FaceSwapPipeline requires an analyzer; detector/embedder-only usage has been removed.")
        faces = self.analyzer.analyze(target_image)
        if not faces:
            return target_image

        return swap_faces_in_image(
            target_image,
            faces,
            swapper=self.swapper,
            identity=identity,
            identity_mode=self.identity_mode,
            mask_backend=self.mask_backend,
            device=self.device,
            restorer=self.restorer,
            restorer_visibility=self.restorer_visibility,
            face_index=self.face_index,
            max_faces=self.max_faces,
            gate_identity=self.gate_identity,
            gate_enabled=self.gate_enabled,
            gate_threshold=self.gate_threshold,
        )

    def swap_image_file(
        self,
        source_paths: List[str | Path],
        target_path: str | Path,
        output_path: str | Path,
        identity: Optional[np.ndarray] = None,
    ) -> None:
        target_img = image_utils.load_image(target_path)
        if identity is None:
            if self.identity_mode == "pose":
                identity = self.compute_pose_identities_from_paths(source_paths)
            else:
                identity = self.compute_identity_from_paths(source_paths)
        swapped = self.swap_image_array(target_img, identity)
        image_utils.save_image(output_path, swapped)

    def swap_video_file(
        self,
        source_paths: List[str | Path],
        video_path: str | Path,
        output_path: str | Path,
        progress: bool = True,
        identity: Optional[np.ndarray] = None,
    ) -> None:
        if identity is None:
            if self.identity_mode == "pose":
                identity = self.compute_pose_identities_from_paths(source_paths)
            else:
                identity = self.compute_identity_from_paths(source_paths)

        frames, props = video_utils.stream_video(video_path)
        # Write to a temporary video first (no audio), then mux audio from the source.
        output_path = Path(output_path)
        # Keep a real video extension for the temporary file so OpenCV
        # uses a video backend instead of the images backend.
        tmp_video = output_path.with_name(output_path.stem + ".noaudio" + output_path.suffix)
        writer = video_utils.create_writer(tmp_video, props["fps"], props["width"], props["height"])

        iterator: Iterable[np.ndarray] = frames
        if progress:
            iterator = tqdm(frames, total=props["frame_count"], desc="Swapping")
        try:
            for frame in iterator:
                swapped = self.swap_image_array(frame, identity)
                writer.write(swapped)
        finally:
            writer.release()
        # Attempt to carry over audio from the original video; if ffmpeg is missing,
        # this will fall back to the silent video.
        video_utils.mux_audio_from_source(video_path, tmp_video, output_path)
