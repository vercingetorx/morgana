# Morgana Architecture & Developer Guide

This document is meant as a map of the repo and a reference for anyone who wants to understand or extend the utility (e.g. adding new backends or swappers).

---

## High‑Level Overview

The project implements a small, modular face‑swapping stack:

- **Detection**: find faces and 5‑point landmarks.
- **Identity**: build a 512‑D identity embedding from one or more source images.
- **Swapping**: align each target face, run a swapper ONNX model, and blend results back.
- **Restoration**: optionally run a face restorer on the swapped region and blend.

There are two main backends:

- **InsightFace backend**: wraps `insightface.app.FaceAnalysis` and the official INSwapper.
- **Core backend**: uses only the ONNX models in this repo (buffalo_l + ArcFace + ONNX swappers).

Both backends share:

- The **identity building** logic (`average` / `pose` modes).
- A common swappers interface (`swap(target_img, face, identity)`).
- The ability to **build, save, and reuse identities** via `.npz` files.

---

## Directory Map

At a glance:

- `faceswap/`
  - `cli/`
    - `main.py` – CLI entrypoint, argument parsing, backend selection, identity file handling.
  - `backends/`
    - `insightface_backend.py` – InsightFace + INSwapper/Hyperswap/ReSwapper wrapper.
    - `core_backend.py` – Builder for the core `FaceSwapPipeline` backend (core backend).
  - `core/`
    - `pipeline.py` – `FaceSwapPipeline`: detector + embedder + swapper + optional restorer.
    - `alignment.py` – ArcFace/INSwapper alignment, FFHQ‑style paste‑back helpers.
    - `types.py` – `Face` dataclass (bbox, 5‑pt landmarks, optional 106‑pt landmarks).
  - `detection/`
    - `buffalo.py` – ONNX SCRFD detector (`det_10g.onnx`), produces `Face` objects.
    - `landmark106.py` – 2d106 landmark refiner (buffalo_l’s `2d106det.onnx`).
  - `embedding/`
    - `arcface.py` – ONNX ArcFace embedder (buffalo_l `w600k_r50.onnx`).
  - `swappers/`
    - `inswapper.py` – INSwapper ONNX wrapper (ONNXRuntime + ArcFace‑style alignment).
    - `hyperswap.py` – Hyperswap ONNX wrapper (256×256 FFHQ template + learned mask).
    - `reswapper.py` – ReSwapper ONNX wrapper (128/256, uses ArcFace‑style alignment).
  - `restoration/`
    - `ultra_sharp.py` – Generic ONNX restorer/upscaler (`UltraSharpRestorer`).
  - `utils/`
    - `image.py` – image I/O and helpers.
    - `video.py` – video I/O, frame streaming, audio muxing.
    - `sam_mask.py` – SAM‑based mask builder for restoration blending.
    - `identity_build.py` – backend‑agnostic identity building utilities.
    - `identity_io.py` – load/save identity files (`.npz`).
    - `model_loader.py` – ONNXRuntime session creation, ZIP extraction.

At the repo root:

- `README.md` – user‑facing usage and examples.
- `ARCHITECTURE.md` – this document.
- `models/` – shipped models (buffalo_l, inswapper, hyperswap, reswapper, restorers).

---

## Execution Flow

### CLI entrypoint

All user flows go through `faceswap/cli/main.py`:

1. **Parse arguments** – `parse_args()` defines:
   - I/O: `--source`, `--target`, `--output`.
   - Backend / compute: `--backend`, `--swapper`, `--device`.
   - Identity: `--identity-file`, `--identity-mode`, `--save-identity`, `--build-identity`, `--merge-identities`.
   - Identity gating: `--gate-identity-file`, `--gate-identity`, `--gate-threshold`.
   - Face selection: `--max-faces`, `--face-index`.
   - Restoration & blending: `--restorer`, `--restorer-visibility`, `--mask-backend`.
   - Detection extras: `--refine-landmarks`.
2. **Expand sources** – `_expand_sources()` flattens mixed file/dir `--source` into a list of image paths.
3. **Resolve swapper model path** – maps `--swapper` to a specific ONNX (`inswapper_128`, `hyperswap_1a_256`, `reswapper_128/256`).
4. **Identity handling**:
   - If `--merge-identities` is set:
     - Require at least two `--identity-file` arguments.
     - Load all identities via `load_identity`, blend them equally with `blend_identity_dicts`, and save via `save_identity` to `--output`, then exit (no swapping).
   - Else, identity can come from:
     - A single `--identity-file` (mode stored in the file overrides `--identity-mode`), or
     - Source images, built according to `--identity-mode` (see below).
5. **Build identity (optional) from source images**:
   - With `--save-identity`: compute identity once (core or InsightFace backend) from `--source` and save via `save_identity`.
   - With `--build-identity`: compute identity from `--source`, save to `--output` or `identity.npz`, then exit (no swapping).
6. **Dispatch to backend for swapping**:
   - `--backend insightface` → call `swap_*_insightface`.
   - `--backend core` → construct `FaceSwapPipeline` via `build_minimal_pipeline` and call its methods.

The CLI supports:

- Image target (`--target some.jpg`).
- Video target (`--target some.mp4`/`.mkv`/…).
- Directory target (`--target some_dir/`), iterating over images.

### Core backend flow

`faceswap/backends/core_backend.py` defines:

```python
build_minimal_pipeline(...) -> FaceSwapPipeline
```

It wires:

- `BuffaloLDetector` (with optional `Landmark106Refiner`).
- `ArcFaceEmbedder`.
- `BuffaloAnalyzer` – a thin wrapper that:
  - runs `BuffaloLDetector.detect(image)` to get faces with bbox/landmarks/score,
  - runs `ArcFaceEmbedder.embed(image, face)` on each detected face,
  - writes the normalized 512‑D embedding into `Face.embedding`.
- One of `InSwapperONNX`, `HyperswapONNX`, or `ReSwapperONNX`.
- Optional `UltraSharpRestorer`.

`FaceSwapPipeline` (`faceswap/core/pipeline.py`) is analyzer‑centric:

- **Identity building**:
  - `compute_identity(source_image)`:
    - Calls `analyzer.analyze(source_image)` to get faces with embeddings.
    - Chooses a representative face via `_pick_face_for_identity`.
    - Returns a dict identity `{"avg": embedding}`.
  - `compute_identity_from_paths(source_paths)`:
    - Runs the analyzer on each source image, collects embeddings.
    - Uses `average_identity` to compute a single vector.
    - Returns `{"avg": avg}`.
  - `compute_pose_identities_from_paths(source_paths)`:
    - Same as above, plus yaw bucketing.
    - Returns a dict with `avg/front/left/right` vectors.
- **Swapping**:
  - `swap_image_array(target_image, identity_dict)`:
    - Uses `analyzer.analyze(target_image)` to get faces with bbox/landmarks/embedding.
    - If gate‑identity is **disabled**:
      - Applies `sort_and_select_faces(faces, face_index, max_faces)` to choose faces:
        - Faces are sorted by bbox area (largest → smallest).
        - `face_index` chooses a single face by index; `max_faces` caps how many from the head of the list.
    - If gate‑identity is **enabled**:
      - Skips geometric selection; all detected faces are candidates.
      - Normalizes `gate_identity["avg"]` once for the call.
      - For each face, uses its `Face.embedding` and cosine similarity vs the gate vector:
        - Faces with `sim < 0.25` are skipped and never swapped/restored.
    - For each face that passes selection/gating:
      - Chooses the swap identity vector from `identity_dict`:
        - Starts from `identity_dict["avg"]`.
        - In pose mode, uses yaw to pick `left/right/front` when present.
      - Calls `swapper.swap(output, face, face_identity)` (INSwapper/Hyperswap/ReSwapper).
      - Optionally runs restoration only on the swapped face region and blends with the chosen mask backend.
  - `swap_image_file(...)` / `swap_video_file(...)`:
    - Load frames.
    - Build or accept a precomputed identity dict (from `--identity-file` or `--source`).
    - Call `swap_image_array` per frame and manage image/video I/O.

### InsightFace backend flow

`faceswap/backends/insightface_backend.py` does the same pipeline using `insightface`:

1. `_make_app(device)` – builds a `FaceAnalysis` instance (`buffalo_l` multipack, SCRFD + ArcFace).
2. `build_insightface_identities(source_paths, device, identity_mode, refine_landmarks)`:
   - Wraps `FaceAnalysis` in an `InsightFaceAnalyzer` that produces `CoreFace` objects.
   - Uses `FaceSwapPipeline.compute_*_identities_from_paths` for average/pose identities, sharing logic with the core backend.
3. `swap_image_file_insightface(..., identity=None, face_index=None, max_faces=None, gate_identity=None, gate_enabled=False)`:
   - Chooses swapper:
     - `--swapper inswapper` → INSwapper ONNX weights wrapped by the shared `InSwapperONNX` (same wrapper used by the core backend).
     - `--swapper hyperswap-*` → your `HyperswapONNX` wrapper.
     - `--swapper reswapper-*` → your `ReSwapperONNX` wrapper.
   - Uses identity:
     - If `identity` is provided (from file), use it directly as a dict identity (`{"avg": ..., ...}`).
     - Else build via `build_insightface_identities`, which returns a dict in the same format as the core backend.
   - Face selection and gating:
     - Calls `app.get(tgt)` to get InsightFace `Face` objects (bbox + landmarks + embeddings).
     - Builds an initial list of candidates: faces with a valid `bbox`.
     - If `gate_enabled` is **False**:
       - Applies `sort_and_select_faces` with `face_index` / `max_faces` to choose faces to swap.
     - If `gate_enabled` is **True**:
       - Skips geometric selection; all detected faces are candidates.
       - Normalizes `gate_identity["avg"]` once.
       - For each face, pulls its embedding (`normed_embedding` or `embedding`), normalizes it, and computes cosine similarity to the gate vector.
       - Faces with `sim < 0.25` are skipped and never swapped/restored.
   - For each target face that passes selection/gating:
     - Selects `id_vec` from the identity dict:
       - Starts from `identities["avg"]`.
       - In pose mode, uses yaw and available buckets (`left/right/front`) to refine `id_vec`.
     - Converts the InsightFace face to your `CoreFace` (`_insight_face_to_core`) and calls `swapper.swap(result, core_face, id_vec)`.
   - Optional restoration works the same as in the core backend and is applied only to faces that were swapped.
4. `swap_video_file_insightface(..., identity=None, face_index=None, max_faces=None, gate_identity=None, gate_enabled=False)` mirrors the image flow over frames:
   - Builds or accepts an identity dict.
   - Streams frames and calls `app.get(frame)` for each.
   - Applies `sort_and_select_faces` and/or gate‑identity filtering per frame in exactly the same way as the image path.
   - Swaps only the faces that pass gating.
   - Runs restoration on those swapped faces and muxes audio back into the output video.

---

## Identity Building & Files

### Identity building (shared)

Backend‑agnostic utilities live in `faceswap/utils/identity_build.py`:

- `average_identity(embeddings)` – two‑pass mean with cosine outlier rejection.
- `pose_aware_identities(embeddings, yaw_scores)` – buckets by yaw (`left/right/front`) plus `avg`.

Both backends use these helpers; only the source of embeddings differs (ONNX ArcFace vs InsightFace ArcFace).

There are two layers to identity building:

- **Per‑image embeddings** – the analyzer (core or InsightFace) detects a face and runs an embedder to produce a 512‑D, unit‑norm embedding.
- **Per‑person identities** – multiple embeddings for the same person are combined into a single identity dict.

Per‑person identity modes:

- `identity-mode=average`
  - Take all per‑image embeddings for a single person.
  - Compute a mean embedding (with simple outlier rejection when there are >3 samples).
  - Normalize once; result is stored as `{"avg": vec}`.
  - With a single source image, this is effectively just that image’s embedding.

- `identity-mode=pose`
  - For each embedding, estimate a coarse yaw from 5‑point landmarks.
  - Bucket embeddings into `front`, `left`, `right` based on yaw.
  - Compute a normalized average per bucket and a global `avg`.
  - Stored as a dict with at least `{"avg": ...}` and optionally `front/left/right`.
  - At swap time, per‑face yaw selects the closest bucket when available, with `avg` as fallback.

`average` is the safest default for most users (especially when you have few or mostly frontal images). `pose` is best when you have many images per person with good pose coverage and want better fidelity under strong head turns.

### Identity file format

Saved via `faceswap/utils/identity_io.py`:

- `save_identity(identity, path, mode=None)`:
  - `identity`: dict (`{"avg": ..., "front": ..., "left": ..., "right": ...}`).
  - `mode`: `"average"` or `"pose"`.
  - Stored in `.npz` with:
    - `_type` = `"dict"` (older `_type="vector"` files are normalized on load).
    - `_mode` = `"average" | "pose" | ""`.
    - Identity arrays (`avg`, `front`, `left`, `right`, …).
- `load_identity(path)`:
  - Returns `(identity_dict, mode)`:
    - For legacy vector files, the loader exposes them uniformly as `{"avg": vector}`.
    - For dict files, all non‑metadata keys become entries in the identity dict.

CLI rules:

CLI rules:

- Building per‑person identities from images:
  - `--identity-mode` selects `average` vs `pose` when you use `--save-identity` or `--build-identity` without `--identity-file`.
- Using existing identities:
  - For swapping, exactly one `--identity-file` is allowed; its stored `_mode` is authoritative and `--identity-mode` is ignored.
  - For merging, `--merge-identities` requires at least two `--identity-file` arguments:
    - All identities are loaded via `load_identity` and blended equally with `blend_identity_dicts`:
      - Each key (`avg`, `front`, `left`, `right`, …) is averaged across all inputs that contain it, then re‑normalized.
      - The merged file’s `_mode` is the common stored mode if all inputs agree, otherwise `"average"`.
    - `--identity-mode` does **not** influence the merge; it only matters when building per‑person identities from images.

---

## Adding a New Swapper

To add a new swapper model:

1. **Implement a wrapper** in `faceswap/swappers/your_swapper.py`:

   ```python
   class YourSwapperONNX:
       def __init__(self, model_path: Path, providers: Optional[list[str]] = None):
           # load ONNX, infer IO names/shapes, set self.image_size, etc.

       def swap(self, target_img: np.ndarray, face: Face, identity: np.ndarray) -> np.ndarray:
           # 1) align using face.landmarks and your template
           # 2) preprocess crop + identity
           # 3) run session
           # 4) postprocess and paste back (using paste_back or your own masking)
   ```

   Requirements:

   - `face` is your `faceswap.core.types.Face` (bbox + 5‑point landmarks).
   - `identity` is a normalized 512‑D embedding (same as ArcFace output).
   - Return the full target image with the face swapped.

2. **Wire it into the core backend**:

  - Import your wrapper in `faceswap/backends/core_backend.py`.
   - Extend `build_minimal_pipeline`:

     ```python
     if swapper_backend == "your_swapper":
         swapper = YourSwapperONNX(swapper_path)
     ```

   - Add a model resolver in `faceswap/config.py` if you want a default path.

3. **Wire it into the CLI**:

   - Add a choice in `--swapper` (in `faceswap/cli/main.py`).
   - Map that choice to `base_backend` and a default `swapper_path` if appropriate.

4. **(Optional) Wire it into the InsightFace backend**:

   - Import your wrapper in `faceswap/backends/insightface_backend.py`.
   - In `swap_image_file_insightface` / `swap_video_file_insightface`, treat it like Hyperswap/ReSwapper:
     - Convert InsightFace `Face` → `CoreFace` via `_insight_face_to_core`.
     - Call `YourSwapperONNX.swap(result, core_face, id_vec)`.

As long as your wrapper implements `swap(target_img, face, identity)`, both backends can use it with the existing plumbing.

---

## Adding a New Backend

If you want a new backend (e.g. a different detector+embedder stack):

1. Implement a builder under `faceswap/backends/`, mirroring `core_backend`:

   ```python
   def build_my_backend_pipeline(... ) -> FaceSwapPipeline:
       detector = MyDetector(...)
       embedder = MyEmbedder(...)
       swapper = ExistingSwapperONNX(...)
       restorer = UltraSharpRestorer(...) or None
       return FaceSwapPipeline(...)
   ```

2. Update the CLI:

   - Add `"my_backend"` to `--backend` choices.
   - In `main()`, branch on `args.backend == "my_backend"` and:
     - Build your pipeline via `build_my_backend_pipeline`.
     - Follow the same image/video/dir logic as the core backend.

3. Keep identity building consistent:

   - Use `average_identity` / `pose_aware_identities` for combining embeddings.
   - Ensure your embedder returns normalized vectors comparable to ArcFace (or adjust swappers accordingly).

This keeps all backends aligned on identity semantics and swapper interfaces, while letting you experiment with different detectors/embedders or full external stacks (like InsightFace). 
