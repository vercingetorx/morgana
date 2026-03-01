# Morgana

Lightweight, modular face swapping pipeline that runs on standard PyTorch/ONNX tooling. No ComfyUI, no heavyweight wrappers — just detection, identity extraction, swapping, and optional restoration hooks.

## Layout

- `morgana/cli`: CLI entrypoint (`python -m morgana.cli.main`).
- `morgana/detection`: buffalo_l det_10g ONNX detector (bbox + 5 keypoints, optional 106-pt refiner).
- `morgana/embedding`: ArcFace embedder for identity vectors.
- `morgana/swappers`: InsightFace InSwapper/Hyperswap/ReSwapper ONNX wrappers.
- `models/`: Drop your detector/arcface/swapper/restorer weights here (already populated).

## Quickstart

```bash
pip install -r requirements.txt

# swap image -> image (multiple sources or a directory give more robust identity)
# Default backend: core (buffalo_l detector + ArcFace + ONNX swappers)
python -m morgana.cli.main \
  --source ref_faces/ \
  --target target_frame.jpg \
  --output out.jpg \

# swap image -> video
python -m morgana.cli.main \
  --source ref_faces/ \
  --target target_video.mkv \
  --output out.mkv \

# choose a specific restoration model (from models/facerestore)
python -m morgana.cli.main \
  --source ref_faces/ \
  --target target_video.mkv \
  --output out_gfpgan14_x2.mkv \
  --restorer gfpgan-14 \
```

Defaults assume the bundled models:
- Detector: `models/buffalo_l/det_10g.onnx`
- ArcFace: `models/buffalo_l/w600k_r50.onnx`
- Swapper (default): `models/insightface/inswapper_128.onnx`
- Alternative swapper: any Hyperswap ONNX, e.g. `models/hyperswap/hyperswap_1a_256.onnx` (use `--swapper hyperswap`).

Set `FACESWAP_MODELS_DIR` if you store weights elsewhere.

## Concepts

- **Identity images (`--source`)**  
  One or more images (or directories) of the person whose face you want to transfer. Multiple images give a more stable identity.

- **Target (`--target`)**  
  Image, video, or directory of images you want to process.

- **Backends (`--backend`)**
  - `core` (default): uses pure ONNX components in this repo (buffalo_l detector + ArcFace + ONNX swapper).
  - `insightface`: optional backend using InsightFace `FaceAnalysis` + INSwapper. Requires `insightface` to be installed separately (`pip install insightface`).

- **Swappers (`--swapper`)**
  - `inswapper`: InsightFace INSwapper ONNX.
  - `hyperswap-1a/1b/1c`: Hyperswap variants (identity + target crop + learned mask).
  - `reswapper-128/256`: ReSwapper ONNX models (128×128 or 256×256).

  When you pick `--swapper inswapper` you are always using:

  - The official InsightFace INSwapper ONNX weights and architecture.
  - Driven through the shared `InSwapperONNX` wrapper in this repo.
  - With faces/landmarks/embeddings coming either from:
    - InsightFace `FaceAnalysis` (when `--backend insightface`), or
    - The local ONNX stack (when `--backend core`).

  This means INSwapper’s behavior and alignment are unified across both backends; swapping quality differences between backends should now only come from how faces are detected/embedded, not from different INSwapper glue.

- **Identity modes (`--identity-mode`)**
  - `average`: build a single averaged, normalized 512‑D identity vector.
  - `pose`: build pose‑aware identities (`avg`, `front`, `left`, `right`) using yaw; each target face picks the closest pose bucket.

- **Restoration (`--restorer`)**  
  Optional face restoration/upscaling using ONNX models under `models/facerestore` (GFPGAN, GPEN, RestoreFormer, CodeFormer, ultra_sharp_x4). Applied only to the swapped face region, *after* the swap has been done.

## Core pipeline

High‑level steps (both backends, INSwapper/Hyperswap/ReSwapper):

1. **Detect faces on sources**, pick a good face per image.
2. **Embed with ArcFace** to get normalized 512‑D identity vectors.
3. **Build identity** from all source embeddings using `average` or `pose` mode.
4. **Detect faces on target** (image, video, or each image in a directory).
5. **Align each target face** to the swapper’s expected crop size:
   - INSwapper / ReSwapper: ArcFace‑style 5‑point template.
   - Hyperswap: FFHQ‑style 5‑point template.
6. **Run the swapper** with the identity + aligned crop.
7. **Blend back to target frame** using the swapper’s mask or an elliptical mask.
8. **Optionally run restoration** on the face region and blend with `--restorer-visibility`.

### What the main knobs actually do

- `--backend`
  - `insightface`: use InsightFace’s own detector/embeddings (`FaceAnalysis`) to drive the swappers.
  - `core`: use the local ONNX detector (`BuffaloLDetector`) and ArcFace embedder to drive the same swappers.
  - In both cases, INSwapper, Hyperswap, and ReSwapper are called through the same ONNX wrappers; the backend mainly changes how faces/landmarks/embeddings are obtained.

- `--swapper`
  - `inswapper`: use the InsightFace INSwapper ONNX model for swapping, via the shared `InSwapperONNX` wrapper.
  - `hyperswap-1a/1b/1c`: use Hyperswap ONNX models, which generate both a swapped face and a learned mask that decides where to blend.
  - `reswapper-128/256`: use ReSwapper ONNX models at 128×128 or 256×256, using the same ArcFace‑style alignment as INSwapper.
  - Choice here affects **how the fake face is generated and how the swapper’s own mask behaves** (e.g. how it handles occlusions like hair, hands, glasses).

## Backends in practice

### InsightFace backend (`--backend insightface`)

- Uses `insightface.app.FaceAnalysis` for detection + embeddings.
- Swapper choice:
  - `--swapper inswapper` → InsightFace INSwapper.
  - `--swapper hyperswap-*` → `HyperswapONNX` wrapper.
  - `--swapper reswapper-*` → `ReSwapperONNX` wrapper.
- Supports `--identity-mode average|pose` and uses the same identity‑building logic as the core backend.

### Core backend (`--backend core`)

- Uses:
  - `BuffaloLDetector` (SCRFD `det_10g.onnx` + optional 106‑pt refiner).
  - `ArcFaceEmbedder` (`w600k_r50.onnx`).
  - ONNX swappers: INSwapper, Hyperswap, or ReSwapper.
- Everything is driven through `FaceSwapPipeline` and mirrors InsightFace behavior as closely as possible.

Example (core backend + Hyperswap + restorer):

```bash
python -m morgana.cli.main \
  --backend core \
  --source ref_faces/ \
  --target target_video.mkv \
  --output out_hyperswap_gpen512.mkv \
  --swapper hyperswap-1a \
  --restorer gpen-512
```

## Identity files (build once, reuse everywhere)

Identities are stored as `.npz` files via `numpy.savez` with metadata:

- `_type`: `"dict"` (older `"vector"` files are normalized on load).
- `_mode`: `"average"` or `"pose"`.
- Data arrays (always a dict in memory):  
  - At minimum: `avg` (the canonical identity vector).  
  - For pose‑aware identities: `avg`, `front`, `left`, `right`, ...

There are two layers to identity building:

- Per‑image: the backend’s analyzer detects a face and runs an embedder to get a 512‑D, unit‑norm identity vector.
- Per‑person: multiple embeddings for the same person are combined into a single identity dict stored in the `.npz`.

`--identity-mode` controls only this per‑person combination step when building from images:

- `average`:
  - All per‑image embeddings for one person are averaged (with simple outlier rejection for larger sets) and re‑normalized.
  - The result is stored as `{"avg": vec}`.
  - With a single source image, this is effectively just that image’s embedding.
- `pose`:
  - Each embedding is assigned a coarse yaw (left / right / front) from its 5‑point landmarks.
  - Separate averages are computed per yaw bucket plus a global `avg`.
  - Stored as `{"avg": ..., "front": ..., "left": ..., "right": ...}` where buckets exist.
  - At swap time, each face’s yaw selects the closest bucket when present, falling back to `avg`.

In practice:

- `average` is the safest default for most users (few or mostly frontal images, simpler and more consistent across frames).
- `pose` is best when you have many images per person with good pose coverage and want better fidelity under strong head turns.

### Build and save an identity only

You can build an identity once and reuse it later:

```bash
# Build a pose-aware identity and save it
python -m morgana.cli.main \
  --backend core \
  --source ref_faces/ \
  --identity-mode pose \
  --build-identity \
  --output identities/alice_pose.npz

```

If `--output` is omitted with `--build-identity`, the identity is saved to `identity.npz` in the current directory.

### Reuse a saved identity (no source images)

```bash
python -m morgana.cli.main \
  --backend core \
  --identity-file identities/alice_pose.npz \
  --target target_frame.jpg \
  --output out_alice_on_target.jpg
```

Notes:
- When a single `--identity-file` is provided for swapping, the stored `_mode` in the identity file (`average` or `pose`) is used and `--identity-mode` is ignored.
- For pose identities, per‑face yaw in the target frame selects the best bucket (`left/right/front`), falling back to `avg`.

You can also combine swapping and identity saving:

```bash
python -m morgana.cli.main \
  --backend core \
  --source ref_faces/ \
  --target target_frame.jpg \
  --output out.jpg \
  --save-identity identities/alice_pose.npz \
  --identity-mode pose
```

Internally, both the core and InsightFace backends treat identities as a single, unified type:

- A dict `{"avg": ..., "front": ..., "left": ..., "right": ...}` where only `avg` is required.
- The same format is used for:
  - The **swap identity** (A): who you paste in.
  - The optional **gate identity** (C, see below): which person in the target is allowed to be touched.

Legacy vector‑only identities are normalized to `{"avg": vector}` when loaded.

### Merge multiple identities into a new one

You can also create a synthetic identity that blends several existing identities equally:

```bash
python -m morgana.cli.main \
  --identity-file identities/alice_pose.npz \
  --identity-file identities/bob_pose.npz \
  --merge-identities \
  --output identities/alice_bob_blend.npz
```

- All input `.npz` files are treated as separate people.
- For each key present in at least one file (`avg`, `front`, `left`, `right`, …), Morgana:
  - collects all vectors for that key,
  - averages them,
  - and re‑normalizes the result.
- The merged file’s `_mode` is:
  - the common stored mode if all inputs agree,
  - otherwise `"average"`.
- `--identity-mode` does **not** affect merging; it only matters when building per‑person identities from images.

## Targets: image, video, directory

- **Image** (`--target some.jpg`)  
  Writes a single output image to `--output`.

- **Video** (`--target some.mkv / .mp4 / ...`)  
  Writes a video to `--output` (extension normalized to `.mkv` if needed).  
  Audio is copied from the source via ffmpeg (through OpenCV wrappers).

- **Directory of images** (`--target some_dir/`)  
  Processes each image file in the directory independently and writes to an output directory:

  ```bash
  python -m morgana.cli.main \
    --source ref_faces/ \
    --target target_frames/ \
    --output swapped_frames/ \
    --backend core
  ```

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

## Notes
- Minimal dependencies: numpy, pillow, opencv-python-headless, onnxruntime, torch, tqdm.
- Restoration models (GFPGAN/GPEN/RestoreFormer/CodeFormer) are shipped; wire them into `morgana/restoration` if you want post-processing passes.
- ffmpeg is used only through OpenCV; feel free to swap in pure-ffmpeg frame readers/writers if you prefer. 
  - With a **restorer** (`--restorer != none`), we use `--mask-backend` **only for the restoration blend**, not for the core swap:
    - `sam-vit-b` / `sam-vit-l`: use Segment Anything (SAM) ViT‑B / ViT‑L checkpoints to produce a detailed mask for the face region, then blend restored vs swapped pixels inside that mask.
    - `ellipse`: use a simple elliptical mask around the face region as a fallback or when you do not want to load SAM.
  - This means `--mask-backend` changes *how the restored face is blended back into the already swapped frame*, but it does **not** control how INSwapper/Hyperswap/ReSwapper decide what to overwrite in the first place. Occlusion handling (e.g. hair partially covering the face) is primarily determined by the swapper model and its own masking, not by `--mask-backend`.

## Face selection and identity gating

### Geometric selection: `--face-index`, `--max-faces`

By default, both backends may detect multiple faces in an image or frame. You can control *which* faces are swapped using simple geometric rules:

- `--face-index N`
  - Faces in each frame are sorted by bounding‑box area (largest → smallest).
  - `--face-index` selects a single face by index in this sorted list:
    - `0` = largest face (default).
    - `1` = second largest, etc.
  - When `--face-index` is set, `--max-faces` is ignored.
  - If the index is out of range for a frame, no faces are swapped in that frame.

- `--max-faces K`
  - When `--face-index` is **not** set, `--max-faces` caps how many faces per frame are swapped:
    - Faces are sorted by area; only the first `K` are considered.
  - If `--max-faces` is omitted, all detected faces are candidates.

These rules are applied per frame for both images and videos, and are implemented identically in the core and InsightFace backends.

### Identity gating: only swap onto a specific target identity

In many videos you have multiple people in the frame (B, C, D, …) but only want to swap onto one of them (say C), regardless of who is largest or closest to the camera. Morgana supports this explicitly via a **separate gate identity**:

- **Swap identity A** (what to paste):
  - Built from `--source` or loaded from `--identity-file` as described above.
  - This is the face that will be pasted into the target.

- **Gate identity C** (who is allowed to be touched in the target):
  - Built separately and saved, e.g.:

    ```bash
    python -m morgana.cli.main \
      --backend core \
      --source ref_faces_target_person/ \
      --build-identity \
      --output identities/target_person.npz
    ```

  - Used at swap time via:

    ```bash
    --gate-identity-file identities/target_person.npz
    --gate-identity
    ```

When `--gate-identity` is enabled:

- For each image/frame, both backends:
  - Detect faces and attach a 512‑D embedding per face (core: ArcFace; InsightFace: FaceAnalysis).
  - Ignore `--face-index` and `--max-faces` entirely: **all** detected faces are candidates.
  - Compare each face’s embedding to the gate identity’s `avg` vector using cosine similarity.
  - Only faces with similarity ≥ `--gate-threshold` (default `0.25`) are swapped; all other faces in the frame are left untouched.
- The swap identity A is still used as before to decide what to paste; the gate identity C is only used to decide *which* target faces are eligible.

This gives you the behavior:

- “Paste A onto C, never onto B or D,”
- without relying on fragile heuristics like “largest face,” and in the same way for both images and videos, in both backends.
