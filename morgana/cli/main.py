from __future__ import annotations

import argparse
from pathlib import Path

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        desc = kwargs.get("desc")
        if desc:
            print(f"{desc}...", flush=True)
        return iterable

from morgana import config
from morgana.backends import BACKENDS, BackendProtocol
from morgana.utils.identity_io import load_identity, save_identity
from morgana.utils.image import is_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal modular face-swapper for images and videos.")

    # I/O: what to read and where to write.
    parser.add_argument(
        "--source",
        nargs="+",
        help="One or more source face images or directories used for identity (multiple improves robustness).",
    )
    parser.add_argument(
        "--target",
        help="Target image, video, or directory of images to process.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Path to save swapped output, or identity file when using --build-identity (use a .npz extension for identities). "
            "For directory targets, if omitted, the target directory is used as the output directory."
        ),
    )

    # Backend / compute: detector+embedder stack and swapper.
    parser.add_argument(
        "--backend",
        choices=["core", "insightface"],
        default="core",
        help=(
            "Overall pipeline backend. 'core' (default) uses buffalo_l detector + ArcFace + ONNX; "
            "'insightface' uses InsightFace FaceAnalysis+INSwapper and requires the insightface package."
        ),
    )
    parser.add_argument(
        "--swapper",
        choices=["inswapper", "hyperswap-1a", "hyperswap-1b", "hyperswap-1c", "reswapper-128", "reswapper-256"],
        default="hyperswap-1a",
        help=(
            "Which swapper implementation/variant to use. "
            "inswapper = INSwapper ONNX; "
            "hyperswap-1a/1b/1c = Hyperswap ONNX variants; "
            "reswapper-128/256 = ReSwapper ONNX/InsightFace variants."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="ONNX Runtime / InsightFace device to use (cpu/cuda).",
    )

    # Identity: how to build or load identities from source faces.
    parser.add_argument(
        "--identity-file",
        action="append",
        help=(
            "Precomputed identity file (.npz) to use instead of source images. "
            "May be specified multiple times; when multiple files are given, their identities "
            "can be blended with equal weight into a single synthetic identity when using --merge-identities."
        ),
    )
    parser.add_argument(
        "--identity-mode",
        choices=["average", "pose"],
        default="average",
        help="How to build the source identity from multiple images: average over all embeddings, or pose-aware buckets.",
    )
    parser.add_argument(
        "--save-identity",
        help="Optional path to save the computed identity (.npz) for reuse.",
    )
    parser.add_argument(
        "--build-identity",
        action="store_true",
        help="Build and save an identity from --source without performing any swap.",
    )
    parser.add_argument(
        "--merge-identities",
        action="store_true",
        help=(
            "Blend all provided --identity-file entries into a single identity .npz and exit "
            "(no swapping). Each input identity contributes equally."
        ),
    )

    # Identity gating: restrict which target faces are allowed to be swapped.
    parser.add_argument(
        "--gate-identity-file",
        help=(
            "Optional identity file (.npz) used to decide which target faces are "
            "allowed to be swapped. Faces whose embedding does not match this "
            "identity (above a fixed cosine similarity threshold) are left untouched."
        ),
    )
    parser.add_argument(
        "--gate-identity",
        action="store_true",
        help=(
            "Enable identity-based gating using --gate-identity-file. When set, only "
            "target faces matching the gate identity are swapped; other faces in the "
            "frame are ignored."
        ),
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=0.25,
        help=(
            "Cosine similarity threshold used when --gate-identity is enabled. "
            "Faces with similarity to the gate identity below this value are not swapped."
        ),
    )

    # Face selection: which faces in each frame to actually swap.
    parser.add_argument(
        "--max-faces",
        type=int,
        default=None,
        help=(
            "Maximum number of faces to swap per image/frame. "
            "Faces are ranked by bounding-box area (largest first). "
            "Default: no limit."
        ),
    )
    parser.add_argument(
        "--face-index",
        type=int,
        default=None,
        help=(
            "Swap only a single face chosen by index after sorting faces by "
            "bounding-box area (0 = largest face). If set, --max-faces is "
            "ignored. If the index is out of range for a frame, no faces "
            "are swapped in that frame. Default: not set (no forced single-face selection)."
        ),
    )

    # Restoration & blending.
    parser.add_argument(
        "--restorer",
        choices=["none", "gpen-512", "gpen-1024", "gpen-2048", "gfpgan-14", "restoreformer-pp", "codeformer", "ultra-sharp-x4"],
        default="none",
        help="Face restoration model from models/facerestore (or the generic ultra_sharp_x4 upscaler) to use before optional upscaling.",
    )
    parser.add_argument(
        "--restorer-visibility",
        type=float,
        default=0.25,
        help="Blend between original swapped face and restored face inside the face region (0.0 = no restoration, 1.0 = full restoration).",
    )
    parser.add_argument(
        "--mask-backend",
        choices=["sam-vit-b", "sam-vit-l", "ellipse"],
        default="sam-vit-b",
        help=(
            "Mask backend for blending swapped/restored faces. "
            "'sam-vit-b' / 'sam-vit-l' use Segment Anything ViT-B / ViT-L checkpoints; "
            "'ellipse' uses a simple elliptical mask."
        ),
    )

    # Detection extras.
    parser.add_argument(
        "--refine-landmarks",
        action="store_true",
        help=(
            "Use the optional 106-point buffalo_l 2d106det landmark refiner to improve 5-point landmarks "
            "for alignment (requires 2d106det.onnx; adds extra ONNX compute per face)."
        ),
    )
    return parser.parse_args()


def _expand_sources(sources) -> list[Path]:
    """Expand a mix of image paths and directories into a flat list of image files."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files: list[Path] = []
    for item in sources:
        p = Path(item)
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file() and child.suffix.lower() in exts:
                    files.append(child)
        elif p.is_file():
            files.append(p)
    if not files:
        raise SystemExit("No source images found in given --source arguments")
    return files


def main() -> None:
    args = parse_args()

    try:
        # Expand sources (if any).
        source_paths: list[Path] = []
        if args.source:
            source_paths = _expand_sources(args.source)

        # If we are only merging identities, we do not require a target or source.
        # All work is done on the provided identity files and we exit after saving.
        if args.merge_identities:
            if not args.identity_file or len(args.identity_file) < 2:
                raise SystemExit("--merge-identities requires at least two --identity-file arguments")
            if not args.output:
                raise SystemExit("--merge-identities requires --output pointing to the blended identity .npz")

            from morgana.utils.identity_build import blend_identity_dicts

            id_dicts = []
            modes: list[str] = []
            for path_str in args.identity_file:
                ident, stored_mode = load_identity(path_str)
                id_dicts.append(ident)
                if stored_mode:
                    modes.append(stored_mode)

            blended = blend_identity_dicts(id_dicts)
            if modes and all(m == modes[0] for m in modes):
                merged_mode = modes[0]
            else:
                merged_mode = "average"

            out_path = Path(args.output)
            save_identity(blended, out_path, mode=merged_mode)
            print(f" Merged {len(id_dicts)} identities into: {out_path}")
            return

        # For swapping/build-identity flows, ensure we have either images or an identity file.
        if not source_paths and not args.identity_file:
            raise SystemExit("At least one of --source or --identity-file must be provided")

        # Decode swapper kind into a base type and variant (if any).
        raw_backend = args.swapper
        base_backend = "inswapper"
        variant: str | None = None
        if raw_backend.startswith("hyperswap"):
            base_backend = "hyperswap"
            if "-" in raw_backend:
                _, variant = raw_backend.split("-", 1)
            else:
                variant = "1a"
        elif raw_backend.startswith("reswapper"):
            base_backend = "reswapper"
            if "-" in raw_backend:
                _, variant = raw_backend.split("-", 1)
            else:
                variant = "128"

        # Determine effective swapper model path based on --swapper choice.
        # When using Hyperswap/ReSwapper, map the short variant name to the
        # corresponding ONNX under models/.
        if base_backend == "hyperswap":
            swapper_path = config.hyperswap_model_path(variant or "1a")
        elif base_backend == "reswapper":
            swapper_path = config.reswapper_model_path(variant or "128")
        else:
            swapper_path = config.inswapper_model_path()

        # Backend and detector descriptions.
        if args.backend == "insightface":
            backend_desc = "InsightFace FaceAnalysis + INSwapper"
            detector_desc = "InsightFace FaceAnalysis (buffalo_l)"
        else:
            backend_desc = "Minimal (buffalo_l detector + ArcFace + ONNX)"
            detector_desc = "buffalo_l det_10g.onnx"

        # Restorer description (if any).
        if args.restorer != "none":
            restorer_path = config.restorer_model_path(args.restorer)
            restorer_desc = f"{args.restorer} ({Path(restorer_path).name})"
        else:
            restorer_desc = "none"

        # Load or build identity. When a single identity file is provided,
        # its stored mode (if any) is authoritative; CLI --identity-mode is ignored.
        identity = None
        if args.identity_file:
            if len(args.identity_file) > 1:
                raise SystemExit(
                    "Multiple --identity-file arguments are only supported together with --merge-identities. "
                    "For swapping, provide exactly one --identity-file."
                )
            identity, stored_mode = load_identity(args.identity_file[0])
            identity_mode = stored_mode if stored_mode else "average"
        else:
            identity_mode = args.identity_mode

        # Optional gate identity: separate identity file used only to decide
        # which target faces are eligible to be swapped.
        gate_identity = None
        if args.gate_identity:
            if not args.gate_identity_file:
                raise SystemExit("--gate-identity requires --gate-identity-file pointing to a .npz identity")
            gate_identity, _ = load_identity(args.gate_identity_file)

        # Instantiate backend implementation based on --backend.
        backend_cls = BACKENDS[args.backend]
        backend: BackendProtocol = backend_cls(
            device=args.device,
            swapper_backend=base_backend,
            swapper_path=swapper_path,
            restorer_name=args.restorer,
            identity_mode=identity_mode,
            restorer_visibility=args.restorer_visibility,
            mask_backend=args.mask_backend,
            face_index=args.face_index,
            max_faces=args.max_faces,
            gate_identity=gate_identity,
            gate_enabled=bool(args.gate_identity),
            gate_threshold=args.gate_threshold,
            refine_landmarks=args.refine_landmarks,
        )

        # Optionally build + save identity alongside a swap run.
        if args.save_identity and identity is None:
            if not source_paths:
                raise SystemExit("--save-identity requires --source when no --identity-file is provided")
            identity = backend.build_identity(source_paths)
            save_identity(identity, args.save_identity, mode=identity_mode)

        # Identity-only mode: build and save identity, then exit.
        if args.build_identity:
            if identity is None:
                if not source_paths:
                    raise SystemExit("--build-identity requires --source when no --identity-file is provided")
                identity = backend.build_identity(source_paths)
            out_path = Path(args.output) if args.output else Path("identity.npz")
            save_identity(identity, out_path, mode=identity_mode)
            print(f" Built {identity_mode} identity and saved to: {out_path}")
            return

        # From this point on we require a target for swapping.
        if not args.target:
            raise SystemExit("--target is required unless --build-identity or --merge-identities is set")

        target_path = Path(args.target)
        if target_path.is_dir():
            target_kind = "dir"
            target_label = target_path.name
        else:
            target_kind = "video" if is_video(args.target) else "image"
            target_label = target_path.name

        # For image/video targets, an explicit output path is required.
        if target_kind in ("image", "video") and not args.output:
            raise SystemExit("--output is required for image and video targets")

        # Normalize video output container for video targets.
        video_exts = {".mkv", ".mp4", ".mov", ".avi", ".webm"}
        video_output_path = None
        if target_kind == "video":
            video_output_path = Path(args.output)
            if video_output_path.suffix.lower() not in video_exts:
                video_output_path = video_output_path.with_suffix(".mkv")
                print(f" Output: {video_output_path.name} (forced .mkv for video)")

        # Swapper backend description.
        if args.backend == "insightface":
            swapper_backend_desc = f"insightface+{raw_backend}"
        else:
            swapper_backend_desc = raw_backend

        print("=== Morgana ===")
        print(f" Backend: {backend_desc}")
        print(f" Swapper backend: {swapper_backend_desc}")
        print(f" Swapper model: {Path(swapper_path).name}")
        print(f" Detector: {detector_desc}")
        print(f" Restorer: {restorer_desc}")
        print(f" Identity mode: {identity_mode}")
        print(f" Restorer visibility: {args.restorer_visibility}")
        print(f" Mask backend: {args.mask_backend}")
        print(f" Target ({target_kind}): {target_label}")
        print(f" Source images: {len(source_paths)}")

        # Backend-agnostic swapping using the selected backend implementation.
        if target_path.is_dir():
            exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            out_dir = Path(args.output) if args.output else target_path
            out_dir.mkdir(parents=True, exist_ok=True)
            targets = sorted(
                p for p in target_path.iterdir() if p.is_file() and p.suffix.lower() in exts
            )
            print(f" Stage: swapping images ({args.backend} backend)...")
            for t in tqdm(targets, desc="Swapping images", unit="img"):
                # Avoid overwriting inputs when output directory equals target
                # directory by always appending a "-swapped" suffix.
                out_name = f"{t.stem}-swapped{t.suffix}" if t.suffix else f"{t.stem}-swapped"
                out_path = out_dir / out_name
                backend.swap_image(source_paths, t, out_path, identity=identity)
        elif is_video(args.target):
            print(f" Stage: swapping video ({args.backend} backend)...")
            if video_output_path is None:
                raise RuntimeError("video_output_path is None for video target")
            backend.swap_video(source_paths, args.target, video_output_path, identity=identity)
        else:
            print(f" Stage: swapping image ({args.backend} backend)...")
            backend.swap_image(source_paths, args.target, Path(args.output), identity=identity)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
