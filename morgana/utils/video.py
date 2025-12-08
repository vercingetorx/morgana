from __future__ import annotations

from pathlib import Path
from typing import Generator, Tuple
import subprocess

import cv2
import numpy as np

from morgana.core.types import resolve_path


def stream_video(path: str | Path) -> Tuple[Generator[np.ndarray, None, None], dict]:
    p = resolve_path(path)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {p}")

    props = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

    def generator() -> Generator[np.ndarray, None, None]:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()

    return generator(), props


class FFmpegVideoWriter:
    """
    Simple ffmpeg-based video writer that accepts BGR uint8 frames and encodes
    them with libx264 by default. This replaces OpenCV's VideoWriter so we have
    proper control over the codec (h264/HEVC/VP9 etc. in the future).
    """

    def __init__(self, path: str | Path, fps: float, width: int, height: int) -> None:
        self.path = resolve_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        # Default to H.264 in an appropriate container. We can later expose
        # codec/CRF via CLI, but this already gives a real h264 stream instead
        # of OpenCV's opaque mp4v.
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx265",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(self.path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if self._proc.stdin is None:
            self._proc.kill()
            raise RuntimeError("Failed to create ffmpeg video writer (no stdin)")

    def write(self, frame: np.ndarray) -> None:
        if self._proc.stdin is None:
            return
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        h, w = frame.shape[:2]
        if h != self.height or w != self.width:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self._proc.stdin.write(frame.tobytes())

    def release(self) -> None:
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        self._proc.wait()
        self._proc = None  # type: ignore[assignment]


def create_writer(path: str | Path, fps: float, width: int, height: int) -> FFmpegVideoWriter:
    """
    Factory for a video writer. Currently returns an ffmpeg-based writer that
    encodes H.264 video, which then gets audio muxed in a separate step.
    """
    return FFmpegVideoWriter(path, fps=fps, width=width, height=height)


def mux_audio_from_source(
    source_video: str | Path, silent_video: str | Path, output_video: str | Path
) -> None:
    """
    Copy audio track from source_video into silent_video using ffmpeg.

    - If ffmpeg is not available or fails, falls back to copying the
      silent_video to output_video (video without audio).
    """
    src = resolve_path(source_video)
    silent = resolve_path(silent_video)
    out = resolve_path(output_video)
    if src == out:
        # Avoid overwriting the source in-place.
        raise ValueError("output_video must be different from source_video")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(silent),
        "-i",
        str(src),
        "-c",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-shortest",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # On success, remove the intermediate silent video; only the final
        # muxed output should remain.
        if silent != out:
            silent.unlink(missing_ok=True)
    except Exception:
        # Fallback: keep the silent video if muxing fails.
        if silent != out:
            out.unlink(missing_ok=True)
            silent.replace(out)
