from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

import onnxruntime as ort


def ensure_unzipped(zip_path: Path, output_dir: Path, members: Optional[Iterable[str]] = None) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = members or zf.namelist()
        for name in names:
            dest = output_dir / name
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, dest.open("wb") as dst:
                    dst.write(src.read())
            extracted.append(dest)
    return extracted


def build_session(model_path: Path, providers: Optional[List[str]] = None) -> ort.InferenceSession:
    """
    Create an ONNX Runtime session with reduced log noise.

    We mirror the effect of `export ORT_LOG_SEVERITY_LEVEL=3` by setting
    the session log severity to ERROR. This keeps benign warnings (like
    unused initializers) out of the console while still surfacing real issues.
    """
    available = providers or ort.get_available_providers()
    opts = ort.SessionOptions()
    # 0=VERBOSE,1=INFO,2=WARNING,3=ERROR,4=FATAL
    opts.log_severity_level = 3
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=available)
