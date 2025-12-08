from __future__ import annotations

from pathlib import Path
from typing import Optional

import urllib.error
import urllib.request


HF_REPO_BASE = "https://huggingface.co/xioren00/morgana-models/resolve/main"


def ensure_model_file(path: Path, local_root: Path, hf_prefix: str = "models") -> Path:
    """
    Ensure that `path` exists on disk, downloading it from the Hugging Face
    repo if needed.

    - `path` is the desired local file path (e.g. models/buffalo_l/det_10g.onnx).
    - `local_root` is the local models root (typically config.models_root()).
    - `hf_prefix` is the leading directory under the HF repo; for our layout
      this is always "models", so that files live at:

        xioren00/morgana-models/models/...

    The remote URL is constructed as:

        f"{HF_REPO_BASE}/{hf_prefix}/{rel.as_posix()}"

    where `rel = path.relative_to(local_root)`. If the file already exists
    locally, no network access is performed.

    If an accompanying `.onnx.data` file exists in the HF repo (for models
    using external data), it is also downloaded next to `path`.
    """
    if path.exists():
        return path

    try:
        rel = path.relative_to(local_root)
    except ValueError:
        # Path is outside the expected local_root; do not attempt a download
        # based on a guessed layout. Caller will handle the missing file.
        return path

    rel_str = rel.as_posix()
    if hf_prefix:
        remote_path = f"{hf_prefix}/{rel_str}"
    else:
        remote_path = rel_str

    url = f"{HF_REPO_BASE}/{remote_path}"
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f" Downloading model: {rel_str} from {url}")
        urllib.request.urlretrieve(url, path)
    except urllib.error.HTTPError as exc:  # pragma: no cover - network failure
        raise FileNotFoundError(f"Could not download model from {url}: {exc}") from exc

    # Best-effort fetch for external data sidecar (e.g. model.onnx.data).
    data_url = url + ".data"
    data_path = Path(str(path) + ".data")
    try:
        urllib.request.urlretrieve(data_url, data_path)
    except urllib.error.HTTPError as exc:  # pragma: no cover - optional file
        if exc.code != 404:
            raise

    return path
