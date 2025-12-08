from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from morgana import config
from morgana.utils.model_loader import build_session
from morgana.utils.model_download import ensure_model_file


class SAMMasker:
    """
    Thin wrapper around a SAM predictor that can produce a soft mask from a box
    prompt. The mask is returned as a float array in [0, 1] with the same HxW
    as the input image.
    """

    def __init__(self, encoder_session, decoder_session, img_size: int):
        self.encoder = encoder_session
        self.decoder = decoder_session
        self.img_size = int(img_size)

    def mask_from_box(self, image_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Compute a soft mask for the given bounding box using ONNX-exported SAM
        encoder/decoder pairs.

        The output mask has the same spatial size as the input image and values
        in [0, 1].
        """
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1 = max(0.0, min(x1, w - 1.0))
        y1 = max(0.0, min(y1, h - 1.0))
        x2 = max(x1 + 1.0, min(x2, w - 1.0))
        y2 = max(y1 + 1.0, min(y2, h - 1.0))
        box = np.array([x1, y1, x2, y2], dtype=np.float32)

        # Resize image to the encoder's expected spatial size.
        img_size = self.img_size
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        img = resized.astype(np.float32).transpose(2, 0, 1)[None, ...]  # (1,3,H,W) in [0,255]

        # Scale box into the resized image coordinate frame.
        scale_x = float(img_size) / float(w)
        scale_y = float(img_size) / float(h)
        box_enc = np.array(
            [
                box[0] * scale_x,
                box[1] * scale_y,
                box[2] * scale_x,
                box[3] * scale_y,
            ],
            dtype=np.float32,
        )[None, ...]  # (1,4)

        # Run encoder and decoder ONNX models.
        emb = self.encoder.run(["image_embeddings"], {"image": img})[0]
        low_res_masks, _ = self.decoder.run(
            ["low_res_masks", "iou_predictions"],
            {"image_embeddings": emb, "boxes": box_enc},
        )
        # SAM decoder outputs logits. Mirror SAM's own behavior by
        # treating them as logits and thresholding at 0.0 (equivalent
        # to sigmoid(logits) > 0.5) instead of clamping to [0, 1].
        mask_lr = low_res_masks[0, 0].astype(np.float32)  # (Hm, Wm), logits

        # Upsample low-res mask to the original image resolution.
        mask_sq = cv2.resize(mask_lr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask_sq, (w, h), interpolation=cv2.INTER_LINEAR)

        # Binary region mask in {0,1}; edges will be softened later by
        # the Gaussian blur in the pipeline.
        mask = (mask > 0.0).astype(np.float32)
        return mask


@lru_cache()
def build_sam_masker(device: str = "cpu", variant: Optional[str] = None) -> SAMMasker:
    """
    Build (and cache) a SAMMasker using the SAM checkpoints under models/sams.

    Variant selection (no implicit fallback):
    - variant == "vit_b": require sam_vit_b_01ec64.pth.
    - variant == "vit_l": require sam_vit_l_0b3195.pth.
    - any other value (including None) is an error.
    """
    models_dir = config.models_root() / "sams"
    vit_b = models_dir / "sam_vit_b_01ec64.pth"
    vit_l = models_dir / "sam_vit_l_0b3195.pth"

    if variant == "vit_b":
        encoder_path = models_dir / "sam_vit_b_01ec64_encoder.onnx"
        decoder_path = models_dir / "sam_vit_b_01ec64_decoder.onnx"
    elif variant == "vit_l":
        encoder_path = models_dir / "sam_vit_l_0b3195_encoder.onnx"
        decoder_path = models_dir / "sam_vit_l_0b3195_decoder.onnx"
    else:
        raise RuntimeError(
            f"SAM variant must be 'vit_b' or 'vit_l', got {variant!r}."
        )

    encoder_path = ensure_model_file(encoder_path, config.models_root())
    decoder_path = ensure_model_file(decoder_path, config.models_root())

    # Choose ONNX Runtime providers based on the requested device.
    providers = None
    dev = (device or "").lower()
    if dev.startswith("cuda"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    encoder_sess = build_session(encoder_path, providers=providers)
    decoder_sess = build_session(decoder_path, providers=providers)

    # Infer SAM image size from encoder input shape (N,3,H,W).
    in_shape = encoder_sess.get_inputs()[0].shape
    if len(in_shape) != 4 or not isinstance(in_shape[2], int):
        raise RuntimeError(f"Unexpected SAM encoder input shape: {in_shape!r}")
    img_size = int(in_shape[2])

    return SAMMasker(encoder_session=encoder_sess, decoder_session=decoder_sess, img_size=img_size)
