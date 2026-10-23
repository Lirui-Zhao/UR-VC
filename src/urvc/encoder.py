"""SigLIP-2 video encoding used when a dataset has no cached embeddings."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _frames(video_path: Path) -> Iterator[np.ndarray]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Video encoding requires `pip install 'ur-vc[encoder]'`") from exc
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


@torch.inference_mode()
def encode_videos(
    video_paths: list[Path],
    *,
    model_name: str,
    device: str,
    batch_size: int = 32,
) -> list[np.ndarray]:
    """Return one L2-normalized feature matrix per video."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Encoding requires `pip install 'ur-vc[encoder]'`") from exc

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    if dtype == torch.float16:
        model = model.half()

    encoded: list[np.ndarray] = []
    for path in video_paths:
        pieces: list[torch.Tensor] = []
        batch: list[np.ndarray] = []
        for frame in _frames(path):
            batch.append(frame)
            if len(batch) < batch_size:
                continue
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else None) for k, v in inputs.items()}
            pieces.append(F.normalize(model.get_image_features(**inputs).float(), dim=1).cpu())
            batch.clear()
        if batch:
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else None) for k, v in inputs.items()}
            pieces.append(F.normalize(model.get_image_features(**inputs).float(), dim=1).cpu())
        if not pieces:
            raise RuntimeError(f"No frames decoded from {path}")
        encoded.append(torch.cat(pieces).numpy().astype(np.float32, copy=False))
    return encoded
