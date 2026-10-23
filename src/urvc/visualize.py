"""Render a compact progress overlay for one episode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import read_info


def render_progress_video(
    dataset: Path,
    *,
    episode: int,
    output: Path,
    camera: str = "top_head",
) -> None:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Visualization requires `pip install 'ur-vc[visualize]'`") from exc

    info = read_info(dataset)
    chunk = episode // int(info.get("chunks_size", 1000))
    parquet = dataset / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
    video = dataset / "videos" / f"chunk-{chunk:03d}" / \
        f"observation.images.{camera}" / f"episode_{episode:06d}.mp4"
    if output.resolve().is_relative_to(dataset.resolve()):
        raise ValueError("Visualization output must be outside the dataset")
    df = pd.read_parquet(parquet)
    if "urvc_progress" not in df:
        raise ValueError("Dataset has no urvc_progress column")
    raw = np.asarray(df["progress"] if "progress" in df else
                     df["progress_gt"] if "progress_gt" in df else
                     np.arange(len(df)) / max(len(df), 1))
    corrected = np.asarray(df["urvc_progress"])

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(info.get("fps", 30))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError(f"Could not create video: {output}")
    try:
        i = 0
        while i < len(df):
            ok, frame = cap.read()
            if not ok:
                break
            cv2.rectangle(frame, (12, 12), (320, 82), (0, 0, 0), -1)
            cv2.putText(frame, f"frame {i:04d}", (22, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"time progress  {raw[i]:.3f}", (22, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (190, 190, 190), 1, cv2.LINE_AA)
            cv2.putText(frame, f"UR-VC          {corrected[i]:.3f}", (22, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            i += 1
    finally:
        cap.release()
        writer.release()
    if i != len(df):
        raise ValueError(f"Video has {i} decoded frames but parquet has {len(df)} rows")
