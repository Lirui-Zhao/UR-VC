#!/usr/bin/env python3
"""Package UR-VC-150 and a self-contained three-episode example for a release.

Usage: python tools/prepare_release.py --source ../UR-VC-150 --output artifacts/v0.1.0
The source is read-only. The output directory must not already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from urvc.dataset import parquet_paths, read_info, run_dataset


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))


def validate(root: Path) -> dict:
    info = read_info(root)
    paths = parquet_paths(root)
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    counts = [pq.ParquetFile(path).metadata.num_rows for path in paths]
    if [int(path.stem.split("_")[-1]) for path in paths] != list(range(len(paths))):
        raise ValueError("Release episode IDs must be contiguous from zero")
    if counts != [ep["length"] for ep in episodes] or sum(counts) != info["total_frames"]:
        raise ValueError("Episode lengths do not match metadata")
    if len(paths) != info["total_episodes"]:
        raise ValueError("Episode count does not match metadata")
    cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    for ep in range(len(paths)):
        for camera in cameras:
            path = root / info["video_path"].format(
                episode_chunk=ep // info["chunks_size"], video_key=camera, episode_index=ep,
            )
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Missing or linked video: {path}")
    for path in paths:
        values = pq.read_table(path, columns=["urvc_progress"])["urvc_progress"].to_numpy()
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"Invalid progress: {path}")
    return {"episodes": len(paths), "frames": sum(counts), "videos": len(paths) * len(cameras)}


def make_mini(source: Path, working: Path, output: Path) -> dict:
    ids = {0, 1, 2}
    info = read_info(source)
    (working / "meta").mkdir(parents=True)
    for name in ["info.json", "tasks.jsonl", "urvc.json"]:
        shutil.copy2(source / "meta" / name, working / "meta" / name)
    for name in ["episodes.jsonl", "episodes_stats.jsonl"]:
        rows = [row for row in read_jsonl(source / "meta" / name) if row["episode_index"] in ids]
        write_jsonl(working / "meta" / name, rows)
    episodes = read_jsonl(working / "meta/episodes.jsonl")
    if [e["episode_index"] for e in episodes] != [0, 1, 2]:
        raise ValueError("Source must contain episodes 0, 1, and 2")
    for ep in sorted(ids):
        relative = Path(info["data_path"].format(episode_chunk=0, episode_index=ep))
        (working / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, working / relative)
        for camera, feature in info["features"].items():
            if feature["dtype"] != "video":
                continue
            relative = Path(info["video_path"].format(episode_chunk=0, episode_index=ep, video_key=camera))
            (working / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, working / relative)
    info.update(total_episodes=3, total_frames=sum(e["length"] for e in episodes),
                total_videos=9, total_chunks=1, splits={"train": "0:3"})
    (working / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    shutil.copy2(source / "LICENSE", working / "LICENSE")
    result = run_dataset(working, output, device="cpu", show_progress=False)
    summary = validate(output)
    summary.update(tau=0.3, rho=0.9, matched_frames=int((result.matched_episodes > 0).sum()),
                   match_coverage=float((result.matched_episodes > 0).mean()),
                   mean_progress=float(result.progress.mean()))
    (output / "meta/reference.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "README.md").write_text(
        "# UR-VC-mini\n\n"
        "The first three complete episodes of UR-VC-150: 3,009 frames at 30 FPS, "
        "all three cameras, state/action data, and precomputed SigLIP 2 giant embeddings.\n\n"
        "`urvc_progress` was recomputed using only these three episodes with tau=0.3 "
        "and rho=0.90. It differs from the full dataset's labels. "
        "Expected summary values are in `meta/reference.json`. "
        "This is a runnable example, not the paper's evaluation split.\n\n"
        "Run correction and visualization using the "
        "[quick start](https://github.com/Lirui-Zhao/UR-VC#quick-start).\n\n"
        "Copyright (c) 2026 UR-VC Authors. Licensed under "
        "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see LICENSE. "
        "Please cite [UR-VC](https://arxiv.org/abs/2607.12892).\n"
    )
    shutil.rmtree(working)
    return summary


def archive(root: Path, output: Path) -> dict:
    with tarfile.open(output, "w:gz", compresslevel=1) as tar:
        tar.add(root, arcname=root.name)
    if output.stat().st_size >= 2 * 1024**3:
        raise ValueError("Archive exceeds the GitHub Release per-file limit")
    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"file": output.name, "bytes": output.stat().st_size, "sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        parser.error("Source and output must be separate, non-nested directories")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    validate(source)
    mini = make_mini(source, output / "mini-input", output / "UR-VC-mini")
    # Recompute both releases with this code version; do not ship stale labels.
    torch.set_num_threads(8)
    result = run_dataset(source, output / "UR-VC-150", device="cpu",
                         query_chunk=256, candidate_chunk=65536)
    full = validate(output / "UR-VC-150")
    full.update(tau=0.3, rho=0.9, matched_frames=int((result.matched_episodes > 0).sum()),
                match_coverage=float((result.matched_episodes > 0).mean()),
                mean_progress=float(result.progress.mean()))
    (output / "UR-VC-150/meta/reference.json").write_text(json.dumps(full, indent=2) + "\n")
    archives = [archive(output / "UR-VC-mini", output / "UR-VC-mini.tar.gz"),
                archive(output / "UR-VC-150", output / "UR-VC-150.tar.gz")]
    (output / "SHA256SUMS").write_text("".join(f"{item['sha256']}  {item['file']}\n" for item in archives))
    report = {"mini": mini, "full": full, "archives": archives}
    (output / "release-data.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
