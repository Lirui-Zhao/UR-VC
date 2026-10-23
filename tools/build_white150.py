#!/usr/bin/env python3
"""Build the self-contained, white-tablecloth UR-VC 150-episode release set.

The source corpus and feature cache are intentionally passed as arguments.  No
source path, source episode id, or collection-batch id is written to the output.

Selection is reproducible:
  1. retain complete episodes from ``v11_*_lerobot``;
  2. sort by (source dataset directory name, source episode_index);
  3. draw without replacement with ``numpy.random.default_rng(seed)``;
  4. restore stable source order and renumber the draw to 0..N-1.

The task interval is [round(flatten_start_time * fps),
round(fold_end_time * fps)).  Parquet rows, all three videos, and cached
top-head embeddings are trimmed to precisely that interval.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


CAMERAS = (
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
)
EMBEDDING_KEY = "observation.embedding.top_head"
TASK = "Flatten and fold the short sleeve"
ENCODER_ID = "google/siglip2-giant-opt-patch16-384"


@dataclass(frozen=True)
class Episode:
    release_index: int
    source_dataset: Path
    source_index: int
    feature_file: Path
    start_frame: int
    end_frame: int
    raw_length: int

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame

    def source_video(self, camera: str) -> Path:
        return (
            self.source_dataset
            / "videos"
            / "chunk-000"
            / camera
            / f"episode_{self.source_index:06d}.mp4"
        )

    @property
    def source_parquet(self) -> Path:
        return (
            self.source_dataset
            / "data"
            / "chunk-000"
            / f"episode_{self.source_index:06d}.parquet"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--feature-shards-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--version-prefix", default="v11_")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an existing <output>.building directory",
    )
    parser.add_argument(
        "--skip-deep-verify",
        action="store_true",
        help="skip decoded-frame verification (container frame counts are still checked)",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_rows(dataset: Path) -> dict[int, dict[str, str]]:
    candidates = sorted(
        path
        for path in (dataset / "meta").glob("*_record.csv")
        if "error" not in path.name
    )
    if not candidates:
        return {}
    rows: dict[int, dict[str, str]] = {}
    with candidates[0].open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                index = int(Path(row["video"]).stem.split("_")[-1])
            except (KeyError, TypeError, ValueError):
                continue
            rows[index] = row
    return rows


def discover_episodes(args: argparse.Namespace) -> tuple[list[Episode], int]:
    source_root = args.source_root.resolve()
    shards_root = args.feature_shards_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    if not shards_root.is_dir():
        raise FileNotFoundError(f"feature shards root does not exist: {shards_root}")

    source_to_feature: dict[tuple[str, int], tuple[Path, Path]] = {}
    links = sorted(
        shards_root.glob(
            "split_*/videos/chunk-000/"
            "observation.images.top_head/episode_*.mp4"
        )
    )
    for link in links:
        source_video = link.resolve()
        try:
            source_video.relative_to(source_root)
        except ValueError:
            continue
        dataset = source_video.parents[3]
        if not dataset.name.startswith(args.version_prefix):
            continue
        source_index = int(source_video.stem.split("_")[-1])
        split_dir = link.parents[3]
        feature_file = (
            split_dir
            / "features"
            / "chunk-000"
            / "observation.images.top_head"
            / f"{link.stem}.pt"
        )
        key = (dataset.name, source_index)
        if key in source_to_feature:
            raise RuntimeError(f"duplicate cached feature mapping for {key}")
        source_to_feature[key] = (dataset, feature_file)

    per_dataset: dict[Path, tuple[dict[int, int], dict[int, dict[str, str]]]] = {}
    eligible: list[tuple[str, int, Path, Path, int, int, int]] = []
    for (dataset_name, source_index), (dataset, feature_file) in source_to_feature.items():
        if dataset not in per_dataset:
            lengths = {
                int(row["episode_index"]): int(row["length"])
                for row in read_jsonl(dataset / "meta" / "episodes.jsonl")
            }
            per_dataset[dataset] = (lengths, record_rows(dataset))
        lengths, records = per_dataset[dataset]
        row = records.get(source_index)
        raw_length = lengths.get(source_index)
        if row is None or raw_length is None or not feature_file.is_file():
            continue
        try:
            start = int(np.rint(float(row["flatten_start_time"]) * args.fps))
            end = int(np.rint(float(row["fold_end_time"]) * args.fps))
        except (KeyError, TypeError, ValueError):
            continue
        parquet = (
            dataset
            / "data"
            / "chunk-000"
            / f"episode_{source_index:06d}.parquet"
        )
        videos = [
            dataset
            / "videos"
            / "chunk-000"
            / camera
            / f"episode_{source_index:06d}.mp4"
            for camera in CAMERAS
        ]
        if (
            0 <= start < end <= raw_length
            and parquet.is_file()
            and all(video.is_file() for video in videos)
        ):
            eligible.append(
                (
                    dataset_name,
                    source_index,
                    dataset,
                    feature_file,
                    start,
                    end,
                    raw_length,
                )
            )

    eligible.sort(key=lambda item: (item[0], item[1]))
    if len(eligible) < args.episodes:
        raise RuntimeError(
            f"only {len(eligible)} complete episodes, need {args.episodes}"
        )
    rng = np.random.default_rng(args.seed)
    selected_positions = np.sort(
        rng.choice(len(eligible), size=args.episodes, replace=False)
    )
    selected = [eligible[int(position)] for position in selected_positions]
    episodes = [
        Episode(
            release_index=release_index,
            source_dataset=item[2],
            source_index=item[1],
            feature_file=item[3],
            start_frame=item[4],
            end_frame=item[5],
            raw_length=item[6],
        )
        for release_index, item in enumerate(selected)
    ]
    return episodes, len(eligible)


def run_checked(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def probe_video(
    path: Path, ffprobe: str, *, count_frames: bool = False
) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]
    if count_frames:
        command.append("-count_frames")
    command += [
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    result = run_checked(command, capture=True)
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream: {path}")
    return streams[0]


def frame_count(stream: dict[str, Any], *, decoded: bool = False) -> int:
    key = "nb_read_frames" if decoded else "nb_frames"
    value = stream.get(key)
    if value in (None, "N/A"):
        raise RuntimeError(f"video stream lacks {key}: {stream}")
    return int(value)


def preflight_episode(episode: Episode, args: argparse.Namespace) -> None:
    table_rows = pq.ParquetFile(episode.source_parquet).metadata.num_rows
    if table_rows != episode.raw_length:
        raise RuntimeError(
            f"source parquet length mismatch for release episode "
            f"{episode.release_index}: {table_rows} != {episode.raw_length}"
        )
    cached = torch.load(episode.feature_file, map_location="cpu", weights_only=False)
    features = cached["features"] if isinstance(cached, dict) else cached
    if tuple(features.shape) != (episode.raw_length, 1536):
        raise RuntimeError(
            f"feature shape mismatch for release episode {episode.release_index}: "
            f"{tuple(features.shape)}"
        )
    if features.dtype != torch.float32:
        raise RuntimeError(f"expected float32 features, got {features.dtype}")
    norms = features.norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=5e-4, rtol=0):
        raise RuntimeError(
            f"features are not L2-normalized for release episode "
            f"{episode.release_index}"
        )
    for camera in CAMERAS:
        stream = probe_video(episode.source_video(camera), args.ffprobe)
        if frame_count(stream) != episode.raw_length:
            raise RuntimeError(
                f"source {camera} length mismatch for release episode "
                f"{episode.release_index}"
            )
        if stream.get("r_frame_rate") != f"{args.fps}/1":
            raise RuntimeError(
                f"source {camera} is not {args.fps} fps for release episode "
                f"{episode.release_index}"
            )


def feature_stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    numeric = array.astype(np.float64, copy=False)
    return {
        "min": numeric.min(axis=0).tolist(),
        "max": numeric.max(axis=0).tolist(),
        "mean": numeric.mean(axis=0).tolist(),
        "std": numeric.std(axis=0).tolist(),
        "count": [int(numeric.shape[0])],
    }


def set_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index >= 0:
        return table.set_column(index, name, values)
    return table.append_column(name, values)


def write_episode_data(
    episode: Episode,
    output_root: Path,
    global_start: int,
    stats_dir: Path,
) -> None:
    output = (
        output_root
        / "data"
        / "chunk-000"
        / f"episode_{episode.release_index:06d}.parquet"
    )
    stats_path = stats_dir / f"episode_{episode.release_index:06d}.numeric.json"
    if output.is_file() and stats_path.is_file():
        if pq.ParquetFile(output).metadata.num_rows == episode.length:
            return

    table = pq.read_table(episode.source_parquet).slice(
        episode.start_frame, episode.length
    )
    cached = torch.load(episode.feature_file, map_location="cpu", weights_only=False)
    tensor = cached["features"] if isinstance(cached, dict) else cached
    embeddings = (
        tensor[episode.start_frame : episode.end_frame]
        .contiguous()
        .numpy()
        .astype(np.float32, copy=False)
    )
    if embeddings.shape != (episode.length, 1536):
        raise RuntimeError(
            f"trimmed feature shape mismatch for release episode "
            f"{episode.release_index}: {embeddings.shape}"
        )

    length = episode.length
    frame_index = np.arange(length, dtype=np.int64)
    timestamp = (frame_index / np.float32(30.0)).astype(np.float32)
    episode_index = np.full(length, episode.release_index, dtype=np.int64)
    global_index = np.arange(
        global_start, global_start + length, dtype=np.int64
    )
    task_index = np.zeros(length, dtype=np.int64)
    progress = (frame_index / np.float32(length)).astype(np.float32)

    replacements: dict[str, pa.Array] = {
        "timestamp": pa.array(timestamp, type=pa.float32()),
        "frame_index": pa.array(frame_index, type=pa.int64()),
        "episode_index": pa.array(episode_index, type=pa.int64()),
        "index": pa.array(global_index, type=pa.int64()),
        "task_index": pa.array(task_index, type=pa.int64()),
        "progress": pa.array(progress, type=pa.float32()),
        EMBEDDING_KEY: pa.FixedSizeListArray.from_arrays(
            pa.array(embeddings.reshape(-1), type=pa.float32()), 1536
        ),
    }
    for name, values in replacements.items():
        table = set_column(table, name, values)
    table = table.replace_schema_metadata(None)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.parquet")
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
    )
    os.replace(temporary, output)

    numeric: dict[str, Any] = {}
    for name in (
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "progress",
    ):
        column = table[name].combine_chunks()
        if pa.types.is_fixed_size_list(column.type):
            values = column.values.to_numpy(zero_copy_only=False).reshape(length, -1)
        else:
            values = column.to_numpy(zero_copy_only=False)
        numeric[name] = feature_stats(values)
    numeric[EMBEDDING_KEY] = feature_stats(embeddings)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(numeric, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def video_stats(path: Path, length: int, ffmpeg: str) -> dict[str, Any]:
    result = run_checked(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            "fps=3,scale=160:120:flags=area",
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture=True,
    )
    pixels = np.frombuffer(result.stdout, dtype=np.uint8)
    frame_size = 160 * 120 * 3
    if not len(pixels) or len(pixels) % frame_size:
        raise RuntimeError(f"could not sample video for stats: {path}")
    pixels = pixels.reshape(-1, 120, 160, 3).astype(np.float64) / 255.0
    minimum = pixels.min(axis=(0, 1, 2))
    maximum = pixels.max(axis=(0, 1, 2))
    mean = pixels.mean(axis=(0, 1, 2))
    std = pixels.std(axis=(0, 1, 2))
    nest = lambda values: [[[float(value)]] for value in values]
    return {
        "min": nest(minimum),
        "max": nest(maximum),
        "mean": nest(mean),
        "std": nest(std),
        "count": [length],
    }


def encode_video(
    episode: Episode,
    camera: str,
    output_root: Path,
    stats_dir: Path,
    args: argparse.Namespace,
) -> None:
    output = (
        output_root
        / "videos"
        / "chunk-000"
        / camera
        / f"episode_{episode.release_index:06d}.mp4"
    )
    stats_path = (
        stats_dir
        / f"episode_{episode.release_index:06d}.{camera}.json"
    )
    valid = False
    if output.is_file():
        try:
            valid = frame_count(probe_video(output, args.ffprobe)) == episode.length
        except Exception:
            valid = False
    if not valid:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.tmp.mp4")
        if temporary.exists():
            temporary.unlink()
        run_checked(
            [
                args.ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(episode.source_video(camera)),
                "-vf",
                (
                    f"trim=start_frame={episode.start_frame}:"
                    f"end_frame={episode.end_frame},setpts=PTS-STARTPTS"
                ),
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "8",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(temporary),
            ]
        )
        stream = probe_video(temporary, args.ffprobe)
        if frame_count(stream) != episode.length:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"trimmed video length mismatch for release episode "
                f"{episode.release_index}, {camera}"
            )
        os.replace(temporary, output)
    if not stats_path.is_file():
        stats = video_stats(output, episode.length, args.ffmpeg)
        stats_path.write_text(
            json.dumps(stats, separators=(",", ":")), encoding="utf-8"
        )


def write_metadata(
    output_root: Path,
    episodes: list[Episode],
    stats_dir: Path,
) -> None:
    total_frames = sum(episode.length for episode in episodes)
    video_feature = {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30,
            "video.channels": 3,
            "has_audio": False,
        },
    }
    features: dict[str, Any] = {
        camera: dict(video_feature) for camera in CAMERAS
    }
    features.update(
        {
            "observation.state": {
                "dtype": "float32",
                "shape": [14],
                "names": None,
            },
            "action": {"dtype": "float32", "shape": [14], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "progress": {"dtype": "float32", "shape": [1], "names": None},
            EMBEDDING_KEY: {
                "dtype": "float32",
                "shape": [1536],
                "names": None,
            },
        }
    )
    info = {
        "codebase_version": "v2.1",
        "robot_type": "agilex",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episodes) * len(CAMERAS),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    urvc = {
        "schema_version": 1,
        "encoder": {
            "id": ENCODER_ID,
            "camera_key": "observation.images.top_head",
            "embedding_field": EMBEDDING_KEY,
            "embedding_dim": 1536,
            "embedding_dtype": "float32",
            "l2_normalized": True,
        },
        "matching": {"tau": 0.3, "rho": 0.9},
        "fields": {
            "progress": "progress",
            "urvc_progress": "urvc_progress",
        },
    }
    meta = output_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (meta / "urvc.json").write_text(
        json.dumps(urvc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": TASK}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (meta / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(
                json.dumps(
                    {
                        "episode_index": episode.release_index,
                        "tasks": [TASK],
                        "length": episode.length,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (meta / "episodes_stats.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            stats = json.loads(
                (
                    stats_dir
                    / f"episode_{episode.release_index:06d}.numeric.json"
                ).read_text(encoding="utf-8")
            )
            for camera in CAMERAS:
                stats[camera] = json.loads(
                    (
                        stats_dir
                        / f"episode_{episode.release_index:06d}.{camera}.json"
                    ).read_text(encoding="utf-8")
                )
            handle.write(
                json.dumps(
                    {"episode_index": episode.release_index, "stats": stats},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def validate_output(
    output_root: Path,
    episodes: list[Episode],
    args: argparse.Namespace,
) -> None:
    expected_global = 0
    for episode in episodes:
        parquet = (
            output_root
            / "data"
            / "chunk-000"
            / f"episode_{episode.release_index:06d}.parquet"
        )
        schema = pq.read_schema(parquet)
        embedding_type = schema.field(EMBEDDING_KEY).type
        if not (
            pa.types.is_fixed_size_list(embedding_type)
            and embedding_type.list_size == 1536
            and embedding_type.value_type == pa.float32()
        ):
            raise RuntimeError(f"bad embedding schema: {parquet}")
        table = pq.read_table(
            parquet,
            columns=[
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
                "progress",
            ],
        )
        if table.num_rows != episode.length:
            raise RuntimeError(f"bad parquet row count: {parquet}")
        values = table.to_pydict()
        frame_index = np.asarray(values["frame_index"], dtype=np.int64)
        global_index = np.asarray(values["index"], dtype=np.int64)
        progress = np.asarray(values["progress"], dtype=np.float32)
        if not np.array_equal(frame_index, np.arange(episode.length)):
            raise RuntimeError(f"bad frame_index: {parquet}")
        if not np.array_equal(
            global_index,
            np.arange(expected_global, expected_global + episode.length),
        ):
            raise RuntimeError(f"bad global index: {parquet}")
        if not np.all(
            np.asarray(values["episode_index"]) == episode.release_index
        ):
            raise RuntimeError(f"bad episode_index: {parquet}")
        if not np.all(np.asarray(values["task_index"]) == 0):
            raise RuntimeError(f"bad task_index: {parquet}")
        expected_progress = (
            np.arange(episode.length, dtype=np.float32)
            / np.float32(episode.length)
        )
        if not np.array_equal(progress, expected_progress):
            raise RuntimeError(f"bad progress: {parquet}")
        expected_global += episode.length

    video_jobs = [
        (
            output_root
            / "videos"
            / "chunk-000"
            / camera
            / f"episode_{episode.release_index:06d}.mp4",
            episode.length,
        )
        for episode in episodes
        for camera in CAMERAS
    ]

    def validate_video(job: tuple[Path, int]) -> None:
        path, expected = job
        stream = probe_video(
            path, args.ffprobe, count_frames=not args.skip_deep_verify
        )
        actual = frame_count(stream, decoded=not args.skip_deep_verify)
        if actual != expected:
            raise RuntimeError(f"bad decoded frame count: {path}: {actual}")
        if stream.get("codec_name") != "h264":
            raise RuntimeError(f"bad output codec: {path}")
        if stream.get("r_frame_rate") != f"{args.fps}/1":
            raise RuntimeError(f"bad output fps: {path}")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        list(executor.map(validate_video, video_jobs))

    if any(path.is_symlink() for path in output_root.rglob("*")):
        raise RuntimeError("output must not contain symlinks")
    metadata_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (output_root / "meta").iterdir()
        if path.is_file()
    )
    for forbidden in (
        str(args.source_root.resolve()),
        str(args.feature_shards_root.resolve()),
    ):
        if forbidden in metadata_text:
            raise RuntimeError("source provenance leaked into output metadata")


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if shutil.which(args.ffmpeg) is None or shutil.which(args.ffprobe) is None:
        raise RuntimeError("ffmpeg and ffprobe are required")

    episodes, eligible_count = discover_episodes(args)
    total_raw = sum(episode.raw_length for episode in episodes)
    total_trimmed = sum(episode.length for episode in episodes)
    source_batches = len({episode.source_dataset.name for episode in episodes})
    print(
        f"[selection] eligible={eligible_count}, selected={len(episodes)}, "
        f"source_batches={source_batches}, seed={args.seed}"
    )
    print(
        f"[selection] raw_frames={total_raw}, trimmed_frames={total_trimmed}, "
        f"length_range=[{min(e.length for e in episodes)},"
        f"{max(e.length for e in episodes)}]"
    )

    print("[preflight] validating parquets, full embeddings, and three cameras")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.workers, 8)
    ) as executor:
        list(executor.map(lambda episode: preflight_episode(episode, args), episodes))
    print("[preflight] OK")
    if args.preflight_only:
        return

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    building = output.with_name(f"{output.name}.building")
    if building.exists() and not args.resume:
        raise FileExistsError(
            f"partial build exists; pass --resume or inspect it first: {building}"
        )
    building.mkdir(parents=True, exist_ok=True)
    stats_dir = building / "._build_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    print("[data] writing trimmed full-control parquets and 1536-d embeddings")
    global_start = 0
    for position, episode in enumerate(episodes, start=1):
        write_episode_data(episode, building, global_start, stats_dir)
        global_start += episode.length
        if position % 10 == 0 or position == len(episodes):
            print(f"[data] {position}/{len(episodes)}")

    print("[video] frame-exact H.264 trim for all three cameras")
    video_jobs = [
        (episode, camera)
        for episode in episodes
        for camera in CAMERAS
    ]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = [
            executor.submit(
                encode_video, episode, camera, building, stats_dir, args
            )
            for episode, camera in video_jobs
        ]
        for position, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            future.result()
            if position % 30 == 0 or position == len(futures):
                print(f"[video] {position}/{len(futures)}")

    print("[metadata] writing clean LeRobot v2.1 metadata and episode stats")
    write_metadata(building, episodes, stats_dir)
    print("[verify] validating indices, schemas, and decoded video frame counts")
    validate_output(building, episodes, args)
    shutil.rmtree(stats_dir)
    os.replace(building, output)
    size_bytes = sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file()
    )
    print(
        f"[done] {output}: episodes={len(episodes)}, frames={total_trimmed}, "
        f"videos={len(video_jobs)}, bytes={size_bytes}"
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace")
        print(detail, file=sys.stderr)
        raise
