"""Minimal, format-preserving LeRobot v2.1 dataset adapter."""

from __future__ import annotations

import json
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .core import CorrectionResult, correct_progress
from .encoder import encode_videos
from .models import resolve_model


@dataclass
class LoadedDataset:
    parquet_paths: list[Path]
    features: np.ndarray
    progress: np.ndarray
    episode_ids: np.ndarray
    frame_counts: list[int]
    encoder: str | None
    embedding_source: str


def read_info(root: Path) -> dict[str, Any]:
    info = json.loads((root / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError("UR-VC supports LeRobot v2.1 only; v3 shared-file datasets are not supported")
    return info


def _episode_index(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected episode filename: {path}") from exc


def parquet_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("data/chunk-*/episode_*.parquet"), key=_episode_index)
    if not paths:
        raise FileNotFoundError(f"No LeRobot episode parquet files under {root}")
    if len({_episode_index(path) for path in paths}) != len(paths):
        raise ValueError("Duplicate episode IDs across parquet files")
    return paths


def _embedding_array(column: pa.ChunkedArray, name: str) -> np.ndarray:
    """Convert a fixed-size-list Arrow column without per-row Python objects."""
    combined = column.combine_chunks()
    if not pa.types.is_fixed_size_list(combined.type):
        raise ValueError(f"{name} must be a fixed-size-list column")
    flat = combined.values.to_numpy(zero_copy_only=False)
    return np.asarray(flat, dtype=np.float32).reshape(len(combined), combined.type.list_size)


def _load_external_embeddings(
    path: Path, camera: str,
) -> tuple[np.ndarray, np.ndarray | None, str | None]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as d:
            features = np.asarray(d["features"], np.float32)
            episodes = np.asarray(d["episode_ids"], np.int64) if "episode_ids" in d else None
            model = str(d["model"].item()) if "model" in d else None
            cached_camera = str(d["camera"].item()) if "camera" in d else None
    elif path.suffix in {".pt", ".pth"}:
        d = torch.load(path, map_location="cpu", weights_only=True)
        features = d["features"].float().numpy()
        episodes = d.get("episode_ids", d.get("video_ids"))
        episodes = episodes.long().numpy() if episodes is not None else None
        model, cached_camera = d.get("model"), d.get("camera")
    else:
        raise ValueError("--embeddings supports .npz, .pt, or .pth files")
    if cached_camera and cached_camera not in {camera, f"observation.images.{camera}"}:
        raise ValueError("External embedding camera does not match --camera")
    return features, episodes, model or None


def _dataset_encoder(root: Path, camera: str) -> str | None:
    path = root / "meta" / "urvc.json"
    if not path.exists():
        return None
    meta = json.loads(path.read_text())
    encoder = meta.get("encoder")
    if isinstance(encoder, dict):
        if encoder.get("embedding_field") == f"observation.embedding.{camera}" or \
                encoder.get("camera_key") == f"observation.images.{camera}":
            return encoder.get("id")
    elif meta.get("camera") in {camera, f"observation.images.{camera}"}:
        return encoder or None
    return None


def _camera_video(root: Path, episode: int, camera: str, chunks_size: int) -> Path:
    return root / "videos" / f"chunk-{episode // chunks_size:03d}" / \
        f"observation.images.{camera}" / f"episode_{episode:06d}.mp4"


def load_dataset(
    root: Path,
    *,
    camera: str,
    embeddings_path: Path | None,
    model_name: str,
    device: str,
    encoder_batch_size: int,
) -> LoadedDataset:
    info = read_info(root)
    paths = parquet_paths(root)
    schemas = [pq.read_schema(path) for path in paths]
    counts = [pq.ParquetFile(path).metadata.num_rows for path in paths]
    if any(n == 0 for n in counts):
        raise ValueError("Empty episodes are not supported")
    episode_ids = np.concatenate([
        np.full(n, _episode_index(path), dtype=np.int64) for path, n in zip(paths, counts, strict=True)
    ])
    raw_parts: list[np.ndarray] = []
    tasks: set[int] = set()
    for path, schema, count in zip(paths, schemas, counts, strict=True):
        if "task_index" in schema.names:
            tasks.update(pq.read_table(path, columns=["task_index"])["task_index"].unique().to_pylist())
        if "episode_index" in schema.names:
            ids = pq.read_table(path, columns=["episode_index"])["episode_index"].unique().to_pylist()
            if ids != [_episode_index(path)]:
                raise ValueError(f"episode_index does not match filename: {path.name}")
        if "progress" in schema.names:
            values = pq.read_table(path, columns=["progress"])["progress"].combine_chunks()
            raw_parts.append(np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float32))
        elif "progress_gt" in schema.names:
            values = pq.read_table(path, columns=["progress_gt"])["progress_gt"].combine_chunks()
            raw_parts.append(np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float32))
        else:
            raw_parts.append(np.arange(count, dtype=np.float32) / max(count, 1))
    progress = np.concatenate(raw_parts)
    if len(tasks) > 1 or (not tasks and info.get("total_tasks", 1) > 1):
        raise ValueError("UR-VC expects a single task; split multi-task datasets before correction")
    if not np.isfinite(progress).all() or np.any((progress < 0) | (progress > 1)):
        raise ValueError("Progress must be finite and in [0, 1]")

    embedding_key = f"observation.embedding.{camera}"
    if embeddings_path is not None:
        features, cached_eps, encoder = _load_external_embeddings(embeddings_path, camera)
        embedding_source = "external"
        if cached_eps is not None and not np.array_equal(cached_eps, episode_ids):
            raise ValueError("External embedding episode_ids do not match the dataset")
    elif all(embedding_key in schema.names for schema in schemas):
        encoder, embedding_source = _dataset_encoder(root, camera), "dataset"
        features = np.concatenate([
            _embedding_array(pq.read_table(path, columns=[embedding_key])[embedding_key], embedding_key)
            for path in paths
        ])
    else:
        encoder, embedding_source = model_name, "encoded"
        chunks_size = int(info.get("chunks_size", 1000))
        videos = [_camera_video(root, _episode_index(path), camera, chunks_size) for path in paths]
        missing = [str(path) for path in videos if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {camera} videos, first: {missing[0]}")
        encoded = encode_videos(
            videos, model_name=model_name, device=device, batch_size=encoder_batch_size
        )
        if [len(x) for x in encoded] != counts:
            pairs = [(len(x), n) for x, n in zip(encoded, counts, strict=True)]
            raise ValueError(f"Video/parquet frame-count mismatch: {pairs[:5]}")
        features = np.concatenate(encoded)

    if len(features) != len(progress):
        raise ValueError(f"Embedding rows ({len(features)}) != dataset rows ({len(progress)})")
    return LoadedDataset(paths, features, progress, episode_ids, counts, encoder, embedding_source)


def _fixed_list(values: np.ndarray) -> pa.Array:
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])


def _write_frame(source: Path, output: Path, corrected: np.ndarray) -> None:
    table = pq.read_table(source)
    column = pa.array(np.asarray(corrected, dtype=np.float32))
    if "urvc_progress" in table.column_names:
        table = table.set_column(table.column_names.index("urvc_progress"), "urvc_progress", column)
    else:
        table = table.append_column("urvc_progress", column)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output, compression="zstd")


def _scalar_stats(values: np.ndarray) -> dict[str, list[float | int]]:
    values = np.asarray(values)
    return {
        "min": [float(values.min())],
        "max": [float(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(len(values))],
    }


def _update_metadata(output: Path, result: CorrectionResult, loaded: LoadedDataset, *, camera: str,
                     tau: float, rho: float) -> None:
    info_path = output / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info.setdefault("features", {})["urvc_progress"] = {
        "dtype": "float32", "shape": [1], "names": None
    }
    info_path.write_text(json.dumps(info, indent=4) + "\n")

    stats_path = output / "meta" / "episodes_stats.jsonl"
    existing: dict[int, dict[str, Any]] = {}
    if stats_path.exists():
        for line in stats_path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                existing[int(item["episode_index"])] = item
    lines, offset = [], 0
    for path, n in zip(loaded.parquet_paths, loaded.frame_counts, strict=True):
        ep = _episode_index(path)
        item = existing.get(ep, {"episode_index": ep, "stats": {}})
        item.setdefault("stats", {})["urvc_progress"] = _scalar_stats(result.progress[offset:offset + n])
        lines.append(json.dumps(item, separators=(",", ":")))
        offset += n
    stats_path.write_text("\n".join(lines) + "\n")

    urvc_meta = {
        "version": "0.1.0",
        "encoder": loaded.encoder,
        "embedding_source": loaded.embedding_source,
        "camera": f"observation.images.{camera}",
        "embedding_key": f"observation.embedding.{camera}",
        "tau": tau,
        "rho": rho,
        "episode_balanced": True,
        "same_episode_excluded": True,
        "output": "urvc_progress",
    }
    (output / "meta" / "urvc.json").write_text(json.dumps(urvc_meta, indent=2) + "\n")


def _publish_directory(staging: Path, output: Path, overwrite: bool) -> None:
    """Install a complete dataset, restoring the old output if the rename fails."""
    if not output.exists():
        staging.rename(output)
        return
    if not overwrite:
        raise FileExistsError(f"Output exists: {output}")
    backup = Path(tempfile.mkdtemp(prefix=f".{output.name}.backup-", dir=output.parent))
    backup.rmdir()
    output.rename(backup)
    try:
        staging.rename(output)
    except BaseException:
        backup.rename(output)
        raise
    try:
        shutil.rmtree(backup)
    except OSError:
        warnings.warn(f"Output is complete; could not remove old-output backup: {backup}", stacklevel=2)


def run_dataset(
    input_root: Path,
    output_root: Path,
    *,
    camera: str = "top_head",
    embeddings_path: Path | None = None,
    embedding_cache: Path | None = None,
    model: str = "giant",
    device: str = "auto",
    tau: float = 0.3,
    rho: float = 0.90,
    query_chunk: int = 64,
    candidate_chunk: int = 131072,
    encoder_batch_size: int = 32,
    overwrite: bool = False,
    show_progress: bool = True,
) -> CorrectionResult:
    if output_root.is_symlink():
        raise ValueError("Output must not be a symlink")
    input_root, output_root = input_root.resolve(), output_root.resolve()
    if input_root.is_relative_to(output_root) or output_root.is_relative_to(input_root):
        raise ValueError("Input and output must be separate, non-nested directories")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_root}; pass --overwrite to replace it")
        if not output_root.is_dir():
            raise ValueError("Output must be a directory")
    if encoder_batch_size <= 0:
        raise ValueError("encoder_batch_size must be positive")
    if embedding_cache is not None:
        embedding_cache = embedding_cache.resolve()
        if embedding_cache.suffix != ".npz":
            raise ValueError("--embedding-cache must end in .npz")
        if any(embedding_cache.is_relative_to(root) for root in [input_root, output_root]):
            raise ValueError("Embedding cache must be outside the input and output directories")
        if embeddings_path is not None and embedding_cache == embeddings_path.resolve():
            raise ValueError("Embedding cache must not overwrite the source embeddings")
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else \
        ("cpu" if device == "auto" else device)
    resolved_model = resolve_model(model)

    loaded = load_dataset(
        input_root,
        camera=camera,
        embeddings_path=embeddings_path,
        model_name=resolved_model,
        device=resolved_device,
        encoder_batch_size=encoder_batch_size,
    )
    result = correct_progress(
        loaded.features,
        loaded.progress,
        loaded.episode_ids,
        tau=tau,
        rho=rho,
        device=resolved_device,
        query_chunk=query_chunk,
        candidate_chunk=candidate_chunk,
        show_progress=show_progress,
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.staging-", dir=output_root.parent) as tmp:
        staging = Path(tmp) / "dataset"
        shutil.copytree(input_root, staging, symlinks=False)
        offset = 0
        for source_path, n in zip(loaded.parquet_paths, loaded.frame_counts, strict=True):
            relative = source_path.relative_to(input_root)
            _write_frame(source_path, staging / relative, result.progress[offset:offset + n])
            offset += n
        _update_metadata(staging, result, loaded, camera=camera, tau=tau, rho=rho)
        if embedding_cache is not None:
            embedding_cache.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".urvc-cache-", dir=embedding_cache.parent) as cache_dir:
                cache = Path(cache_dir) / "features.npz"
                np.savez_compressed(
                    cache, features=loaded.features, episode_ids=loaded.episode_ids,
                    model=np.asarray(loaded.encoder or ""), camera=np.asarray(camera),
                )
                cache.replace(embedding_cache)
        _publish_directory(staging, output_root, overwrite)
    return result
