#!/usr/bin/env python3
"""Atomically add a precomputed ``urvc_progress`` vector to a LeRobot dataset.

The result NPZ must contain one of ``urvc_progress``, ``progress``, or
``corrected_progress``. Optional ``episode_ids``, ``frame_counts``, and
``episode_offsets`` arrays are validated when present. Other arrays (for
example internal support counts) are ignored and never written.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


RESULT_KEYS = ("urvc_progress", "progress", "corrected_progress")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def episode_index(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"unexpected episode filename: {path}") from exc


def scalar_stats(values: np.ndarray) -> dict[str, list[float | int]]:
    return {
        "min": [float(values.min())],
        "max": [float(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(len(values))],
    }


def atomic_json(path: Path, value: Any, *, jsonl: bool = False) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if jsonl:
        text = "\n".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            for item in value
        )
        temporary.write_text(text + "\n", encoding="utf-8")
    else:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    os.replace(temporary, path)


def load_result(
    result_path: Path,
    counts: np.ndarray,
    expected_episode_ids: np.ndarray,
) -> np.ndarray:
    with np.load(result_path, allow_pickle=False) as result:
        selected_key = next((key for key in RESULT_KEYS if key in result), None)
        if selected_key is None:
            raise ValueError(
                f"{result_path} must contain one of {', '.join(RESULT_KEYS)}"
            )
        corrected = np.asarray(result[selected_key], dtype=np.float32).reshape(-1)
        total = int(counts.sum())
        if len(corrected) != total:
            raise ValueError(
                f"result length {len(corrected)} does not match dataset rows {total}"
            )
        if not np.isfinite(corrected).all():
            raise ValueError("urvc_progress contains NaN or infinite values")
        if np.any(corrected < 0) or np.any(corrected > 1):
            low, high = float(corrected.min()), float(corrected.max())
            raise ValueError(
                f"urvc_progress must be in [0,1], observed [{low},{high}]"
            )

        if "frame_counts" in result:
            observed = np.asarray(result["frame_counts"], dtype=np.int64).reshape(-1)
            if not np.array_equal(observed, counts):
                raise ValueError("result frame_counts do not match dataset episodes")
        if "episode_offsets" in result:
            observed = np.asarray(result["episode_offsets"], dtype=np.int64).reshape(-1)
            expected = np.concatenate(
                [np.array([0], dtype=np.int64), np.cumsum(counts)]
            )
            if not np.array_equal(observed, expected):
                raise ValueError("result episode_offsets do not match dataset episodes")
        if "episode_ids" in result:
            observed = np.asarray(result["episode_ids"], dtype=np.int64).reshape(-1)
            if not np.array_equal(observed, expected_episode_ids):
                raise ValueError("result episode_ids do not match dataset row order")
    return corrected


def update_metadata(
    dataset: Path,
    corrected: np.ndarray,
    counts: np.ndarray,
    episode_indices: np.ndarray,
) -> None:
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.setdefault("features", {})["urvc_progress"] = {
        "dtype": "float32",
        "shape": [1],
        "names": None,
    }

    stats_path = dataset / "meta" / "episodes_stats.jsonl"
    stats_items = [
        json.loads(line)
        for line in stats_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_episode = {
        int(item["episode_index"]): item for item in stats_items
    }
    if set(by_episode) != set(int(value) for value in episode_indices):
        raise ValueError("episodes_stats.jsonl does not match parquet episodes")
    offset = 0
    ordered_stats: list[dict[str, Any]] = []
    for index, count in zip(episode_indices, counts, strict=True):
        item = by_episode[int(index)]
        item.setdefault("stats", {})["urvc_progress"] = scalar_stats(
            corrected[offset : offset + int(count)]
        )
        ordered_stats.append(item)
        offset += int(count)

    urvc_path = dataset / "meta" / "urvc.json"
    urvc = (
        json.loads(urvc_path.read_text(encoding="utf-8"))
        if urvc_path.exists()
        else {}
    )
    urvc.setdefault("schema_version", 1)
    urvc.setdefault("matching", {"tau": 0.3, "rho": 0.9})
    urvc.setdefault("fields", {})["progress"] = "progress"
    urvc["fields"]["urvc_progress"] = "urvc_progress"
    urvc["correction"] = {
        "episode_balanced": True,
        "same_episode_excluded": True,
        "fallback": "progress",
    }

    # Metadata is committed only after all Parquet files have been replaced.
    atomic_json(info_path, info)
    atomic_json(stats_path, ordered_stats, jsonl=True)
    atomic_json(urvc_path, urvc)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    result_path = args.result.resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"dataset does not exist: {dataset}")
    if not result_path.is_file():
        raise FileNotFoundError(f"result does not exist: {result_path}")

    paths = sorted(
        dataset.glob("data/chunk-*/episode_*.parquet"), key=episode_index
    )
    if not paths:
        raise FileNotFoundError(f"no episode parquets under {dataset}")
    indices = np.asarray([episode_index(path) for path in paths], dtype=np.int64)
    if not np.array_equal(indices, np.arange(len(paths), dtype=np.int64)):
        raise ValueError("dataset episode indices must be contiguous from zero")
    counts = np.asarray(
        [pq.ParquetFile(path).metadata.num_rows for path in paths], dtype=np.int64
    )
    row_episode_ids = np.repeat(indices, counts)
    corrected = load_result(result_path, counts, row_episode_ids)

    staging = dataset / "._urvc_progress_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        offset = 0
        staged_paths: list[Path] = []
        for position, (source, count) in enumerate(
            zip(paths, counts, strict=True), start=1
        ):
            count_int = int(count)
            table = pq.read_table(source)
            values = pa.array(
                corrected[offset : offset + count_int], type=pa.float32()
            )
            column_index = table.schema.get_field_index("urvc_progress")
            if column_index >= 0:
                table = table.set_column(
                    column_index, "urvc_progress", values
                )
            else:
                table = table.append_column("urvc_progress", values)
            staged = staging / source.name
            pq.write_table(
                table,
                staged,
                compression="zstd",
                compression_level=3,
                use_dictionary=False,
            )
            check = pq.read_table(staged, columns=["urvc_progress"])[
                "urvc_progress"
            ].combine_chunks()
            observed = np.asarray(
                check.to_numpy(zero_copy_only=False), dtype=np.float32
            )
            if not np.array_equal(
                observed, corrected[offset : offset + count_int]
            ):
                raise RuntimeError(f"staged verification failed: {staged}")
            staged_paths.append(staged)
            offset += count_int
            if position % 25 == 0 or position == len(paths):
                print(f"[stage] {position}/{len(paths)}")

        # Each replacement is atomic. The command is idempotent, so an
        # interruption during this short commit loop is safely recoverable by
        # running the same command again.
        for source, staged in zip(paths, staged_paths, strict=True):
            os.replace(staged, source)
        update_metadata(dataset, corrected, counts, indices)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    # Final dataset-level verification.
    offset = 0
    for source, count in zip(paths, counts, strict=True):
        values = pq.read_table(source, columns=["urvc_progress"])[
            "urvc_progress"
        ].combine_chunks()
        observed = np.asarray(
            values.to_numpy(zero_copy_only=False), dtype=np.float32
        )
        count_int = int(count)
        if not np.array_equal(
            observed, corrected[offset : offset + count_int]
        ):
            raise RuntimeError(f"final verification failed: {source}")
        offset += count_int
    print(
        f"[done] wrote urvc_progress for {len(paths)} episodes / "
        f"{len(corrected)} frames"
    )


if __name__ == "__main__":
    main()
