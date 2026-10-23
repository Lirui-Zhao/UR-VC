import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import urvc.dataset as adapter
from urvc.dataset import run_dataset
from urvc.models import BASE_MODEL, GIANT_MODEL


def _write_episode(path, ep, progress, features):
    df = pd.DataFrame({
        "timestamp": np.arange(len(progress), dtype=np.float32) / 30,
        "frame_index": np.arange(len(progress), dtype=np.int64),
        "episode_index": np.full(len(progress), ep, dtype=np.int64),
        "index": np.arange(ep * len(progress), (ep + 1) * len(progress), dtype=np.int64),
        "task_index": np.zeros(len(progress), dtype=np.int64),
        "progress": np.asarray(progress, np.float32),
    })
    table = pa.Table.from_pandas(df, preserve_index=False)
    embedding = pa.FixedSizeListArray.from_arrays(
        pa.array(np.asarray(features, np.float32).reshape(-1)), len(features[0])
    )
    table = table.append_column("observation.embedding.top_head", embedding)
    pq.write_table(table, path / f"episode_{ep:06d}.parquet")


def test_end_to_end_preserves_dataset_and_adds_only_prediction(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    data, meta = source / "data/chunk-000", source / "meta"
    data.mkdir(parents=True)
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1", "chunks_size": 1000, "features": {}
    }))
    (meta / "episodes_stats.jsonl").write_text("")
    _write_episode(data, 0, [.1, .2], [[1, 0], [1, 0]])
    _write_episode(data, 1, [.15, .25], [[1, 0], [1, 0]])

    result = run_dataset(source, output, device="cpu", show_progress=False, query_chunk=1, candidate_chunk=1)
    assert len(result.progress) == 4
    out = pd.read_parquet(output / "data/chunk-000/episode_000000.parquet")
    assert "progress" in out
    assert "observation.embedding.top_head" in out
    assert "urvc_progress" in out
    info = json.loads((output / "meta/info.json").read_text())
    assert info["features"]["urvc_progress"]["dtype"] == "float32"
    assert (output / "meta/urvc.json").exists()


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    data, meta = root / "data/chunk-000", root / "meta"
    data.mkdir(parents=True)
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1", "chunks_size": 1000, "total_tasks": 1, "features": {}
    }))
    for ep in [10, 42]:
        _write_episode(data, ep, [.1, .2], [[1, 0], [1, 0]])
    (meta / "episodes_stats.jsonl").write_text("\n".join(json.dumps({
        "episode_index": ep, "stats": {"action": {"mean": [ep]}}
    }) for ep in [10, 42]) + "\n")
    for camera in ["top_head", "hand_left", "hand_right"]:
        videos = root / "videos/chunk-000" / f"observation.images.{camera}"
        videos.mkdir(parents=True)
        for ep in [10, 42]:
            (videos / f"episode_{ep:06d}.mp4").write_bytes(b"unchanged video bytes")
    return root


def _run(source, output, **kwargs):
    return run_dataset(source, output, device="cpu", show_progress=False, **kwargs)


def _snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_preserves_original_fields_videos_and_noncontiguous_stats(source, tmp_path):
    before = _snapshot(source)
    output = tmp_path / "result"
    _run(source, output)
    assert _snapshot(source) == before
    for relative, contents in before.items():
        if relative.suffix == ".parquet":
            original = pq.read_table(source / relative)
            written = pq.read_table(output / relative)
            for column in original.column_names:
                assert written[column].equals(original[column])
        elif relative.suffix == ".mp4":
            assert (output / relative).read_bytes() == contents
    stats = [json.loads(s) for s in (output / "meta/episodes_stats.jsonl").read_text().splitlines()]
    assert [item["episode_index"] for item in stats] == [10, 42]
    for item in stats:
        assert item["stats"]["action"]["mean"] == [item["episode_index"]]
        assert item["stats"]["urvc_progress"]["count"] == [2]


@pytest.mark.parametrize("target", ["same", "ancestor", "descendant", "symlink"])
def test_rejects_overlapping_paths_before_deletion(source, tmp_path, target):
    before = _snapshot(source)
    output = {"same": source, "ancestor": tmp_path, "descendant": source / "result"}.get(target)
    if target == "symlink":
        output = tmp_path / "alias"
        output.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError):
        _run(source, output, overwrite=True)
    assert _snapshot(source) == before


@pytest.mark.parametrize("stage", ["input", "correction", "write", "metadata", "rename"])
def test_failure_preserves_previous_output_and_input(source, tmp_path, monkeypatch, stage):
    output = tmp_path / "result"
    output.mkdir()
    (output / "important.txt").write_text("previous result")
    before = _snapshot(source)

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    if stage == "input":
        monkeypatch.setattr(adapter, "load_dataset", fail)
    elif stage == "correction":
        monkeypatch.setattr(adapter, "correct_progress", fail)
    elif stage == "write":
        monkeypatch.setattr(adapter, "_write_frame", fail)
    elif stage == "metadata":
        monkeypatch.setattr(adapter, "_update_metadata", fail)
    else:
        rename = Path.rename

        def fail_install(self, target):
            if self.name == "dataset" and target == output:
                fail()
            return rename(self, target)

        monkeypatch.setattr(Path, "rename", fail_install)
    with pytest.raises(OSError, match="injected failure"):
        _run(source, output, overwrite=True)
    assert _snapshot(source) == before
    assert _snapshot(output) == {Path("important.txt"): b"previous result"}
    assert not list(tmp_path.glob(".result.*"))


def test_successful_overwrite_replaces_only_output(source, tmp_path):
    output = tmp_path / "result"
    output.mkdir()
    (output / "old.txt").write_text("old")
    with pytest.raises(FileExistsError):
        _run(source, output)
    _run(source, output, overwrite=True)
    assert not (output / "old.txt").exists()
    assert (output / "meta/urvc.json").is_file()
    assert not list(tmp_path.glob(".result.*"))


@pytest.mark.parametrize("provenance", ["structured", "legacy", "unknown"])
def test_reused_encoder_provenance_is_not_model_argument(source, tmp_path, provenance):
    if provenance == "structured":
        meta = {"encoder": {"id": GIANT_MODEL, "embedding_field": "observation.embedding.top_head"}}
    else:
        meta = {"encoder": GIANT_MODEL, "camera": "observation.images.top_head"}
    if provenance != "unknown":
        (source / "meta/urvc.json").write_text(json.dumps(meta))
    cache = tmp_path / "cache.npz"
    output = tmp_path / "result"
    _run(source, output, model="base", embedding_cache=cache)
    written = json.loads((output / "meta/urvc.json").read_text())
    expected = GIANT_MODEL if provenance != "unknown" else None
    assert written["encoder"] == expected
    assert written["embedding_source"] == "dataset"
    with np.load(cache) as d:
        assert d["model"].item() == (expected or "")
    reused = tmp_path / "reused"
    _run(source, reused, model="base", embeddings_path=cache)
    assert json.loads((reused / "meta/urvc.json").read_text())["encoder"] == expected


@pytest.mark.parametrize("suffix", [".npz", ".pt"])
def test_external_embeddings_check_camera_and_episode_order(source, tmp_path, suffix):
    cache = tmp_path / f"cache{suffix}"

    def save(camera, ids):
        if suffix == ".npz":
            np.savez(cache, features=np.ones((4, 2), np.float32), episode_ids=ids,
                     camera=camera, model=GIANT_MODEL)
        else:
            torch.save({"features": torch.ones((4, 2)), "episode_ids": torch.tensor(ids),
                        "camera": camera, "model": GIANT_MODEL}, cache)

    save("hand_left", [10, 10, 42, 42])
    with pytest.raises(ValueError, match="camera"):
        _run(source, tmp_path / "wrong-camera", embeddings_path=cache)
    save("top_head", [42, 42, 10, 10])
    with pytest.raises(ValueError, match="episode_ids"):
        _run(source, tmp_path / "wrong-order", embeddings_path=cache)
    save("top_head", [10, 10, 42, 42])
    output = tmp_path / "result"
    _run(source, output, embeddings_path=cache, model="base")
    assert json.loads((output / "meta/urvc.json").read_text())["encoder"] == GIANT_MODEL


def test_generated_embeddings_record_actual_encoder(source, tmp_path, monkeypatch):
    for path in source.glob("data/chunk-*/*.parquet"):
        table = pq.read_table(path).drop(["observation.embedding.top_head"])
        pq.write_table(table, path)
    calls = []

    def encode(paths, **kwargs):
        calls.append(kwargs)
        return [np.ones((2, 2), np.float32) for _ in paths]

    monkeypatch.setattr(adapter, "encode_videos", encode)
    output = tmp_path / "result"
    _run(source, output, model="base")
    meta = json.loads((output / "meta/urvc.json").read_text())
    assert calls[0]["model_name"] == BASE_MODEL
    assert meta["encoder"] == BASE_MODEL
    assert meta["embedding_source"] == "encoded"


@pytest.mark.parametrize("version", ["v3.0", "v2.0", None])
def test_rejects_unsupported_format_even_with_cached_features(source, tmp_path, version):
    path = source / "meta/info.json"
    info = json.loads(path.read_text())
    info["codebase_version"] = version
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="v2.1"):
        _run(source, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_rejects_multi_task_data(source, tmp_path):
    path = source / "data/chunk-000/episode_000042.parquet"
    table = pq.read_table(path)
    table = table.set_column(table.column_names.index("task_index"), "task_index", pa.array([1, 1]))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="single task"):
        _run(source, tmp_path / "result")


@pytest.mark.parametrize("location", ["input", "output"])
def test_cache_cannot_modify_dataset(source, tmp_path, location):
    output = tmp_path / "result"
    cache = (source if location == "input" else output) / "cache.npz"
    with pytest.raises(ValueError, match="cache must be outside"):
        _run(source, output, embedding_cache=cache)
