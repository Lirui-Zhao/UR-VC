import numpy as np
import pytest

from urvc.core import correct_progress


def test_episode_balanced_mean_and_same_episode_exclusion():
    # Query frame 0 can match one frame from episode 1 and one from episode
    # 2. Duplicate near-identical frames in episode 1 still contribute once.
    features = np.array([
        [1.0, 0.0],  # ep 0 query
        [1.0, 0.0],  # ep 0, must be excluded
        [1.0, 0.0],  # ep 1 winner, progress .2
        [.99, .01],  # ep 1 duplicate, must not be a second vote
        [1.0, 0.0],  # ep 2 winner, progress .4
    ], np.float32)
    progress = np.array([.3, .31, .2, .21, .4], np.float32)
    episodes = np.array([0, 0, 1, 1, 2])
    result = correct_progress(features, progress, episodes, tau=.3, rho=.9, query_chunk=2, candidate_chunk=2)
    assert np.isclose(result.progress[0], .3)
    assert result.matched_episodes[0] == 2


def test_temporal_band_and_no_match_fallback():
    features = np.array([[1.0, 0.0], [1.0, 0.0]], np.float32)
    progress = np.array([.1, .9], np.float32)
    episodes = np.array([0, 1])
    result = correct_progress(features, progress, episodes, tau=.3, rho=.9)
    np.testing.assert_allclose(result.progress, progress)
    np.testing.assert_array_equal(result.matched_episodes, 0)


def test_similarity_floor():
    features = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    progress = np.array([.2, .3], np.float32)
    episodes = np.array([0, 1])
    result = correct_progress(features, progress, episodes, rho=.5)
    np.testing.assert_allclose(result.progress, progress)


def test_arbitrary_episode_ids_and_empty_input():
    features = np.array([[1.0, 0.0], [1.0, 0.0]], np.float32)
    result = correct_progress(features, np.array([.2, .25]), np.array([10, 42]), rho=.9)
    np.testing.assert_allclose(result.progress, [.25, .2])
    empty = correct_progress(np.empty((0, 2), np.float32), np.empty(0), np.empty(0, np.int64))
    assert empty.progress.shape == (0,)


@pytest.mark.parametrize("chunks", [(1, 1), (7, 9), (64, 100)])
def test_matches_independent_numpy_reference(chunks):
    rng = np.random.default_rng(3)
    features = rng.normal(size=(37, 12)).astype(np.float32)
    features[1:3] = features[0]  # Include exact ties across candidate chunks.
    progress = rng.random(37).astype(np.float32)
    episodes = rng.integers(0, 5, 37)
    normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
    expected, counts = [], []
    for i in range(len(progress)):
        votes = []
        for ep in np.unique(episodes):
            if ep == episodes[i]:
                continue
            candidates = np.flatnonzero((episodes == ep) & (np.abs(progress - progress[i]) <= .3))
            if not len(candidates):
                continue
            similarity = normalized[candidates] @ normalized[i]
            if similarity.max() >= .3:
                votes.append(progress[candidates[similarity == similarity.max()]].max())
        expected.append(np.mean(votes) if votes else progress[i])
        counts.append(len(votes))
    result = correct_progress(features, progress, episodes, rho=.3,
                              query_chunk=chunks[0], candidate_chunk=chunks[1])
    np.testing.assert_allclose(result.progress, expected, atol=1e-6)
    np.testing.assert_array_equal(result.matched_episodes, counts)
