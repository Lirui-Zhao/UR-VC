"""The training-free, episode-balanced UR-VC correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


@dataclass(frozen=True)
class CorrectionResult:
    """Per-frame corrected progress and the number of contributing episodes."""

    progress: np.ndarray
    matched_episodes: np.ndarray


def _as_tensor(value: np.ndarray | torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


@torch.inference_mode()
def correct_progress(
    features: np.ndarray | torch.Tensor,
    progress: np.ndarray | torch.Tensor,
    episode_ids: np.ndarray | torch.Tensor,
    *,
    tau: float = 0.3,
    rho: float = 0.90,
    device: str | torch.device = "cpu",
    query_chunk: int = 64,
    candidate_chunk: int = 131072,
    show_progress: bool = False,
) -> CorrectionResult:
    """Correct progress using one best matching frame per other episode.

    The implementation is exact and candidate-chunked: it never materializes
    the full frame-by-frame similarity matrix. Ties follow the original
    implementation and keep the largest progress value among equal-similarity
    representatives.
    """
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and non-negative")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")
    if query_chunk <= 0 or candidate_chunk <= 0:
        raise ValueError("chunk sizes must be positive")

    dev = torch.device(device)
    feats = _as_tensor(features, dtype=torch.float32, device=dev)
    raw = _as_tensor(progress, dtype=torch.float32, device=dev).flatten()
    eps_in = _as_tensor(episode_ids, dtype=torch.int64, device=dev).flatten()
    if feats.ndim != 2:
        raise ValueError(f"features must be [frames, dim], got {tuple(feats.shape)}")
    if len(feats) != len(raw) or len(raw) != len(eps_in):
        raise ValueError("features, progress, and episode_ids must have equal length")
    if len(feats) == 0:
        return CorrectionResult(np.empty(0, np.float32), np.empty(0, np.int64))
    if not torch.isfinite(feats).all() or not torch.isfinite(raw).all():
        raise ValueError("features and progress must be finite")

    # Compact arbitrary episode IDs to 0..E-1 for scatter-reduce.
    _, eps = torch.unique(eps_in, sorted=True, return_inverse=True)
    n_episodes = int(eps.max().item()) + 1
    feats = F.normalize(feats, dim=1)
    output = raw.clone()
    support = torch.zeros(len(raw), dtype=torch.int64, device=dev)
    neg_inf = torch.tensor(float("-inf"), device=dev)

    starts = range(0, len(raw), query_chunk)
    iterator = tqdm(starts, total=(len(raw) + query_chunk - 1) // query_chunk, desc="UR-VC") \
        if show_progress else starts
    for qs in iterator:
        qe = min(qs + query_chunk, len(raw))
        qf, qp, qep = feats[qs:qe], raw[qs:qe], eps[qs:qe]
        qn = qe - qs
        best_sim = torch.full((qn, n_episodes), float("-inf"), device=dev)
        best_label = torch.full((qn, n_episodes), float("nan"), device=dev)

        for cs in range(0, len(raw), candidate_chunk):
            ce = min(cs + candidate_chunk, len(raw))
            sim = qf @ feats[cs:ce].T
            cep = eps[cs:ce]
            cp = raw[cs:ce]
            sim.masked_fill_(qep[:, None] == cep[None, :], float("-inf"))
            sim.masked_fill_((qp[:, None] - cp[None, :]).abs() > tau, float("-inf"))

            scatter_index = cep[None, :].expand(qn, -1)
            chunk_best = torch.full_like(best_sim, float("-inf"))
            chunk_best.scatter_reduce_(1, scatter_index, sim, reduce="amax", include_self=True)

            # Recover the progress attached to the per-episode maximum. If
            # several frames tie exactly, select the largest progress, matching
            # the research implementation's second scatter-max.
            win = torch.isfinite(sim) & (sim == chunk_best.gather(1, scatter_index))
            labels = torch.where(win, cp[None, :].expand_as(sim), neg_inf)
            chunk_label = torch.full_like(best_sim, float("-inf"))
            chunk_label.scatter_reduce_(1, scatter_index, labels, reduce="amax", include_self=True)

            improve = chunk_best > best_sim
            tie = (chunk_best == best_sim) & torch.isfinite(chunk_best)
            best_label = torch.where(improve, chunk_label, best_label)
            best_label = torch.where(tie, torch.maximum(best_label, chunk_label), best_label)
            best_sim = torch.maximum(best_sim, chunk_best)

        valid = best_sim >= rho
        count = valid.sum(dim=1)
        votes = torch.where(valid, best_label, torch.zeros_like(best_label))
        corrected = votes.sum(dim=1) / count.clamp_min(1)
        output[qs:qe] = torch.where(count > 0, corrected, qp)
        support[qs:qe] = count

    return CorrectionResult(
        output.detach().cpu().numpy().astype(np.float32, copy=False),
        support.detach().cpu().numpy().astype(np.int64, copy=False),
    )
