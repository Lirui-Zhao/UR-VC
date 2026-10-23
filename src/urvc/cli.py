"""Command line interface for UR-VC."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import run_dataset
from .visualize import render_progress_video


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="urvc", description="Correct time-derived robot progress labels.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="add urvc_progress to a LeRobot dataset")
    run.add_argument("input", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--camera", default="top_head")
    run.add_argument("--embeddings", type=Path)
    run.add_argument("--embedding-cache", type=Path)
    run.add_argument(
        "--model",
        default="giant",
        help="giant (default), base, a Hugging Face ID, or local model path",
    )
    run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run.add_argument("--tau", type=float, default=0.3)
    run.add_argument("--rho", type=float, default=0.90)
    run.add_argument("--query-chunk", type=int, default=64)
    run.add_argument("--candidate-chunk", type=int, default=131072)
    run.add_argument("--encoder-batch-size", type=int, default=32)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--quiet", action="store_true")

    viz = sub.add_parser("visualize", help="render top-head video with progress values")
    viz.add_argument("dataset", type=Path)
    viz.add_argument("--episode", type=int, required=True)
    viz.add_argument("--output", type=Path, required=True)
    viz.add_argument("--camera", default="top_head")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "run":
        result = run_dataset(
            args.input,
            args.output,
            camera=args.camera,
            embeddings_path=args.embeddings,
            embedding_cache=args.embedding_cache,
            model=args.model,
            device=args.device,
            tau=args.tau,
            rho=args.rho,
            query_chunk=args.query_chunk,
            candidate_chunk=args.candidate_chunk,
            encoder_batch_size=args.encoder_batch_size,
            overwrite=args.overwrite,
            show_progress=not args.quiet,
        )
        matched = float((result.matched_episodes > 0).mean()) if len(result.progress) else 0.0
        print(f"Wrote {args.output}: {len(result.progress)} frames, match coverage {matched:.1%}")
    else:
        render_progress_video(
            args.dataset, episode=args.episode, output=args.output, camera=args.camera
        )
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
