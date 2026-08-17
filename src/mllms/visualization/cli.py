"""Create figures and CSV summaries from saved experiment outputs."""

import argparse
from pathlib import Path

from mllms.visualization.blimp import plot_blimp
from mllms.visualization.patching import plot_patching
from mllms.visualization.sva import plot_sva
from mllms.visualization.training import plot_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="report", required=True)
    training = subparsers.add_parser("training")
    training.add_argument("--run-dir", type=Path, required=True)
    training.add_argument("--output-dir", type=Path)
    training.add_argument("--smooth-window", type=int, default=20)
    blimp = subparsers.add_parser("blimp")
    blimp.add_argument("--results-dir", type=Path, required=True)
    blimp.add_argument("--output-dir", type=Path)
    sva = subparsers.add_parser("sva")
    sva.add_argument("--results-dir", type=Path, required=True)
    sva.add_argument("--output-dir", type=Path)
    patching = subparsers.add_parser("patching")
    patching.add_argument("--results-dir", type=Path, required=True)
    patching.add_argument("--output-dir", type=Path)
    patching.add_argument("--top-k", type=int, default=50)
    patching.add_argument(
        "--language", choices=("original", "clone"), default="original"
    )
    patching.add_argument(
        "--metric", choices=("delta_ld", "recovery"), default="recovery"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report == "training":
        plot_training(args.run_dir, args.output_dir or args.run_dir, args.smooth_window)
    elif args.report == "blimp":
        plot_blimp(args.results_dir, args.output_dir or args.results_dir)
    elif args.report == "sva":
        plot_sva(args.results_dir, args.output_dir or args.results_dir)
    else:
        plot_patching(
            args.results_dir,
            args.output_dir or args.results_dir,
            args.top_k,
            args.language,
            args.metric,
        )


if __name__ == "__main__":
    main()
