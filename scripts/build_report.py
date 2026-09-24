"""Build the compact cross-language SVA report evidence pack."""

from __future__ import annotations

import argparse
from pathlib import Path

from mllms.reporting import build_evidence_pack


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "reports/cross_language_sva/data",
    )
    args = parser.parse_args()
    build_evidence_pack(root, args.output_dir, iterations=args.iterations)


if __name__ == "__main__":
    main()
