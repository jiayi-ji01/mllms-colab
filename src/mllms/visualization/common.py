"""Small IO and plotting helpers shared by result visualizations."""

import csv
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def latest_by_step(records: list[dict], record_type: str) -> list[dict]:
    by_step = {
        int(record["step"]): record
        for record in records
        if record.get("type") == record_type
    }
    return [by_step[step] for step in sorted(by_step)]


def smooth(values: list[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.array(
        [
            array[max(0, index - window + 1) : index + 1].mean()
            for index in range(len(array))
        ]
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
