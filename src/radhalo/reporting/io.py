"""Paths and CSV output shared by the analysis scripts."""

import argparse
import csv
import os
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[3]


def parse_args(description: str) -> argparse.Namespace:
    load_dotenv(override=True)  # the project .env wins over the shell environment
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data",
        type=Path,
        default=os.environ.get("RADHALO_DATA", REPO / "data" / "rad-halo"),
        help="directory with the PhysioNet release (rad_halo_dataset.json)",
    )
    parser.add_argument("--results", type=Path, default=REPO / "results")
    return parser.parse_args()


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_metric_values(values: dict, path: Path) -> None:
    write_csv([{"metric": k, "value": v} for k, v in values.items()], path)
