"""Merge the physician-reviewed imaging modality and anatomical region into the dataset (Sec. 3.2).

Inputs: `rad-halo_dataset.json` and the reviewed sheet of `05_classify_imaging.py`. Adds
`imaging_technique` and `anatomy_location` to every report ("Unknown" if missing from the sheet).
"""

import argparse
from pathlib import Path

from labelstudio import read_json, write_json

UNKNOWN = "Unknown"


def merge(reports: list[dict], rows: list[dict]) -> list[int]:
    """Add the labels in place; returns the ids missing from `rows` (later rows win for duplicates)."""
    labels = {int(row["radhalo_id"]): (row["imaging_technique"], row["anatomy_location"]) for row in rows}
    missing = []
    for report in reports:
        if report["radhalo_id"] not in labels:
            missing.append(report["radhalo_id"])
        report["imaging_technique"], report["anatomy_location"] = labels.get(
            report["radhalo_id"], (UNKNOWN, UNKNOWN)
        )
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="rad-halo_dataset.json")
    parser.add_argument("--reviewed", type=Path, required=True, help="reviewed Excel sheet")
    parser.add_argument(
        "--out", type=Path, required=True, help="e.g. rad-halo_dataset_imaging_technique.json"
    )
    args = parser.parse_args()

    import pandas as pd

    reports = read_json(args.dataset)
    missing = merge(reports, pd.read_excel(args.reviewed, engine="openpyxl").to_dict("records"))
    if missing:
        print(f"{len(missing)} reports missing from the reviewed sheet: {missing}")
    write_json(reports, args.out)
    print(f"{len(reports)} reports -> {args.out}")


if __name__ == "__main__":
    main()
