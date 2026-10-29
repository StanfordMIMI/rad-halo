"""List which residents annotated each report (`label_studio/assignments.csv`, App. B).

Derived from the release labels. `partition` is the Label Studio task file a resident received:
1 = ids 0-49 (annotator_1 and annotator_2), 2 = ids 50-174 (annotator_1), 3 = ids 175-299 (annotator_2),
4 = annotator_3's half of ids 50-299, 5 = annotator_4's half, and `extra` = the 21 reports of partition 4
that annotator_4 annotated in addition. Partitions 4 and 5 were drawn outside the committed code.
"""

import argparse
import csv
from pathlib import Path

from labelstudio import ANNOTATORS

from radhalo.data.release import ERROR_TYPES, load_reports, resident_labels, sentences

PARTITION_BOUNDS = (50, 175)  # first ids of partitions 2 and 3


def report_annotators(report: dict) -> list[str]:
    raters = {
        k for error_type in ERROR_TYPES for s in sentences(report, error_type) for k in resident_labels(s)
    }
    return [a for a in ANNOTATORS if a in raters]


def partition(rid: int, annotator: str, annotators: list[str]) -> str:
    if annotator == "annotator_3":
        return "4"
    if annotator == "annotator_4":
        return "extra" if "annotator_3" in annotators else "5"
    if rid < PARTITION_BOUNDS[0]:
        return "1"
    return "2" if rid < PARTITION_BOUNDS[1] else "3"


def assignment_rows(reports: list[dict]) -> list[dict]:
    rows = []
    for report in reports:
        annotators = report_annotators(report)
        rows += [
            {
                "radhalo_id": report["radhalo_id"],
                "annotator": a,
                "partition": partition(report["radhalo_id"], a, annotators),
            }
            for a in annotators
        ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True, help="release data directory")
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parents[1] / "label_studio" / "assignments.csv"
    )
    args = parser.parse_args()

    rows = assignment_rows(load_reports(args.release))
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["radhalo_id", "annotator", "partition"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} assignments -> {args.out}")


if __name__ == "__main__":
    main()
