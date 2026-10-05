"""Derive `derived/prehighlighted_findings.csv` (App. B.3 coverage numbers).

Inputs: the resident annotation tasks (output of `02_import_prehighlight_review.py`, distributed to the
residents) and the release `rad_halo_dataset.json`. A findings sentence counts as pre-highlighted if its
text equals the text of a highlighted span after lower-casing and collapsing whitespace (spans may store
newlines as the two characters "\\n"). Writes `radhalo_id, sentence_id` (index into `findings_sentences`).
"""

import argparse
import csv
import re
from pathlib import Path

from labelstudio import FINDINGS, read_json

from radhalo.data.release import load_reports


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\\n", " ")).strip().lower()


def prehighlighted(reports: list[dict], tasks: list[dict]) -> list[dict]:
    by_id = {t["data"]["generated_impression_id"]: t for t in tasks}
    rows = []
    for report in reports:
        task = by_id[report["radhalo_id"]]
        spans = {
            normalize(r["value"]["text"])
            for prediction in task["predictions"]
            for r in prediction["result"]
            if r["from_name"] == FINDINGS[0]
        }
        rows += [
            {"radhalo_id": report["radhalo_id"], "sentence_id": i}
            for i, sentence in enumerate(report["findings_sentences"])
            if normalize(sentence["text"]) in spans
        ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="resident annotation tasks (JSON)")
    parser.add_argument("--release", type=Path, required=True, help="release data directory")
    parser.add_argument("--out", type=Path, default=Path("derived/prehighlighted_findings.csv"))
    args = parser.parse_args()

    rows = prehighlighted(load_reports(args.release), read_json(args.tasks))
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["radhalo_id", "sentence_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} pre-highlighted findings sentences -> {args.out}")


if __name__ == "__main__":
    main()
