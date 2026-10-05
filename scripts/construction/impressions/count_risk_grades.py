"""Count MedVAL risk grades of all impressions generated until the 5/5/5/35 targets were met.

Writes derived/risk_grade_counts.csv (App. A.7). Some sources were generated in a
longer run and later truncated to the final targets; the generation order is restored
from the seeded shuffle of the eligible reports, then buckets are filled until full.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from radhalo.generation.sampling import generation_order, source_id

TARGETS = {1: 5, 2: 5, 3: 5, 4: 35}


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def truncate_to_targets(log: list[dict], eligible_note_ids: list[str]) -> list[dict]:
    model = log[0]["config"]["generation_model"]
    order = generation_order(eligible_note_ids, log[0]["config"]["random_seed"], model["path"], model["type"])
    position = {note_id: i for i, note_id in enumerate(order)}
    filled = {level: 0 for level in TARGETS}
    kept = []
    for item in sorted(log, key=lambda x: position[x["radiology_report"]["note_id"]]):
        filled[item["risk_level"]] += 1
        kept.append(item)
        if all(filled[lv] >= n for lv, n in TARGETS.items()):
            break
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs", type=Path, required=True, help="directory with one generation log per source"
    )
    parser.add_argument("--eligible", type=Path, required=True, help="eligible reports (JSONL with note_id)")
    parser.add_argument("--out", type=Path, default=Path("derived/risk_grade_counts.csv"))
    args = parser.parse_args()

    eligible_note_ids = [r["note_id"] for r in read_jsonl(args.eligible)]
    rows = []
    for path in sorted(args.logs.glob("*.jsonl")):
        log = read_jsonl(path)
        counts = Counter(item["risk_level"] for item in truncate_to_targets(log, eligible_note_ids))
        source = source_id(log[0]["config"]["generation_model"])
        rows += [
            {"impression_source": source, "risk_level": lv, "count": counts[lv]} for lv in sorted(counts)
        ]

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["impression_source", "risk_level", "count"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
