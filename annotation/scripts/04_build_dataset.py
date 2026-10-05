"""Build the resident-annotated dataset from the Label Studio exports (Sec. 3.5).

Residents could adjust sentence boundaries. A coordinator (Label Studio user `anchor`) fixed one sentence
segmentation per report and copied each resident annotation onto it. This script reads
- `--anchor`: exports with the anchor segmentation (the anchor user's annotation of each task),
- `--exports`: one subdirectory per resident (`annotator_1` .. `annotator_4`) with that resident's exports,
- `--processed`: `processed_reports.jsonl` of `01_prepare_tasks.py` (note id, risk level, source),
- `--user-map`: Label Studio user ids of the anchor user and the residents (see `user_map.example.yaml`).
Labels are read from the anchor user's copy where present, else from the resident's own annotation; every
sentence of a report the resident annotated is negative unless labelled HALLUCINATION / OMISSION.
Nothing is written if a labelled span is not an anchor sentence or still "To Review".
"""

import argparse
import json
import sys
from pathlib import Path

from annotated_dataset import add_resident_labels, build_reports, validation_problems
from labelstudio import ANNOTATORS, read_exports, read_user_map, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True, help="directory with the anchor exports")
    parser.add_argument(
        "--exports", type=Path, required=True, help="directory with one subdirectory per resident"
    )
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--user-map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="e.g. rad-halo_dataset.json")
    args = parser.parse_args()

    user_map = read_user_map(args.user_map)
    with open(args.processed) as f:
        processed = [json.loads(line) for line in f if line.strip()]
    reports = build_reports(read_exports(args.anchor), processed, user_map["anchor"])
    exports = {a: read_exports(args.exports / a) for a in ANNOTATORS if (args.exports / a).is_dir()}

    problems = validation_problems(exports, reports, user_map)
    if problems:
        for annotator, rid, from_name, start, end, kind in problems:
            print(f"{kind}: {annotator}, radhalo_id {rid}, {from_name} [{start}:{end}]", file=sys.stderr)
        sys.exit(f"{len(problems)} problems; fix them in Label Studio and re-export")

    add_resident_labels(exports, reports, user_map)
    write_json(reports, args.out)
    print(f"{len(reports)} reports annotated by {', '.join(exports)} -> {args.out}")


if __name__ == "__main__":
    main()
