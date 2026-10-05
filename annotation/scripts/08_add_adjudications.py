"""Add the adjudicator's decisions to the disputed sentences (Sec. 3.6).

Inputs: `rad-halo_dataset_imaging_technique.json` and the Label Studio exports of the two adjudication
projects. The decision is read from the first annotation of each task and stored under `adjudicator`
in `is_hallucination` / `is_omission`. Tasks without a decision are skipped and listed.
"""

import argparse
from pathlib import Path

from annotated_dataset import add_adjudications
from labelstudio import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="rad-halo_dataset_imaging_technique.json")
    parser.add_argument(
        "--hallucinations", type=Path, required=True, help="export of the hallucination project"
    )
    parser.add_argument("--omissions", type=Path, required=True, help="export of the omission project")
    parser.add_argument("--out", type=Path, required=True, help="e.g. rad-halo_dataset_adjudicated.json")
    args = parser.parse_args()

    reports = read_json(args.dataset)
    for error_type, export in (("hallucination", args.hallucinations), ("omission", args.omissions)):
        integrated, skipped = add_adjudications(reports, read_json(export), error_type)
        print(
            f"{error_type}: {integrated} adjudicated, {len(skipped)} skipped (radhalo_id, sentence) {skipped}"
        )
    write_json(reports, args.out)
    print(f"{len(reports)} reports -> {args.out}")


if __name__ == "__main__":
    main()
