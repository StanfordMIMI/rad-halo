"""Add the final label of every sentence (Sec. 3.6).

`adjudicated_label` is the adjudicator's label for disputed sentences and the unanimous resident label
otherwise (an undisputed sentence without unanimous residents is an error).
"""

import argparse
from pathlib import Path

from annotated_dataset import finalize
from labelstudio import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="rad-halo_dataset_adjudicated.json")
    parser.add_argument("--out", type=Path, required=True, help="e.g. rad-halo_dataset_final.json")
    args = parser.parse_args()

    reports = read_json(args.dataset)
    finalize(reports)
    write_json(reports, args.out)
    for name, key in (("hallucinations", "impression_sentences"), ("omissions", "findings_sentences")):
        labels = [s["adjudicated_label"] for r in reports for s in r[key]]
        print(f"{name}: {sum(labels)} of {len(labels)} sentences")


if __name__ == "__main__":
    main()
