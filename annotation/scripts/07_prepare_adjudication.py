"""Write adjudication tasks for all sentences on which the residents disagreed (Sec. 3.6).

Input: `rad-halo_dataset_imaging_technique.json`. One task per disputed sentence, with the sentence
highlighted as "To Review" and the residents' labels shown under their anonymised ids
("Annotator 1: Hallucination | Annotator 3: Not Hallucination"). Import into two Label Studio projects
configured with `label_studio/labeling_config_{hallucinations,omissions}.xml`.
"""

import argparse
from pathlib import Path

from annotated_dataset import adjudication_tasks
from labelstudio import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="rad-halo_dataset_imaging_technique.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    reports = read_json(args.dataset)
    for error_type in ("hallucination", "omission"):
        tasks = adjudication_tasks(reports, error_type)
        out = args.out_dir / f"{error_type}_disagreements_for_labelstudio.json"
        write_json(tasks, out)
        print(f"{len(tasks)} {error_type} disagreements -> {out}")


if __name__ == "__main__":
    main()
