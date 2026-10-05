"""Turn the physician-reviewed pre-highlights into predictions for the resident annotation projects.

Input: Label Studio export of the pre-highlight review project (tasks of `01_prepare_tasks.py`, reviewed
with `label_studio/prehighlight_review.xml`). The first annotation of each task becomes its prediction
(model version "v2_preannotated"), so residents start from the reviewed "To Review" highlights.
"""

import argparse
from pathlib import Path

from labelstudio import annotations_to_predictions, read_json, write_json

MODEL_VERSION = "v2_preannotated"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export", type=Path, required=True, help="Label Studio export of the review project"
    )
    parser.add_argument("--out", type=Path, required=True, help="tasks for the resident annotation projects")
    args = parser.parse_args()

    tasks = annotations_to_predictions(read_json(args.export), MODEL_VERSION)
    write_json(tasks, args.out, ensure_ascii=False)
    print(f"{len(tasks)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
