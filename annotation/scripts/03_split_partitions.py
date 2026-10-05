"""Split the annotation tasks into consecutive partitions (App. B).

Partitions 1-3 (50/125/125 tasks, ids 0-49, 50-174, 175-299) were created with this script.
Partition 1 went to annotator_1 and annotator_2, partition 2 to annotator_1, partition 3 to annotator_2.
The second annotators of ids 50-299 (annotator_3, annotator_4) and the 21 reports annotated by
annotator_4 in addition were assigned outside this code; `label_studio/assignments.csv` lists who
annotated each report.
"""

import argparse
from pathlib import Path

from labelstudio import read_json, write_json

SIZES = (50, 125, 125)


def split(tasks: list, sizes: tuple[int, ...] = SIZES) -> list[list]:
    """Consecutive chunks of the given sizes; the sizes must add up to the number of tasks."""
    if sum(sizes) != len(tasks):
        raise ValueError(f"partition sizes sum to {sum(sizes)}, but there are {len(tasks)} tasks")
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    return [tasks[start : start + size] for start, size in zip(starts, sizes)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", type=Path, required=True, help="output of 02_import_prehighlight_review.py"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    args = parser.parse_args()

    for number, part in enumerate(split(read_json(args.tasks), tuple(args.sizes)), start=1):
        out = args.out_dir / f"{args.tasks.stem}_part{number}_{len(part)}.json"
        write_json(part, out, ensure_ascii=False)
        print(f"{len(part)} tasks -> {out}")


if __name__ == "__main__":
    main()
