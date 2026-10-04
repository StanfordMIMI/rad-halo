"""Create the synthetic RadCheck training set from attributed reports (Sec. 4.3, App. D.2).

Report i of the input gets base seed `--seed + i`; see `radhalo.synthetic.samples`. Writes one sample per
line with keys `report_id, features, label, error_type` (35,151 samples from 9,938 reports in the paper).
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from radhalo.data.report import read_reports
from radhalo.synthetic.samples import report_samples

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="attributed reports (JSONL from 03_attribute.py)"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=REPO / "prompts" / "radcheck.txt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    template = args.template.read_text()
    reports = list(read_reports(args.input))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_samples = 0
    with open(args.out, "w") as f:
        for i, report in enumerate(tqdm(reports, desc="Synthetic samples")):
            for sample in report_samples(report, args.seed + i, template):
                f.write(json.dumps(sample) + "\n")
                n_samples += 1
    print(f"{n_samples:,} samples from {len(reports):,} reports -> {args.out}")


if __name__ == "__main__":
    main()
