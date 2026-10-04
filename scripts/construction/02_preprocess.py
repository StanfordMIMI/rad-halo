"""Segment eligible reports into findings and impression sentences (Appendix A.2).

Optionally holds out `--holdout` reports (sklearn `train_test_split`, shuffled with `--split-seed`), draws
`--num-rows` of the rest with `random.sample` (seeded), then drops reports whose longest findings item
exceeds 400 characters (a sign of failed sentence splitting). The output is the report pool for synthetic
data; the Rad-Halo reports were drawn from the eligible reports without this rule.

Synthetic pool of the paper: `--holdout 1000 --split-seed 0 --num-rows 10000 --seed 0`.
"""

import argparse
import json
import random
from pathlib import Path

from sklearn.model_selection import train_test_split

from radhalo.data.mimic import has_short_findings_sentences, process_report
from radhalo.data.report import write_reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="eligible reports (raw MIMIC rows, JSONL)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout", type=int, default=None, help="hold out this many reports first")
    parser.add_argument("--split-seed", type=int, default=None, help="random_state of the holdout split")
    parser.add_argument("--num-rows", type=int, default=None, help="sample this many reports")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.input) as f:
        lines = f.readlines()
    if args.holdout is not None:
        lines, _ = train_test_split(lines, test_size=args.holdout, random_state=args.split_seed, shuffle=True)
    if args.num_rows is not None:
        if args.seed is not None:
            random.seed(args.seed)
        lines = random.sample(lines, min(args.num_rows, len(lines)))

    reports = (process_report(json.loads(line)) for line in lines)
    kept = [r for r in reports if has_short_findings_sentences(r)]
    write_reports(kept, args.out)
    print(f"{len(kept):,} of {len(lines):,} reports kept -> {args.out}")


if __name__ == "__main__":
    main()
