"""Segment eligible reports into findings and impression sentences (Appendix A.2).

Optionally draws `--num-rows` reports with `random.sample` (seeded), then drops reports whose longest
findings item exceeds 400 characters (a sign of failed sentence splitting). The output is the report
pool for synthetic data; the Rad-Halo reports were drawn from the eligible reports without this rule.
"""

import argparse
import json
import random
from pathlib import Path

from radhalo.data.mimic import has_short_findings_sentences, process_report
from radhalo.data.report import write_reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="eligible reports (raw MIMIC rows, JSONL)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-rows", type=int, default=None, help="sample this many reports")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.input) as f:
        lines = f.readlines()
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
