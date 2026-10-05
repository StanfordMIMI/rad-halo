"""Count synthetic training samples by error type and label (App. D.3).

Recomputes derived/synthetic_composition.csv (committed copy) from the output of
`02_create_synthetic_data.py`.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def composition(samples: list[dict]) -> dict[str, int]:
    by_type = Counter((s["error_type"], s["label"]) for s in samples)
    return {
        "total_samples": len(samples),
        "unique_reports": len({s["report_id"] for s in samples}),
        "hallucination_positive": by_type[("hallucination", 1)],
        "hallucination_negative": by_type[("hallucination", 0)],
        "omission_positive": by_type[("omission", 1)],
        "omission_negative": by_type[("omission", 0)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="synthetic samples (JSONL)")
    parser.add_argument("--out", type=Path, default=Path("outputs/synthetic_composition.csv"))
    args = parser.parse_args()

    with open(args.input) as f:
        samples = [json.loads(line) for line in f]
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(composition(samples).items())
    print(f"Composition of {len(samples):,} samples -> {args.out}")


if __name__ == "__main__":
    main()
