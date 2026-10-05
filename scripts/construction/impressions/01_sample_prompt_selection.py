"""Sample the 100 eligible reports of the prompt selection experiment (App. A.4).

`random.seed(seed)` + `random.sample` over the eligible reports, as in the paper run; writes the sampled
MIMIC rows as JSONL in sample order.
"""

import argparse
import json
import random
from pathlib import Path


def sample_indices(n_rows: int, n: int, seed: int) -> list[int]:
    """Indices chosen by `random.sample(rows, n)` after `random.seed(seed)` (depends only on len(rows))."""
    random.seed(seed)
    return random.sample(range(n_rows), min(n, n_rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligible", type=Path, default=Path("data/mimic_eligible.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/prompt_selection/sample.jsonl"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.eligible) as f:
        n_rows = sum(1 for _ in f)
    indices = sample_indices(n_rows, args.n, args.seed)
    wanted = set(indices)
    lines = {}
    with open(args.eligible) as f:
        for i, line in enumerate(f):
            if i in wanted:
                lines[i] = line
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for i in indices:
            f.write(json.dumps(json.loads(lines[i])) + "\n")
    print(f"{len(indices)} of {n_rows:,} reports -> {args.out}")


if __name__ == "__main__":
    main()
