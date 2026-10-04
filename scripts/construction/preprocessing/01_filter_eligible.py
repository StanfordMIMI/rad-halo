"""Select eligible MIMIC-IV-Note radiology reports (Section 3.2).

Keeps reports without "procedure" (any case) that have non-empty FINDINGS and IMPRESSION sections,
and writes the raw MIMIC rows in file order as JSONL (1,419,529 reports for MIMIC-IV-Note v2.2).
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from radhalo.data.mimic import eligible_chunks


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mimic-note-dir",
        type=Path,
        default=os.environ.get("MIMIC_NOTE_DIR"),
        help="MIMIC-IV-Note directory containing note/radiology.csv (default: $MIMIC_NOTE_DIR)",
    )
    parser.add_argument("--out", type=Path, default=Path("data/mimic_eligible.jsonl"))
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()
    if args.mimic_note_dir is None:
        parser.error("set --mimic-note-dir or MIMIC_NOTE_DIR")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_eligible = 0
    for i, chunk in enumerate(
        eligible_chunks(Path(args.mimic_note_dir) / "note" / "radiology.csv", args.chunk_size)
    ):
        chunk.to_json(args.out, orient="records", lines=True, mode="w" if i == 0 else "a")
        n_eligible += len(chunk)
    print(f"{n_eligible:,} eligible reports -> {args.out}")


if __name__ == "__main__":
    main()
