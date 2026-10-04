"""Score every Rad-Halo sentence with RadCheck (Sec. 4.2). Needs a GPU and the RadCheck checkpoint.

Writes one line per report in the format of `predictions/` (default `outputs/detection/radcheck-8b.jsonl`).
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from radhalo.data.release import load_reports
from radhalo.detection.radcheck import load_classifier, score_report

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="directory with the LoRA adapter")
    parser.add_argument(
        "--data",
        type=Path,
        default=os.environ.get("RADHALO_DATA", REPO / "data" / "rad-halo"),
        help="directory with the PhysioNet release (rad_halo_dataset.json)",
    )
    parser.add_argument("--out", type=Path, default=REPO / "outputs" / "detection" / "radcheck-8b.jsonl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--template", type=Path, default=REPO / "prompts" / "radcheck.txt")
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    template = args.template.read_text()
    tokenizer, model = load_classifier(args.checkpoint, device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for report in tqdm(load_reports(args.data), desc="RadCheck"):
            scores = score_report(report, template, tokenizer, model, args.batch_size, device)
            f.write(json.dumps({"radhalo_id": report["radhalo_id"], **scores}) + "\n")
            f.flush()
    print(f"Predictions -> {args.out}")


if __name__ == "__main__":
    main()
