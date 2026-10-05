"""Sentence counts of reference and generated impressions per prompt and model (App. A.4).

Reads every `*.jsonl` of `02_generate_prompt_selection.py` (sorted by file name) and writes one row per
report and prompt: `report_id, prompt, num_sentences_original, num_sentences_<model>...`.
The reference count uses the text after the *last* "IMPRESSION:" of the report, as in the paper run.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from radhalo.data.segmentation import split_impression


def load_generations(directory: Path) -> list[dict]:
    items = []
    for path in sorted(directory.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                items.append(
                    {
                        "model": d["config"]["generation_model"]["path"],
                        "prompt": d["prompt"],
                        "radiology_report": d["radiology_report"],
                        "generated_impression": d["generated_impression"],
                    }
                )
    return items


def report_id(report: dict) -> str:
    return report.get("note_id") or report.get("report_id")


def reference_sentence_count(text: str) -> int:
    if "IMPRESSION:" not in text:
        return 0
    impression = text.split("IMPRESSION:")[-1].strip()
    return len(split_impression(impression)) if impression else 0


def sentence_counts(items: list[dict]) -> pd.DataFrame:
    """One row per report (first-seen order) and prompt; one count column per model."""
    first = {}
    for item in items:
        first.setdefault(report_id(item["radiology_report"]), item)
    prompts = list(dict.fromkeys(item["prompt"] for item in items))
    models = list(dict.fromkeys(item["model"] for item in items))
    generated = {}
    for item in items:
        generated.setdefault((report_id(item["radiology_report"]), item["prompt"], item["model"]), item)

    rows = []
    for rid, item in first.items():
        original = reference_sentence_count(item["radiology_report"]["text"])
        for prompt in prompts:
            row = {"report_id": rid, "prompt": Path(prompt).stem, "num_sentences_original": original}
            for model in models:
                if (rid, prompt, model) in generated:
                    impression = generated[(rid, prompt, model)]["generated_impression"]
                    row[f"num_sentences_{model.split('/')[-1]}"] = len(split_impression(impression))
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, default=Path("data/prompt_selection/generated"))
    parser.add_argument(
        "--out", type=Path, default=Path("data/prompt_selection/prompt_selection_analysis.csv")
    )
    args = parser.parse_args()

    df = sentence_counts(load_generations(args.generated))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    original = df["num_sentences_original"]
    for prompt, group in df.groupby("prompt", sort=False):
        models = [c for c in df.columns if c.startswith("num_sentences_") and c != "num_sentences_original"]
        diffs = {c.removeprefix("num_sentences_"): (group[c] - original[group.index]).mean() for c in models}
        print(prompt, {m: round(d, 2) for m, d in diffs.items()})
    print(f"{len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
