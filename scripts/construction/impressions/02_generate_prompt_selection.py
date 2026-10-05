"""Generate impressions for the prompt selection sample with each instruction variant (App. A.4).

    python scripts/construction/impressions/02_generate_prompt_selection.py --generator qwen3-1.7b

For every prompt, appends one record per report to `<out_dir>/<prompt stem>_<model name>.jsonl`
(model name = last part of the generator path). The few-shot examples are rendered from `--eligible`.
Run from the repository root.
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

from radhalo.generation.loop import batches
from radhalo.generation.prompts import build_messages, load_prompt, mimic_examples
from radhalo.llm import build_model
from radhalo.synthetic.attribution import example_note_ids

REPO = Path(__file__).resolve().parents[3]
PROMPTS = [Path("prompts") / f"generation_{v}.yaml" for v in ("simple", "professional", "strict")]


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--generator", required=True, help="name of a file in configs/generators/")
    parser.add_argument("--sample", type=Path, default=Path("data/prompt_selection/sample.jsonl"))
    parser.add_argument("--eligible", type=Path, default=Path("data/mimic_eligible.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/prompt_selection/generated"))
    parser.add_argument("--prompts", type=Path, nargs="+", default=PROMPTS)
    parser.add_argument("--note-ids", type=Path, default=REPO / "derived" / "few_shot_note_ids.csv")
    args = parser.parse_args()

    generator = OmegaConf.to_container(
        OmegaConf.load(REPO / "configs" / "generators" / f"{args.generator}.yaml")
    )
    reports = read_jsonl(args.sample)
    examples = mimic_examples(example_note_ids(args.note_ids, "generation"), args.eligible)
    predict = build_model(generator)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for prompt_path in args.prompts:
        system_prompt, user_template = load_prompt(prompt_path, examples)
        out = args.out_dir / f"{prompt_path.stem}_{generator['path'].split('/')[-1]}.jsonl"
        for batch in batches(reports, generator["batch_size"]):
            impressions = predict(build_messages(batch, system_prompt, user_template))
            with open(out, "a") as f:
                for impression, report in zip(impressions, batch):
                    record = {
                        "generated_impression": impression,
                        "radiology_report": report,
                        "prompt": str(prompt_path),
                        "config": {"generation_model": generator},
                    }
                    f.write(json.dumps(record) + "\n")
        print(f"{prompt_path.stem}: {len(reports)} impressions -> {out}")


if __name__ == "__main__":
    main()
