"""Run an LLM judge on every Rad-Halo report (Sec. 4.1), one request per report.

    python scripts/detection/run_llm_judge.py judge=gemini-3-pro_low

Writes `<out_dir>/<judge>.jsonl` in the format of `predictions/` and the raw responses to
`<out_dir>/<judge>.raw.jsonl`. Reports already in the prediction file are skipped, so an interrupted run
resumes where it stopped. Any error other than an API overload stops the run.
"""

import json
import logging
from functools import partial
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from radhalo.data.release import load_reports
from radhalo.detection.llm_judge import judge_report, load_prompt, response_schema, with_retries
from radhalo.llm import build_model

log = logging.getLogger(__name__)
JUDGES = Path(__file__).resolve().parents[2] / "configs" / "judges"


def finished_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {json.loads(line)["radhalo_id"] for line in path.read_text().splitlines() if line.strip()}


def write_raw(file, rid: int, response: str) -> None:
    file.write(json.dumps({"radhalo_id": rid, "response": response}) + "\n")
    file.flush()


@hydra.main(config_path="../../configs", config_name="detect", version_base=None)
def main(cfg: DictConfig) -> None:
    key = cfg.judge
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path, raw_path = out_dir / f"{key}.jsonl", out_dir / f"{key}.raw.jsonl"

    done = finished_ids(predictions_path)
    reports = [r for r in load_reports(Path(cfg.data)) if r["radhalo_id"] not in done]
    log.info("%s: %d reports to judge, %d already done", key, len(reports), len(done))

    judge = OmegaConf.to_container(OmegaConf.load(JUDGES / f"{key}.yaml"))
    predict = build_model(judge, response_schema(judge["type"]))
    prompt = load_prompt(Path(cfg.prompt))

    with open(predictions_path, "a") as predictions_file, open(raw_path, "a") as raw_file:
        for i, report in enumerate(reports, 1):
            rid = report["radhalo_id"]
            on_response = partial(write_raw, raw_file, rid)
            scores = with_retries(partial(judge_report, report, prompt, predict, on_response))
            predictions_file.write(json.dumps({"radhalo_id": rid, **scores}) + "\n")
            predictions_file.flush()
            log.info("radhalo_id %d done (%d/%d)", rid, i, len(reports))
    log.info("Predictions -> %s", predictions_path)


if __name__ == "__main__":
    load_dotenv()
    main()
