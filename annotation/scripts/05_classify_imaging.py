"""Pre-fill imaging modality and anatomical region of each report with Claude Sonnet 4.5 (Sec. 3.2).

Input: `rad-halo_dataset.json` of `04_build_dataset.py`. One forced tool call per report with the report
text (ANTHROPIC_API_KEY); labels are restricted to the fixed lists below. Writes an Excel sheet
(`radhalo_id, report_text, imaging_technique, anatomy_location`) that a physician reviewed and corrected;
`06_merge_imaging.py` merges the reviewed sheet. Requires the `annotation` and `api` extras.
"""

import argparse
import os
import time
from collections.abc import Callable
from pathlib import Path

import yaml
from dotenv import load_dotenv
from labelstudio import read_json

REPO = Path(__file__).resolve().parents[2]
MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 256
RATE_LIMIT_WAIT_SECONDS = 30
UNKNOWN = "Unknown"
IMAGING_TECHNIQUES = [
    "X-ray",
    "CT",
    "MRI",
    "Ultrasound",
    "Mammography",
    "Fluoroscopy",
    "Nuclear medicine",
    UNKNOWN,
]
ANATOMY_LOCATIONS = [
    "Head", "Face", "Neck", "Cervical spine", "Thoracic spine", "Lumbar spine", "Spine", "Chest", "Breast",
    "Abdomen", "Pelvis", "Abdomen/Pelvis", "Shoulder", "Elbow", "Wrist/Hand", "Hip", "Knee", "Ankle/Foot",
    "Upper extremity", "Lower extremity", "Head and neck", UNKNOWN,
]  # fmt: skip
CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "imaging_technique": {
            "type": "string",
            "enum": IMAGING_TECHNIQUES,
            "description": "The primary imaging modality used in this report.",
        },
        "anatomy_location": {
            "type": "string",
            "enum": ANATOMY_LOCATIONS,
            "description": "The primary anatomical region examined in this report.",
        },
    },
    "required": ["imaging_technique", "anatomy_location"],
}
TOOL = {
    "name": "classify_report",
    "description": "Classify the radiology report by imaging technique and anatomy location.",
    "input_schema": CLASSIFICATION_SCHEMA,
}


def classification_request(prompt: dict, report_text: str) -> dict:
    """Keyword arguments of `client.messages.create` for one report."""
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": prompt["system"],
        "messages": [{"role": "user", "content": prompt["user"].format(report_text=report_text)}],
        "tools": [TOOL],
        "tool_choice": {"type": "tool", "name": TOOL["name"]},
    }


def classify(
    client, prompt: dict, reports: list[dict], sleep: Callable[[float], None] = time.sleep
) -> list[dict]:
    """One row per report. A rate-limited request is retried once after 30 s (a second failure is raised);
    other errors mark the report Unknown."""
    import anthropic

    rows = []
    for report in reports:
        request = classification_request(prompt, report["report_text"])
        try:
            labels = client.messages.create(**request).content[0].input
        except anthropic.RateLimitError:
            sleep(RATE_LIMIT_WAIT_SECONDS)
            labels = client.messages.create(**request).content[0].input
        except Exception as error:
            print(f"radhalo_id {report['radhalo_id']}: classification failed ({type(error).__name__})")
            labels = {"imaging_technique": UNKNOWN, "anatomy_location": UNKNOWN}
        rows.append(
            {
                "radhalo_id": report["radhalo_id"],
                "report_text": report["report_text"],
                "imaging_technique": labels["imaging_technique"],
                "anatomy_location": labels["anatomy_location"],
            }
        )
    return rows


def main() -> None:
    load_dotenv(override=True)  # the project .env wins over the shell environment
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="rad-halo_dataset.json")
    parser.add_argument("--out", type=Path, required=True, help="Excel sheet for physician review")
    parser.add_argument(
        "--prompt", type=Path, default=REPO / "annotation" / "prompts" / "classify_imaging_technique.yaml"
    )
    args = parser.parse_args()

    import anthropic
    import pandas as pd

    prompt = yaml.safe_load(args.prompt.read_text())
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    rows = classify(client, prompt, read_json(args.dataset))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(args.out, index=False, engine="openpyxl")
    print(f"{len(rows)} reports -> {args.out}")


if __name__ == "__main__":
    main()
