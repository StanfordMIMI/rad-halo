"""Write the fictional example task of the annotation interface figure (`label_studio/example_task.json`).

The report is invented. Unlike the real tasks, every findings sentence is highlighted, and the evaluated
impression is the report's own impression. Requires the `nlp` extra (scispaCy).
"""

import argparse
from pathlib import Path

from labelstudio import find_span, make_task, write_json

from radhalo.data.segmentation import remove_impression_section, split_findings, split_impression

FULL_TEXT = (
    "HISTORY: 65-year-old male with cough and fever, evaluate for pneumonia.\n\n"
    "TECHNIQUE: PA and lateral chest radiograph.\n\n"
    "FINDINGS: The lungs are well expanded. "
    "There is patchy airspace opacity in the right lower lobe. "
    "A 1.5 cm pulmonary nodule is present in the left upper lobe. "
    "The cardiac silhouette is normal in size. "
    "The mediastinal contours are unremarkable. "
    "No pleural effusion or pneumothorax is identified. "
    "The visualized osseous structures are intact.\n\n"
    "IMPRESSION:\n"
    "1. Right lower lobe opacity consistent with pneumonia.\n"
    "2. Mild cardiomegaly.\n"
    "3. No pleural effusion or pneumothorax."
)
FINDINGS_TEXT = FULL_TEXT.split("FINDINGS: ")[1].split("\n\n")[0]
IMPRESSION_TEXT = FULL_TEXT.split("IMPRESSION:\n")[1]


def example_task() -> dict:
    findings_text = remove_impression_section(FULL_TEXT)
    findings = [find_span(findings_text, s) for s in split_findings(FINDINGS_TEXT)]
    impression = [find_span(IMPRESSION_TEXT, s.text) for s in split_impression(IMPRESSION_TEXT)]
    task = make_task(
        findings_text,
        IMPRESSION_TEXT,
        [s for s in findings if s is not None],
        [s for s in impression if s is not None],
        model_version="example_figure",
    )
    task["data"]["generated_impression_id"] = 0
    return task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parents[1] / "label_studio" / "example_task.json"
    )
    args = parser.parse_args()
    write_json([example_task()], args.out, ensure_ascii=False)
    print(f"example task -> {args.out}")


if __name__ == "__main__":
    main()
