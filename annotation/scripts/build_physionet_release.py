"""Build the PhysioNet release of Rad-Halo from the final dataset.

Input: `rad-halo_dataset_final.json` of `09_finalize_dataset.py`. Writes to `--out`:
- `data/rad_halo_dataset.json`: the final reports with `report_text` renamed to
  `background_and_findings_text` and `generation_model` to `impression_source`,
- `data/reports.csv`, `data/findings_sentences.csv`, `data/impression_sentences.csv`: flattened views
  (labels as "true"/"false", empty where a rater did not label the sentence),
- `metadata/schema.json`, `metadata/data_dictionary.csv` and `code/example_usage.py`.
The release README is maintained separately.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

from annotated_dataset import REPORT_COLUMNS, SENTENCE_COLUMNS, release_reports, sentence_rows
from labelstudio import read_json
from physionet_metadata import DATA_DICTIONARY_CSV, SCHEMA_JSON

EXAMPLE_USAGE = Path(__file__).resolve().parents[1] / "physionet" / "example_usage.py"


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def build_release(final_reports: list[dict], out: Path) -> None:
    reports = release_reports(final_reports)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "rad_halo_dataset.json").write_text(
        json.dumps(reports, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    write_csv(data / "reports.csv", REPORT_COLUMNS, [[r[c] for c in REPORT_COLUMNS] for r in reports])
    write_csv(data / "findings_sentences.csv", SENTENCE_COLUMNS, sentence_rows(reports, "omission"))
    write_csv(data / "impression_sentences.csv", SENTENCE_COLUMNS, sentence_rows(reports, "hallucination"))
    metadata = out / "metadata"
    metadata.mkdir(exist_ok=True)
    (metadata / "schema.json").write_text(SCHEMA_JSON, encoding="utf-8", newline="\n")
    (metadata / "data_dictionary.csv").write_text(DATA_DICTIONARY_CSV, encoding="utf-8", newline="\n")
    (out / "code").mkdir(exist_ok=True)
    shutil.copyfile(EXAMPLE_USAGE, out / "code" / "example_usage.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="rad-halo_dataset_final.json")
    parser.add_argument("--out", type=Path, required=True, help="release directory")
    args = parser.parse_args()

    build_release(read_json(args.dataset), args.out)
    print(f"release -> {args.out}")


if __name__ == "__main__":
    main()
