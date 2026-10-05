"""Generate-and-grade loop (Sec. 3.3-3.4, App. A.7).

Reports are taken in seeded order (`sampling.generation_order`) in batches; each impression is graded by
MedVAL against the report text before "IMPRESSION:" and kept in the bucket of its risk level until every
level has reached its target count.
"""

import json
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path

from radhalo.data.mimic import process_report
from radhalo.data.report import ImpressionSentence
from radhalo.data.segmentation import impression_text
from radhalo.generation.prompts import report_context
from radhalo.synthetic.attribution import attribute

log = logging.getLogger(__name__)

TARGETS = {1: 5, 2: 5, 3: 5, 4: 35}
RISK_LEVELS = (1, 2, 3, 4)
Buckets = dict[int, list[dict]]


def batches(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def num_batches(n_reports: int, batch_size: int, max_generations: int) -> int:
    return min(n_reports // batch_size, max_generations // batch_size + 1)


def grade_batch(
    rows: list[dict], impressions: list[str], grade: Callable[[str, str], dict], buckets: Buckets
) -> None:
    """Grade each impression into its risk-level bucket; failed gradings are logged and dropped."""
    for row, impression in zip(rows, impressions):
        try:
            output = grade(report_context(row["text"]), impression)
            buckets[output["attack_prediction"]].append(
                {
                    "radiology_report": row,
                    "generated_impression": impression,
                    "medval_output": {
                        "reason": output["reason"],
                        "err": output["err"],
                        "attack_prediction": output["attack_prediction"],
                    },
                }
            )
        except Exception as e:
            log.warning("Failed to process report: %s", e, exc_info=True)


def generate_and_grade(
    rows: list[dict],
    batch_size: int,
    impressions: Callable[[list[dict]], list[str]],
    grade: Callable[[str, str], dict],
    targets: Mapping[int, int] = TARGETS,
    max_generations: int = 2000,
    on_batch: Callable[[Buckets], None] | None = None,
) -> Buckets:
    """Run the loop over `rows` (already in generation order) and return the risk-level buckets.

    `impressions(batch)` returns one impression per row; `on_batch(buckets)` runs after every batch.
    """
    buckets: Buckets = {level: [] for level in RISK_LEVELS}
    n_batches = num_batches(len(rows), batch_size, max_generations)
    for index, batch in enumerate(batches(rows, batch_size)):
        # As in the paper runs: the last of the n_batches batches is never processed.
        if index + 1 >= n_batches:
            break
        grade_batch(batch, impressions(batch), grade, buckets)
        if on_batch is not None:
            on_batch(buckets)
        counts = {level: len(buckets[level]) for level in RISK_LEVELS}
        log.info("Batch %d/%d: risk-level counts %s", index + 1, n_batches, counts)
        if all(counts[level] >= targets[level] for level in RISK_LEVELS):
            break
    return buckets


def records(buckets: Buckets, config: dict) -> Iterator[dict]:
    """Log records grouped by risk level, in generation order within a level."""
    for level, samples in buckets.items():
        for sample in samples:
            yield {"risk_level": level, "config": config, **sample}


def write_records(buckets: Buckets, config: dict, path: Path) -> None:
    with open(path, "w") as f:
        for record in records(buckets, config):
            f.write(json.dumps(record) + "\n")


def drop_unsupported_notes(sentences: Iterable[ImpressionSentence]) -> list[ImpressionSentence]:
    """Remove recommendation and administrative sentences without attributed findings."""
    return [
        s
        for s in sentences
        if not ((s.is_recommendation or s.is_admin_note) and len(s.attributed_findings_spans) == 0)
    ]


def reference_impression(row: dict, user_template: str, generate: Callable[[str], str]) -> str:
    """The MIMIC reference impression after attribution and removal of unsupported notes, renumbered."""
    report = process_report(row)
    attribute(report, user_template, generate)
    return impression_text(drop_unsupported_notes(report.impression_sentences))
