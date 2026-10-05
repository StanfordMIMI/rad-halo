"""Impression generation prompts (Sec. 3.3, App. A.4-A.6).

The user prompt holds the instruction, three few-shot examples and the target report. The examples are
MIMIC-IV-Note reports and are not part of this repository: `prompts/generation_*.yaml` holds placeholders
such as `{example_1_report}`, filled from the reader's eligible-report file with the note ids in
`derived/few_shot_note_ids.csv`.
"""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml

from radhalo.data.mimic import extract_impression
from radhalo.data.segmentation import clean_whitespace, remove_impression_section, split_numbered_list
from radhalo.llm import Conversation

EXAMPLE_PARTS = ("report", "impression")
# The paper prompts differ from the rendered reports by these insertions (offset into the rendered text,
# inserted characters): whitespace and one full stop, no words.
EXAMPLE_EDITS = {
    (1, "report"): [(331, " ")],
    (3, "report"): [(1128, " "), (1474, " ")],
    (3, "impression"): [(238, ".")],
}


def report_context(text: str) -> str:
    """Report text before "IMPRESSION:", whitespace-cleaned: the model input and the MedVAL reference."""
    return clean_whitespace(remove_impression_section(text))


def numbered_impression(impression: str) -> str:
    """One numbered item per line; an unnumbered impression becomes item 1."""
    items = [item.strip() for item in split_numbered_list(clean_whitespace(impression)) if item.strip()]
    if len(items) == 1 and not items[0].startswith("1. "):
        items[0] = f"1. {items[0]}"
    return "\n".join(items)


def insert_edits(text: str, edits: Iterable[tuple[int, str]]) -> str:
    for offset, inserted in sorted(edits, reverse=True):
        text = text[:offset] + inserted + text[offset:]
    return text


def example_blocks(number: int, text: str) -> dict[str, str]:
    """Report and impression of few-shot example `number`, rendered from its MIMIC report text."""
    blocks = {"report": report_context(text), "impression": numbered_impression(extract_impression(text))}
    return {part: insert_edits(blocks[part], EXAMPLE_EDITS.get((number, part), [])) for part in blocks}


def find_rows(eligible: Path, note_ids: Iterable[str]) -> dict[str, dict]:
    """Rows of the eligible-report JSONL with the given note ids."""
    wanted, rows = set(note_ids), {}
    with open(eligible) as f:
        for line in f:
            if any(note_id in line for note_id in wanted):
                row = json.loads(line)
                if row["note_id"] in wanted:
                    rows[row["note_id"]] = row
    missing = wanted - set(rows)
    if missing:
        raise ValueError(f"{len(missing)} few-shot reports not in {eligible}")
    return rows


def mimic_examples(note_ids: Mapping[int, str], eligible: Path) -> dict[int, dict[str, str]]:
    """Render the few-shot examples ({example number: note id}) from the eligible-report JSONL."""
    rows = find_rows(eligible, note_ids.values())
    return {number: example_blocks(number, rows[note_id]["text"]) for number, note_id in note_ids.items()}


def fill_examples(user_template: str, examples: Mapping[int, Mapping[str, str]]) -> str:
    for number, blocks in examples.items():
        for part in EXAMPLE_PARTS:
            user_template = user_template.replace(f"{{example_{number}_{part}}}", blocks[part])
    if "{example_" in user_template:
        raise ValueError("generation prompt has unfilled few-shot examples")
    return user_template


def load_prompt(path: Path, examples: Mapping[int, Mapping[str, str]]) -> tuple[str, str]:
    """System prompt and user template (with `{full_report_text}`)."""
    prompt = yaml.safe_load(Path(path).read_text())
    return prompt["system"], fill_examples(prompt["user"], examples)


def build_messages(rows: Iterable[dict], system_prompt: str, user_template: str) -> list[Conversation]:
    """One [system, user] conversation per MIMIC row (key `text`)."""
    return [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template.format(full_report_text=report_context(row["text"]))},
        ]
        for row in rows
    ]
