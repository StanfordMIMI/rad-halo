"""Sentence segmentation of findings and impression sections (Appendix A.2).

Sentences are split with the scispaCy model `en_core_sci_sm`. Findings keep section headers
("LUNGS:") as separate items; impressions are split into numbered items first, then into sentences.
"""

import re
from functools import cache

from radhalo.data.report import ImpressionSentence

FINDINGS_HEADER = re.compile(r"(?:^|\n)([A-Z][A-Z\s,/-]+:)\s")
HEADER = re.compile(r"^[A-Z][A-Z\s,/-]+:$")
NUMBERED_ITEM = re.compile(r"(?<!\d)(?=\s*\d+[\.\)]\s+)")
MIN_IMPRESSION_SENTENCE_CHARS = 4


@cache
def _nlp():
    import spacy

    return spacy.load("en_core_sci_sm")


def split_sentences(text: str) -> list[str]:
    return [sent.text.strip() for sent in _nlp()(text).sents]


def clean_whitespace(text: str) -> str:
    """Replace runs of newlines with one space, then runs of spaces with one space."""
    return re.sub(r" {2,}", " ", re.sub(r"[\r\n]+", " ", text))


def remove_impression_section(text: str) -> str:
    """Text before the first "IMPRESSION:", stripped."""
    return text.split("IMPRESSION:")[0].strip()


def split_numbered_list(text: str) -> list[str]:
    """Split "1. a 2. b" into items; the first item holds any text before "1." (possibly empty).

    Numbers must count up from 1 ("1." or "1)"); out-of-sequence numbers stay inside the current item.
    Returns `[text]` unchanged if there is no candidate number at all.
    """
    parts = NUMBERED_ITEM.split(text)
    if len(parts) == 1:
        return parts
    items, current, counter = [], "", 1
    for part in parts:
        stripped = part.lstrip()
        if stripped.startswith(f"{counter}. ") or stripped.startswith(f"{counter}) "):
            items.append(current)
            current = part
            counter += 1
        else:
            current += part
    items.append(current)
    return [item.rstrip() for item in items]


def label_numbered_items(items: list[str]) -> dict[str, str]:
    """Map items of `split_numbered_list` to labels "pre", "1.", "2.", ... and drop the number prefix."""
    labelled = {}
    if not items[0].startswith("1. ") and not items[0].startswith("1) "):
        labelled["pre"] = items[0]
        items = items[1:]
    for i, item in enumerate(items):
        labelled[f"{i + 1}."] = item[2:].strip()
    return labelled


def split_findings(findings_text: str) -> list[str]:
    """Split findings into sentences; section headers become their own items."""
    sentences = []
    for part in FINDINGS_HEADER.split(findings_text):
        if part:
            sentences.extend(split_sentences(clean_whitespace(part).strip()))
    return sentences


def is_header(text: str) -> bool:
    return HEADER.match(text.strip()) is not None


def split_impression(impression_text: str) -> list[ImpressionSentence]:
    """Split an impression into labelled sentences; sentences of 3 characters or fewer are dropped."""
    items = label_numbered_items(split_numbered_list(clean_whitespace(impression_text)))
    return [
        ImpressionSentence(text=sentence, number_label=label)
        for label, item in items.items()
        for sentence in (s.strip() for s in split_sentences(item))
        if len(sentence) >= MIN_IMPRESSION_SENTENCE_CHARS
    ]


def impression_text(sentences: list[ImpressionSentence]) -> str:
    """Rebuild a numbered impression with consecutive numbers; a leading "pre" item stays unnumbered."""
    by_label: dict[str, list[str]] = {}
    for sentence in sentences:
        by_label.setdefault(sentence.number_label, []).append(sentence.text)
    lines, number = [], 1
    for index, (label, texts) in enumerate(by_label.items()):
        if index == 0 and label == "pre":
            lines.append(" ".join(texts))
        else:
            lines.append(f"{number}. {' '.join(texts)}")
            number += 1
    return "\n".join(lines)
