import hashlib
import importlib.util
import json
import random
import sys
import types
from pathlib import Path

import pytest

from radhalo.data.report import ImpressionSentence
from radhalo.generation import medval
from radhalo.generation.loop import (
    drop_unsupported_notes,
    generate_and_grade,
    num_batches,
    records,
    write_records,
)
from radhalo.generation.prompts import (
    build_messages,
    example_blocks,
    fill_examples,
    insert_edits,
    load_prompt,
    numbered_impression,
    report_context,
)

REPO = Path(__file__).resolve().parents[1]
needs_scispacy = pytest.mark.skipif(
    importlib.util.find_spec("en_core_sci_sm") is None, reason="scispaCy model en_core_sci_sm not installed"
)

# Fictional report.
TEXT = (
    "EXAMINATION: Chest radiograph.\n\nFINDINGS: \n\nThe lungs are clear.\n  No effusion.\n\n"
    "IMPRESSION: \n\nNo acute process.\n"
)
EXAMPLES = {
    n: {"report": f"fictional report {n}", "impression": f"1. Fictional impression {n}."} for n in (1, 2, 3)
}


def test_report_context():
    assert (
        report_context(TEXT) == "EXAMINATION: Chest radiograph. FINDINGS: The lungs are clear. No effusion."
    )


def test_numbered_impression():
    assert numbered_impression("No acute process.") == "1. No acute process."
    assert numbered_impression("1. Effusion.\n2. Small\nnodule.") == "1. Effusion.\n2. Small nodule."
    assert numbered_impression("1. Only item.") == "1. Only item."


def test_insert_edits_use_original_offsets():
    assert insert_edits("abcdef", [(2, "X"), (4, "Y")]) == "abXcdYef"
    assert insert_edits("abc", []) == "abc"


def test_example_blocks():
    # Example 2 has no edits.
    assert example_blocks(2, TEXT) == {
        "report": "EXAMINATION: Chest radiograph. FINDINGS: The lungs are clear. No effusion.",
        "impression": "1. No acute process.",
    }


@pytest.mark.parametrize("variant", ["simple", "professional", "strict"])
def test_load_prompt(variant):
    system, user = load_prompt(REPO / "prompts" / f"generation_{variant}.yaml", EXAMPLES)
    assert system == "You are an expert medical professional.\n"
    assert "{example_" not in user and user.count("{full_report_text}") == 1
    assert all(f"fictional report {n}" in user for n in (1, 2, 3))
    assert user.endswith("[REPORT]\n{full_report_text}\n\n[IMPRESSION]")


def test_fill_examples_requires_all():
    with pytest.raises(ValueError):
        fill_examples("{example_1_report} {example_2_report}", {1: EXAMPLES[1]})


def test_build_messages():
    (messages,) = build_messages([{"text": TEXT}], "S", "Report: {full_report_text}")
    assert messages == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Report: " + report_context(TEXT)},
    ]


def rows(n: int) -> list[dict]:
    return [
        {"note_id": f"fictional-{i}", "text": f"FINDINGS: finding {i}. IMPRESSION: impression {i}."}
        for i in range(n)
    ]


def fake_grade(levels: dict):
    """Grade by the report number in the reference; level None raises."""

    def grade(reference: str, candidate: str) -> dict:
        level = levels[int(reference.split("finding ")[1].rstrip("."))]
        if level is None:
            raise RuntimeError("fictional MedVAL failure")
        return {"reason": "r", "err": "e", "attack_prediction": level}

    return grade


def echo(batch: list[dict]) -> list[str]:
    return [f"generated {row['note_id']}" for row in batch]


def test_num_batches():
    assert num_batches(100, 1, 2000) == 100
    assert num_batches(10**6, 20, 2000) == 101


def test_loop_stops_when_targets_met():
    levels = {i: [1, 2, 3, 4][i % 4] for i in range(100)}
    levels[5] = None  # dropped
    levels[6] = 7  # invalid level: dropped
    seen = []
    buckets = generate_and_grade(
        rows(100), 2, echo, fake_grade(levels), {1: 2, 2: 2, 3: 2, 4: 2}, 2000, on_batch=seen.append
    )
    assert [len(b) for b in buckets.values()] == [3, 2, 2, 3]
    assert len(seen) == 6  # 12 reports in batches of 2; reports 5 and 6 dropped
    out = list(records(buckets, {"k": 1}))
    assert [r["risk_level"] for r in out] == [1, 1, 1, 2, 2, 3, 3, 4, 4, 4]
    assert list(out[0]) == [
        "risk_level",
        "config",
        "radiology_report",
        "generated_impression",
        "medval_output",
    ]
    assert out[0]["medval_output"] == {"reason": "r", "err": "e", "attack_prediction": 1}


def test_loop_skips_last_batch_of_the_cap():
    calls = []

    def impressions(batch):
        calls.append(len(batch))
        return echo(batch)

    generate_and_grade(rows(100), 1, impressions, fake_grade(dict.fromkeys(range(100), 1)), max_generations=3)
    assert calls == [1, 1, 1]  # num_batches = 4, the 4th batch is never processed


def test_write_records(tmp_path):
    buckets = generate_and_grade(rows(4), 1, echo, fake_grade(dict.fromkeys(range(4), 4)), max_generations=10)
    write_records(buckets, {"generation_model": {"type": "none"}}, tmp_path / "log.jsonl")
    lines = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    # 4 reports, batch size 1: num_batches = 4, so only 3 are processed.
    assert [line["generated_impression"] for line in lines] == [f"generated fictional-{i}" for i in range(3)]
    assert lines[0]["config"] == {"generation_model": {"type": "none"}}


def test_drop_unsupported_notes():
    sentences = [
        ImpressionSentence("Fact.", "1.", False, False, []),
        ImpressionSentence("Follow up.", "2.", True, False, []),
        ImpressionSentence("Follow up nodule.", "3.", True, False, ["Nodule."]),
        ImpressionSentence("Discussed.", "4.", False, True, []),
        ImpressionSentence("Unattributed.", "5.", None, None, []),
    ]
    assert [s.text for s in drop_unsupported_notes(sentences)] == [
        "Fact.",
        "Follow up nodule.",
        "Unattributed.",
    ]


def test_medval_prompt_digests():
    """Guards the vendored MedVAL text (trailing spaces included) against accidental edits."""
    texts = [
        medval.VALIDATOR_INSTRUCTIONS,
        medval.errors_prompt,
        medval.risk_levels_prompt,
        medval.TASK_PROMPTS["report2impression"],
    ]
    digests = [hashlib.sha256(t.encode()).hexdigest()[:12] for t in texts]
    assert digests == ["5a2d11dbb275", "3bb09d0cf4fc", "fee0c1a0a100", "0d28156796f7"]


def test_medval_grader_with_fake_dspy(monkeypatch):
    calls = []
    dspy = types.ModuleType("dspy")
    dspy.Signature = type("Signature", (), {})
    dspy.InputField = lambda **kw: ("in", kw)
    dspy.OutputField = lambda **kw: ("out", kw)
    dspy.LM = lambda **kw: calls.append(("LM", kw)) or "lm"
    dspy.configure = lambda **kw: calls.append(("configure", kw))

    class ChainOfThought:
        def __init__(self, signature):
            self.signature = signature

        def deepcopy(self):
            return self

        def __call__(self, **kw):
            calls.append(("call", kw))
            return {"reasoning": "fictional", "errors": "None", "risk_level": 2}

    dspy.ChainOfThought = ChainOfThought
    monkeypatch.setitem(sys.modules, "dspy", dspy)
    medval.validator_signature.cache_clear()
    grade = medval.build_grader("project", "location")
    assert grade("reference", "candidate") == {"reason": "fictional", "err": "None", "attack_prediction": 2}
    medval.validator_signature.cache_clear()
    assert calls[0] == (
        "LM",
        {
            "model": "gemini-2.5-flash",
            "temperature": 0.0,
            "max_tokens": 8192,
            "vertex_project": "project",
            "vertex_location": "location",
        },
    )
    assert calls[1] == ("configure", {"lm": "lm"})
    assert calls[2][1]["instruction"].startswith("Summarize the radiology report findings")


def load_script(name: str):
    path = REPO / "scripts" / "construction" / "impressions" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sample_indices_match_random_sample():
    sample_indices = load_script("01_sample_prompt_selection.py").sample_indices
    population = [f"fictional-{i}" for i in range(500)]
    random.seed(0)
    expected = random.sample(population, 20)
    assert [population[i] for i in sample_indices(len(population), 20, 0)] == expected


@needs_scispacy
def test_prompt_selection_counts():
    analysis = load_script("03_analyze_prompt_selection.py")
    report = {
        "note_id": "fictional-1",
        "text": "FINDINGS: Clear. IMPRESSION: 1. Normal. NOTE: Called. IMPRESSION: Done.",
    }
    # The reference count uses the text after the last "IMPRESSION:".
    assert analysis.reference_sentence_count(report["text"]) == 1
    items = [
        {
            "model": "org/model-a",
            "prompt": "prompts/generation_strict.yaml",
            "radiology_report": report,
            "generated_impression": "1. Normal lungs.\n2. No effusion.",
        },
    ]
    df = analysis.sentence_counts(items)
    assert df.to_dict("records") == [
        {
            "report_id": "fictional-1",
            "prompt": "generation_strict",
            "num_sentences_original": 1,
            "num_sentences_model-a": 2,
        }
    ]
