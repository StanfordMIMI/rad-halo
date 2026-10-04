import importlib.util
import json
from pathlib import Path

import pytest

from radhalo.data.release import load_predictions
from radhalo.detection import llm_judge, radcheck
from radhalo.detectors import DETECTORS

REPO = Path(__file__).resolve().parents[1]
needs_scispacy = pytest.mark.skipif(
    importlib.util.find_spec("en_core_sci_sm") is None, reason="scispaCy model en_core_sci_sm not installed"
)

# Fictional report in the release format.
REPORT = {
    "radhalo_id": 0,
    "background_and_findings_text": (
        "EXAMINATION: Chest radiograph.\n\nINDICATION:  Cough.\n\n"
        "FINDINGS: \n\nThe lungs are clear.\nNo pleural  effusion.\nThe lungs are clear."
    ),
    "impression_text": "1. No acute process.\n2. Heart size is normal.",
    "findings_sentences": [
        {"text": "The lungs are clear."},
        {"text": "No pleural  effusion."},
        {"text": "The lungs are clear."},
    ],
    "impression_sentences": [{"text": "No acute process."}, {"text": "Heart size is normal."}],
}


@pytest.fixture
def prompt():
    return llm_judge.load_prompt(REPO / "prompts" / "llm_judge.yaml")


def test_judge_messages(prompt):
    system, user = llm_judge.judge_messages(REPORT, prompt)
    assert system == {"role": "system", "content": prompt["system"]}
    assert "EXAMINATION: Chest radiograph. INDICATION: Cough. \n" in user["content"]
    assert "The lungs are clear.\nNo pleural effusion.\nThe lungs are clear.\n" in user["content"]
    assert user["content"].endswith("No acute process.\nHeart size is normal.")
    assert '{\n  "omissions"' in user["content"]


@pytest.mark.parametrize(
    "response",
    [
        '{"omissions": ["A."], "hallucinations": []}',
        'Answer:\n```json\n{"omissions": ["A."], "hallucinations": []}\n```',
        '```\n{"omissions": ["A."], "hallucinations": []}```',
        'I found one omission. {"omissions": ["A."], "hallucinations": []} That is all.',
    ],
)
def test_parse_response(response):
    assert llm_judge.parse_response(response) == {"omissions": ["A."], "hallucinations": []}


def test_parse_response_fenced_block_is_non_greedy():
    # The fenced pattern stops at the first "}", which is invalid JSON here; the brace span is used instead.
    assert llm_judge.parse_response('```json {"a": {"b": 1}} ```') == {"a": {"b": 1}}


def test_parse_response_failure():
    with pytest.raises(ValueError, match="Could not extract valid JSON"):
        llm_judge.parse_response("no JSON here")


def test_scores_from_output():
    output = {
        "omissions": ["The lungs\nare  clear.", "Not in the report."],
        "hallucinations": ["No acute process."],
    }
    scores = llm_judge.scores_from_output(REPORT, output)
    assert scores == {"findings_scores": [1.0, 0.0, 1.0], "impression_scores": [1.0, 0.0]}
    assert llm_judge.scores_from_output(REPORT, {}) == {
        "findings_scores": [0.0, 0.0, 0.0],
        "impression_scores": [0.0, 0.0],
    }


def test_judge_report_passes_raw_response(prompt):
    seen = []
    response = '```json\n{"omissions": ["No pleural effusion."], "hallucinations": []}\n```'
    scores = llm_judge.judge_report(REPORT, prompt, lambda conversations: [response], seen.append)
    assert seen == [response]
    assert scores["findings_scores"] == [0.0, 1.0, 0.0]


def test_response_schema():
    assert llm_judge.response_schema("local") is None
    assert llm_judge.response_schema("vllm") is None
    assert llm_judge.response_schema("vertex_ai") == llm_judge.RESPONSE_SCHEMA
    assert llm_judge.response_schema("claude_api") == llm_judge.RESPONSE_SCHEMA


def test_with_retries():
    waits, attempts = [], []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("Error code: 529 overloaded_error")
        return {"ok": True}

    assert llm_judge.with_retries(flaky, sleep=waits.append) == {"ok": True}
    assert waits == [10, 20]

    def always_overloaded():
        raise RuntimeError("Overloaded")

    waits.clear()
    with pytest.raises(RuntimeError):
        llm_judge.with_retries(always_overloaded, sleep=waits.append)
    assert waits == [10, 20, 40, 80, 160]

    def broken():
        raise ValueError("bad JSON")

    waits.clear()
    with pytest.raises(ValueError):
        llm_judge.with_retries(broken, sleep=waits.append)
    assert waits == []


@needs_scispacy
def test_radcheck_prompts():
    template = (REPO / "prompts" / "radcheck.txt").read_text()
    findings, impression = radcheck.candidate_prompts(REPORT, template)
    assert len(findings) == 3 and len(impression) == 2
    assert findings[1] == (
        "### TASK: DETECT_OMISSION\n\n[CONTEXT]\nEXAMINATION: Chest radiograph. INDICATION: Cough. \n\n"
        "[FINDINGS]\nThe lungs are clear. No pleural effusion. The lungs are clear.\n\n"
        "[IMPRESSION]\n1. No acute process.\n2. Heart size is normal.\n\n"
        "[CANDIDATE SENTENCE]\nNo pleural effusion."
    )
    assert impression[0].startswith("### TASK: DETECT_HALLUCINATION\n")


@pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.key)
def test_prediction_files_format(detector):
    path = REPO / "predictions" / f"{detector.key}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 300
    assert all(set(row) == {"radhalo_id", "findings_scores", "impression_scores"} for row in rows)


def test_load_predictions_checks_lengths(tmp_path):
    path = tmp_path / "judge.jsonl"
    path.write_text(json.dumps({"radhalo_id": 0, "findings_scores": [0.0], "impression_scores": [1.0, 0.0]}))
    with pytest.raises(ValueError):
        load_predictions(path, [REPORT])
