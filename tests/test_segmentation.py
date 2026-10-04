import importlib.util

import pytest

from radhalo.data.mimic import extract_findings, extract_impression, has_short_findings_sentences, is_eligible
from radhalo.data.report import ImpressionSentence, Report
from radhalo.data.segmentation import (
    clean_whitespace,
    impression_text,
    is_header,
    label_numbered_items,
    remove_impression_section,
    split_numbered_list,
)

needs_scispacy = pytest.mark.skipif(
    importlib.util.find_spec("en_core_sci_sm") is None, reason="scispaCy model en_core_sci_sm not installed"
)

# Fictional report.
REPORT = (
    "EXAMINATION: Chest radiograph.\n\nINDICATION: Cough.\n\n"
    "FINDINGS: \n\nLUNGS: The lungs are clear.\nNo pleural effusion.\n\n"
    "HEART: Normal heart size.\n\n"
    "IMPRESSION: \n\n1. No acute process.\n2. Normal heart size.\n\n"
    "NOTIFICATION: None.\n"
)


def test_extract_sections():
    assert extract_findings(REPORT) == (
        "LUNGS: The lungs are clear. No pleural effusion.\n\nHEART: Normal heart size."
    )
    assert extract_impression(REPORT) == "1. No acute process. 2. Normal heart size."


def test_extract_sections_case_sensitive_and_empty():
    assert extract_findings("Findings: clear. IMPRESSION: ok") is None
    assert extract_findings("FINDINGS:   \nIMPRESSION: ok") is None
    assert extract_impression("FINDINGS: clear.\nIMPRESSION: Normal.") == "Normal."


def test_impression_ends_at_next_header_of_six_capitals():
    text = "FINDINGS: x\nIMPRESSION: Normal.\nShort\nRECOMMENDATION: none"
    assert extract_impression(text) == "Normal. Short"


def test_eligibility():
    assert is_eligible(REPORT)
    assert not is_eligible(REPORT.replace("Cough", "Pre-Procedure check"))
    assert not is_eligible(REPORT.replace("FINDINGS:", "FINDINGS -"))
    assert not is_eligible(float("nan"))


def test_clean_whitespace_and_remove_impression():
    assert clean_whitespace("a\n\nb  c\r\nd") == "a b c d"
    assert remove_impression_section(REPORT).endswith("Normal heart size.")
    assert "IMPRESSION" not in remove_impression_section(REPORT)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1. Clear lungs. 2. No effusion.", ["", "1. Clear lungs.", "2. No effusion."]),
        ("1) Clear lungs. 2) No effusion.", ["", "1) Clear lungs.", "2) No effusion."]),
        (
            "Comparison made. 1. Clear lungs. 2. No effusion.",
            ["Comparison made.", "1. Clear lungs.", "2. No effusion."],
        ),
        ("No numbered list here.", ["No numbered list here."]),
        # Out-of-sequence numbers stay inside the current item.
        ("1. Nodule of 3. 5 mm. 2. Stable.", ["", "1. Nodule of 3. 5 mm.", "2. Stable."]),
        ("2. Starts at two.", ["2. Starts at two."]),
    ],
)
def test_split_numbered_list(text, expected):
    assert split_numbered_list(text) == expected


def test_split_numbered_list_double_digits():
    text = " ".join(f"{i}. Item {i}." for i in range(1, 13))
    items = split_numbered_list(text)
    assert len(items) == 13 and items[0] == "" and items[-1] == "12. Item 12."


def test_label_numbered_items():
    assert label_numbered_items(["", "1. A.", "2. B."]) == {"pre": "", "1.": "A.", "2.": "B."}
    assert label_numbered_items(["1. A.", "2. B."]) == {"1.": "A.", "2.": "B."}
    assert label_numbered_items(["Plain text."]) == {"pre": "Plain text."}


def test_is_header():
    assert is_header("LUNGS:") and is_header("CHEST, ABDOMEN/PELVIS:") and is_header(" HEART: ")
    assert not is_header("Lungs:") and not is_header("LUNGS: clear")


def test_impression_text_renumbers():
    sentences = [
        ImpressionSentence("Comparison made.", "pre"),
        ImpressionSentence("No effusion.", "2."),
        ImpressionSentence("Lungs clear.", "2."),
        ImpressionSentence("Stable.", "4."),
    ]
    assert impression_text(sentences) == "Comparison made.\n1. No effusion. Lungs clear.\n2. Stable."
    assert impression_text([ImpressionSentence("Normal.", "1.")]) == "1. Normal."


def test_report_dict_round_trip_ignores_old_keys():
    report = Report(
        id="r1",
        full_text=REPORT,
        findings_text="LUNGS: Clear.",
        impression_text="Normal.",
        findings_sentences=["LUNGS:", "Clear."],
        impression_sentences=[ImpressionSentence("Normal.", "pre", attributed_findings_spans=["Clear."])],
    )
    d = report.to_dict()
    assert d["impression_sentences"][0]["sentence_text"] == "Normal."
    assert Report.from_dict({**d, "dataset_id": "x", "modality": None}) == report


def test_400_character_rule_counts_headers():
    report = Report("r", "", "", "", ["LUNGS:", "a" * 400], [])
    assert has_short_findings_sentences(report)
    report.findings_sentences.append("B" * 400 + ":")
    assert not has_short_findings_sentences(report)


@needs_scispacy
def test_split_findings_keeps_headers():
    from radhalo.data.segmentation import split_findings

    sentences = split_findings(extract_findings(REPORT))
    assert sentences[0] == "LUNGS:" and "HEART:" in sentences
    assert [s for s in sentences if not is_header(s)][-1] == "Normal heart size."


@needs_scispacy
def test_split_impression_labels_and_drops_short_sentences():
    from radhalo.data.segmentation import split_impression

    sentences = split_impression("1. No acute process. 2. Normal heart size.\n3. Ok.")
    assert [(s.number_label, s.text) for s in sentences] == [
        ("1.", "No acute process."),
        ("2.", "Normal heart size."),
    ]
