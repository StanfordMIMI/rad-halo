"""LLM-as-judge error detection (Sec. 4.1, App. C.1-C.2).

The judge sees the report context, the findings sentences and the impression sentences (one per line)
and returns the omitted findings sentences and hallucinated impression sentences as JSON. Returned
sentences are mapped back by exact text match after whitespace cleaning: a flagged text marks every
sentence with that text, and returned texts that match no sentence are ignored.
"""

import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path

import yaml

from radhalo.data.segmentation import clean_whitespace
from radhalo.llm import Predict

log = logging.getLogger(__name__)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "omissions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of findings sentences that were omitted from the impression",
        },
        "hallucinations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of impression sentences that are hallucinated (not supported by findings)",
        },
    },
    "required": ["omissions", "hallucinations"],
    "additionalProperties": False,
}
# Local backends cannot enforce a schema; the prompt asks for JSON.
NO_SCHEMA_BACKENDS = ("local", "vllm")

MAX_RETRIES = 5
RETRY_BASE_SECONDS = 10


def load_prompt(path: Path) -> dict[str, str]:
    """Read a prompt file with `system` and `user` (a `str.format` template) keys."""
    return yaml.safe_load(Path(path).read_text())


def response_schema(model_type: str) -> dict | None:
    return None if model_type in NO_SCHEMA_BACKENDS else RESPONSE_SCHEMA


def report_context(report: dict) -> str:
    """Cleaned report text before the first "FINDINGS" (examination, indication, technique, ...)."""
    return clean_whitespace(report["background_and_findings_text"].split("FINDINGS")[0])


def judge_messages(report: dict, prompt: dict[str, str]) -> list[dict[str, str]]:
    """System and user message for one report of the release."""
    user = prompt["user"].format(
        report_context_text=report_context(report),
        findings_text="\n".join(clean_whitespace(s["text"]) for s in report["findings_sentences"]),
        impression_text="\n".join(clean_whitespace(s["text"]) for s in report["impression_sentences"]),
    )
    return [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": user}]


def parse_response(response: str) -> dict:
    """Parse the JSON object of a response: whole text, else a ```json block, else first "{" to last "}"."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    first, last = response.find("{"), response.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(response[first : last + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract valid JSON from LLM response. Raw response: {response[:500]}...")


def scores_from_output(report: dict, output: dict) -> dict[str, list[float]]:
    """Binary scores per sentence: 1.0 if its cleaned text is among the returned sentences."""
    omissions = {clean_whitespace(s) for s in output.get("omissions", [])}
    hallucinations = {clean_whitespace(s) for s in output.get("hallucinations", [])}
    return {
        "findings_scores": [
            1.0 if clean_whitespace(s["text"]) in omissions else 0.0 for s in report["findings_sentences"]
        ],
        "impression_scores": [
            1.0 if clean_whitespace(s["text"]) in hallucinations else 0.0
            for s in report["impression_sentences"]
        ],
    }


def judge_report(
    report: dict,
    prompt: dict[str, str],
    predict: Predict,
    on_response: Callable[[str], None] | None = None,
) -> dict[str, list[float]]:
    """Ask the judge about one report; `on_response` receives the raw response before parsing."""
    response = predict([judge_messages(report, prompt)])[0]
    if on_response is not None:
        on_response(response)
    return scores_from_output(report, parse_response(response))


def is_overload_error(error: Exception) -> bool:
    return "overloaded_error" in str(error) or "Overloaded" in str(error)


def with_retries(fn: Callable[[], dict], sleep: Callable[[float], None] = time.sleep) -> dict:
    """Call `fn`, retrying API overload errors with exponential backoff; other errors propagate."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as error:
            if not is_overload_error(error) or attempt == MAX_RETRIES:
                raise
            wait = RETRY_BASE_SECONDS * 2**attempt
            log.warning("API overloaded (attempt %d/%d), retrying in %ds", attempt + 1, MAX_RETRIES + 1, wait)
            sleep(wait)
