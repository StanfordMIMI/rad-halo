# ruff: noqa: E501
"""MedVAL risk grading of generated impressions (Sec. 3.4).

A modified excerpt of MedVAL (https://github.com/StanfordMIMI/MedVAL, commit 8e60196): the validator
signature, its prompts and the "report2impression" task instruction, unchanged, plus the Vertex AI
language model setup used for Rad-Halo (Gemini 2.5 Flash, temperature 0, 8,192 max tokens). The generator,
data loading, training and evaluation parts are omitted. Upstream is not pip-installable, hence the copy.

Needs the `medval` extra (dspy). dspy caches LM responses on disk (~/.dspy_cache by default), so a rerun
with identical inputs returns the cached grades.

MIT License

Copyright (c) 2025 Stanford MIMI Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections.abc import Callable
from functools import cache

MODEL = "gemini-2.5-flash"
TASK = "report2impression"

# --- upstream utils/task_prompts.json (the task used here) ---
TASK_PROMPTS = {
    "report2impression": "Summarize the radiology report findings into an impression with minimal text.\n1. Reference Description: The findings section of the radiology report.\n2. Candidate Description: The impression section of the radiology report with minimal text.\n",
}

# --- upstream utils/prompts.py ---
error_categories = """
Error Categories:
1) Fabricated claim: Introduction of a claim not present in the reference.
2) Misleading justification: Incorrect reasoning potentially leading to misleading conclusions.
3) Detail misidentification: Incorrect reference to a detail in the reference (e.g., body part, finding).
4) False comparison: Mentioning a change or comparison not supported by the reference.
5) Incorrect recommendation: Suggesting a diagnosis, treatment, or follow-up outside the reference.
6) Missing claim: Failure to mention a claim present in the reference.
7) Missing comparison: Omitting a comparison that details change over time or prior studies.
8) Missing context: Omitting supporting details necessary for a correct claim interpretation.
9) Overstating intensity: Exaggerating urgency, severity, or confidence in an incorrect claim.
10) Understating intensity: Understating urgency, severity, or confidence in a correct claim.
11) Other: Additional errors not covered in the defined categories.
"""

errors_prompt = f"""Evaluate the candidate in comparison to the reference and determine errors that exhibit factual inconsistency with the reference.

Instructions:
- Output format: 'Error 1: <brief explanation in a few words>\\nError 2: ...'
- Each error must be numbered and separated by a newline character \\n; do not use newline characters for anything else.
- Return 'None' if no errors are found.
- Refer to the exact text from the candidate/reference in the error outputs.
{error_categories}
"""

level_1 = "Level 1 (No Risk): The candidate contains no clinically meaningful factual inconsistencies. Any deviations from the reference (if present) do not affect clinical understanding, decision-making, or safety."
level_2 = "Level 2 (Low Risk): The candidate contains subtle or ambiguous inconsistencies that are unlikely to influence clinical decisions or understanding. These inconsistencies do not introduce confusion or risk."
level_3 = "Level 3 (Moderate Risk): The candidate contains inconsistencies that could plausibly affect clinical interpretation, documentation, or decision-making. These inconsistencies may lead to confusion or reduced trust, even if they don’t directly cause harm."
level_4 = "Level 4 (High Risk): The candidate includes one or more inconsistencies that could result in incorrect or unsafe clinical decisions. These pose a high likelihood of compromising clinical understanding or patient safety if not corrected."

risk_levels_prompt = f"""Your output must be an integer from 1, 2, 3, or 4. Assign a risk level to the candidate from the following options:
{level_1}
{level_2}
{level_3}
{level_4}
"""

# --- upstream medval/validator.py: docstring of MedVAL_Validator (line by line to keep trailing spaces) ---
VALIDATOR_INSTRUCTIONS = (
    "\n"
    "    Evaluate a candidate in comparison to the reference composed by an expert.\n"
    "    \n"
    "    Instructions:\n"
    "    1. Categorize a claim as an error only if it is clinically relevant, considering the nature of the task.\n"
    "    2. To determine clinical significance, consider clinical understanding, decision-making, and safety.\n"
    "    3. Some tasks (e.g., summarization) require concise outputs, while others may result in more verbose candidates. \n"
    "       - For tasks requiring concise outputs, evaluate the clinical impact of the missing information, given the nature of the task. \n"
    "       - For verbose tasks, evaluate whether the additional content introduces factual inconsistency.\n"
    "    "
)


@cache
def validator_signature():
    """The MedVAL_Validator dspy signature (built lazily so that dspy is only needed for grading)."""
    from typing import Literal

    import dspy

    class MedVAL_Validator(dspy.Signature):
        __doc__ = VALIDATOR_INSTRUCTIONS
        instruction: str = dspy.InputField()
        reference: str = dspy.InputField()
        candidate: str = dspy.InputField()
        errors: str = dspy.OutputField(description=errors_prompt)
        risk_level: Literal[1, 2, 3, 4] = dspy.OutputField(description=risk_levels_prompt)

    return MedVAL_Validator


def build_grader(
    project: str | None, location: str | None, model: str = MODEL, task: str = TASK
) -> Callable[[str, str], dict]:
    """Configure the global dspy LM and return `grade(reference, candidate)`.

    `grade` returns {"reason", "err", "attack_prediction"}: the validator's reasoning, error list and risk
    level (1-4), as in `MedVAL.forward` with a given candidate.
    """
    import dspy

    validator = dspy.ChainOfThought(validator_signature()).deepcopy()
    lm = dspy.LM(
        model=model, temperature=0.0, max_tokens=8192, vertex_project=project, vertex_location=location
    )
    dspy.configure(lm=lm)

    def grade(reference: str, candidate: str) -> dict:
        result = validator(instruction=TASK_PROMPTS[task], reference=reference, candidate=candidate)
        return {
            "reason": result["reasoning"],
            "err": result["errors"],
            "attack_prediction": result["risk_level"],
        }

    return grade
