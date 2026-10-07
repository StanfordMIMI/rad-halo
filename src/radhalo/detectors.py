"""The ten detector configurations evaluated in the paper.

Each frontier model is evaluated at its lowest and highest available reasoning setting (exact parameters in
`configs/judges/`); Qwen 3 8B with thinking off and on; Gemini 2.5 Flash with thinking off.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detector:
    key: str  # prediction file stem and config name
    model: str
    reasoning: str  # "lowest", "highest" or "" for a single configuration
    is_classifier: bool = False

    @property
    def label(self) -> str:
        return f"{self.model} ({self.reasoning})" if self.reasoning else self.model

    @property
    def highest_reasoning(self) -> bool:
        return self.reasoning == "highest"


DETECTORS = [
    Detector("gemini-2.5-flash", "Gemini 2.5 Flash", ""),
    Detector("gemini-3.1-pro_low", "Gemini 3.1 Pro", "lowest"),
    Detector("gemini-3.1-pro_high", "Gemini 3.1 Pro", "highest"),
    Detector("claude-opus-5.5_low", "Claude Opus 5.5", "lowest"),
    Detector("claude-opus-5.5_max", "Claude Opus 5.5", "highest"),
    Detector("gpt-6-astra_low", "GPT-6 Astra", "lowest"),
    Detector("gpt-6-astra_max", "GPT-6 Astra", "highest"),
    Detector("qwen3-8b", "Qwen 3 8B", "lowest"),
    Detector("qwen3-8b_thinking", "Qwen 3 8B", "highest"),
    Detector("radcheck-8b", "RadCheck 8B", "", is_classifier=True),
]

# API judges compared in the self-evaluation check of the Limitations section.
FRONTIER_JUDGES = [d.key for d in DETECTORS if d.model.startswith(("Gemini", "Claude", "GPT"))]


def get(key: str) -> Detector:
    return next(d for d in DETECTORS if d.key == key)
