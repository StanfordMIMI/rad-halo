"""The nine detector configurations evaluated in the paper."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detector:
    key: str  # prediction file stem and config name
    table_label: str
    figure_group: str
    figure_variant: str
    thinking: bool
    is_classifier: bool = False

    @property
    def figure_label(self) -> str:
        if self.figure_variant:
            return f"{self.figure_group} ({self.figure_variant})"
        return self.figure_group


DETECTORS = [
    Detector("gemini-2.5-flash", "Gemini 2.5 Flash", "Gemini 2.5 Flash", "", False),
    Detector("gemini-3-pro_low", "Gemini 3 Pro low", "Gemini 3 Pro", "low thinking", True),
    Detector("gemini-3-pro_high", "Gemini 3 Pro high", "Gemini 3 Pro", "high thinking", True),
    Detector("claude-opus-4.5", "Claude Opus 4.5 no think", "Claude Opus 4.5", "no thinking", False),
    Detector("claude-opus-4.5_16k", "Claude Opus 4.5 16k", "Claude Opus 4.5", "16k thinking", True),
    Detector("claude-opus-4.5_32k", "Claude Opus 4.5 32k", "Claude Opus 4.5", "32k thinking", True),
    Detector("qwen3-8b", "Qwen 3 8B", "Qwen 3 8B", "no thinking", False),
    Detector("qwen3-8b_thinking", "Qwen 3 8B thinking", "Qwen 3 8B", "thinking", True),
    Detector("radcheck-8b", "RadCheck 8B", "RadCheck 8B", "", False, is_classifier=True),
]

FRONTIER_JUDGES = [d.key for d in DETECTORS if d.key.startswith(("gemini", "claude"))]


def get(key: str) -> Detector:
    return next(d for d in DETECTORS if d.key == key)
