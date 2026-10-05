"""LLM backends behind one interface: a batch of conversations in, one response string per conversation out.

A conversation is ``[{"role": "system", "content": ...}, {"role": "user", "content": ...}]``.
Model configs live in ``configs/judges/`` and ``configs/generators/``; ``type`` selects the backend.
"""

from collections.abc import Callable, Mapping

Conversation = list[dict[str, str]]
Predict = Callable[[list[Conversation]], list[str]]


def build_model(cfg: Mapping, json_schema: dict | None = None) -> Predict:
    """Return the batch prediction function for a model config.

    `json_schema` enforces structured JSON output (API backends only).
    """
    backend = cfg["type"]
    if backend == "vertex_ai":
        from radhalo.llm.vertex import build_vertex

        return build_vertex(cfg, json_schema)
    if backend == "claude_api":
        from radhalo.llm.claude import build_claude

        return build_claude(cfg, json_schema)
    if backend == "claude_vertex":
        from radhalo.llm.claude import build_claude_vertex

        return build_claude_vertex(cfg, json_schema)
    if backend == "openai_api":
        from radhalo.llm.openai_api import build_openai

        return build_openai(cfg, json_schema)
    if backend == "local":
        from radhalo.llm.local import build_hf

        return build_hf(cfg, json_schema)
    if backend == "vllm":
        from radhalo.llm.local import build_vllm

        return build_vllm(cfg, json_schema)
    raise ValueError(f"Unknown model type: {backend}")
