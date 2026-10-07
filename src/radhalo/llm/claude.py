"""Claude models through the Anthropic API or Vertex AI (streaming requests).

`claude_api` is the plain text request of the Claude Sonnet 4.5 impression generator (top_p is never sent).
`claude_vertex` serves current models (Claude Opus 5.5 judges) on Vertex AI: adaptive thinking with an effort
level and structured JSON output.
"""

import json
import os
import time
from collections.abc import Mapping


def claude_request(cfg: Mapping, conversation: list[dict]) -> dict:
    """Keyword arguments of `messages.stream` for a `claude_api` config."""
    request = {
        "model": cfg["path"],
        "max_tokens": cfg.get("max_new_tokens"),
        "system": conversation[0]["content"],
        "messages": [conversation[1]],
    }
    if cfg.get("temperature") is not None:
        request["temperature"] = cfg["temperature"]
    return request


def build_claude(cfg: Mapping, json_schema: dict | None = None):
    """Batch prediction function for a `claude_api` config (text output only)."""
    if json_schema is not None:
        raise NotImplementedError("claude_api returns plain text; use claude_vertex for structured output")
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def respond(conversation: list[dict]) -> str:
        with client.messages.stream(**claude_request(cfg, conversation)) as s:
            return s.get_final_message().content[0].text

    def predict(conversations: list[list[dict]]) -> list[str]:
        return [respond(c) for c in conversations]

    return predict


def vertex_request(cfg: Mapping, conversation: list[dict], json_schema: dict | None = None) -> dict:
    """Keyword arguments of `messages.stream` for a current Claude model.

    Thinking is adaptive and always on; `effort` sets its depth. Sampling parameters are not accepted.
    """
    output_config = {"effort": cfg["effort"]}
    if json_schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": json_schema}
    return {
        "model": cfg["path"],
        "max_tokens": cfg["max_new_tokens"],
        "system": conversation[0]["content"],
        "messages": [conversation[1]],
        "output_config": output_config,
    }


def build_claude_vertex(cfg: Mapping, json_schema: dict | None = None):
    """Batch prediction function for a `claude_vertex` config (GCP_PROJECT_ID; region from the config).

    With `usage_log` in the config, token usage, stop reason and duration of every request are appended
    to that JSONL file.
    """
    import anthropic

    client = anthropic.AnthropicVertex(
        project_id=os.getenv("GCP_PROJECT_ID"), region=cfg.get("region", "global")
    )

    def respond(conversation: list[dict]) -> str:
        start = time.monotonic()
        with client.messages.stream(**vertex_request(cfg, conversation, json_schema)) as s:
            message = s.get_final_message()
        if cfg.get("usage_log"):
            with open(cfg["usage_log"], "a") as f:
                f.write(
                    json.dumps(
                        {
                            "input_tokens": message.usage.input_tokens,
                            "output_tokens": message.usage.output_tokens,
                            "stop_reason": message.stop_reason,
                            "seconds": round(time.monotonic() - start, 1),
                        }
                    )
                    + "\n"
                )
        if message.stop_reason != "end_turn":
            raise RuntimeError(f"{cfg['path']} stopped with stop_reason={message.stop_reason}")
        return next(block.text for block in message.content if block.type == "text")

    def predict(conversations: list[list[dict]]) -> list[str]:
        return [respond(c) for c in conversations]

    return predict
