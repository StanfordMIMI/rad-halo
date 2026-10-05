"""Claude models through the Anthropic API or Vertex AI (streaming requests).

`claude_api` reproduces the requests of the paper's Opus 4.5 runs. `claude_vertex` serves current models
(e.g. Opus 5.5) on Vertex AI: adaptive thinking with an effort level and structured JSON output.
"""

import json
import os
import time
from collections.abc import Mapping

TOOL_NAME = "structured_output"
EFFORT_BETA = "effort-2025-11-24"
EXTRACTION_SYSTEM = "Extract the structured data from the analysis into the required JSON format."


def _forced_tool(json_schema: dict) -> dict:
    return {
        "tools": [
            {
                "name": TOOL_NAME,
                "description": "Output the result in the required JSON format",
                "input_schema": json_schema,
            }
        ],
        "tool_choice": {"type": "tool", "name": TOOL_NAME},
    }


def claude_request(cfg: Mapping, conversation: list[dict], json_schema: dict | None = None) -> dict:
    """Keyword arguments of the first `messages.stream` call; a `betas` key selects the beta endpoint.

    top_p is never sent. With extended thinking, temperature is dropped and a JSON schema is applied
    by a second call (`extraction_request`); without thinking, the schema is a forced tool.
    """
    request = {
        "model": cfg["path"],
        "max_tokens": cfg.get("max_new_tokens"),
        "system": conversation[0]["content"],
        "messages": [conversation[1]],
    }
    thinking = cfg.get("thinking")
    if thinking is not None and "budget_tokens" in thinking:
        request["thinking"] = {"type": "enabled", "budget_tokens": thinking["budget_tokens"]}
    else:
        if cfg.get("temperature") is not None:
            request["temperature"] = cfg["temperature"]
        if json_schema is not None:
            request.update(_forced_tool(json_schema))
    if cfg.get("effort") is not None:
        request["betas"] = [EFFORT_BETA]
        request["output_config"] = {"effort": cfg["effort"]}
    return request


def extraction_request(cfg: Mapping, analysis: str, json_schema: dict) -> dict:
    """Second call after a thinking call: same model, no thinking, forced tool on the free-text analysis."""
    return {
        "model": cfg["path"],
        "max_tokens": cfg.get("max_new_tokens"),
        "system": EXTRACTION_SYSTEM,
        "messages": [{"role": "user", "content": analysis}],
        **_forced_tool(json_schema),
    }


def build_claude(cfg: Mapping, json_schema: dict | None = None):
    """Batch prediction function for a `claude_api` config (one or two requests per conversation)."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def stream(request: dict):
        endpoint = client.beta.messages if "betas" in request else client.messages
        with endpoint.stream(**request) as s:
            return s.get_final_message()

    def respond(conversation: list[dict]) -> str:
        request = claude_request(cfg, conversation, json_schema)
        message = stream(request)
        if "thinking" in request:
            text = next(block.text for block in message.content if block.type == "text")
            if json_schema is None:
                return text
            message = stream(extraction_request(cfg, text, json_schema))
        elif json_schema is None:
            return message.content[0].text
        return json.dumps(message.content[0].input)

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
