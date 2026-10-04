"""Claude models through the Anthropic API (streaming requests)."""

import json
import os
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
