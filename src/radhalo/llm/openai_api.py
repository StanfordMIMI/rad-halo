"""OpenAI reasoning models (e.g. GPT-6 Astra) through the Responses API.

Reasoning is always on; `effort` sets its depth (GPT-6 Astra accepts "max" only on the Responses API).
Sampling parameters are not accepted with reasoning. Responses are not stored by OpenAI (`store=False`);
the key must belong to a zero-data-retention project.
"""

import json
import os
import time
from collections.abc import Mapping

SCHEMA_NAME = "judge_output"


def openai_request(cfg: Mapping, conversation: list[dict], json_schema: dict | None = None) -> dict:
    """Keyword arguments of `responses.create` for one conversation."""
    request = {
        "model": cfg["path"],
        "instructions": conversation[0]["content"],
        "input": conversation[1]["content"],
        "reasoning": {"effort": cfg["effort"]},
        "max_output_tokens": cfg["max_new_tokens"],
        "store": False,
    }
    if json_schema is not None:
        request["text"] = {
            "format": {"type": "json_schema", "name": SCHEMA_NAME, "schema": json_schema, "strict": True}
        }
    return request


def output_text(response) -> str:
    """Text of the message output; raises on incomplete and refused responses."""
    if response.status != "completed":
        reason = getattr(response.incomplete_details, "reason", None)
        raise RuntimeError(f"response status {response.status} ({reason})")
    parts = [c for item in response.output if item.type == "message" for c in item.content]
    if any(c.type == "refusal" for c in parts):
        raise RuntimeError("model refused")
    return "".join(c.text for c in parts if c.type == "output_text")


def build_openai(cfg: Mapping, json_schema: dict | None = None):
    """Batch prediction function for an `openai_api` config (OPENAI_API_KEY).

    With `usage_log` in the config, token usage, status and duration of every request are appended to
    that JSONL file.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=cfg.get("timeout", 3600))

    def respond(conversation: list[dict]) -> str:
        start = time.monotonic()
        response = client.responses.create(**openai_request(cfg, conversation, json_schema))
        if cfg.get("usage_log"):
            usage = response.usage
            with open(cfg["usage_log"], "a") as f:
                f.write(
                    json.dumps(
                        {
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "thinking_tokens": usage.output_tokens_details.reasoning_tokens,
                            "stop_reason": response.status,
                            "seconds": round(time.monotonic() - start, 1),
                        }
                    )
                    + "\n"
                )
        return output_text(response)

    def predict(conversations: list[list[dict]]) -> list[str]:
        return [respond(c) for c in conversations]

    return predict
