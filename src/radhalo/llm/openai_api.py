"""OpenAI reasoning models (e.g. GPT-6 Astra) through Chat Completions.

Reasoning is always on; `effort` sets its depth. Sampling parameters are not accepted with reasoning.
Responses are not stored by OpenAI (`store=False`); the key must belong to a zero-data-retention project.
"""

import json
import os
import time
from collections.abc import Mapping

SCHEMA_NAME = "judge_output"


def openai_request(cfg: Mapping, conversation: list[dict], json_schema: dict | None = None) -> dict:
    """Keyword arguments of `chat.completions.create` for one conversation."""
    request = {
        "model": cfg["path"],
        "messages": conversation,
        "reasoning_effort": cfg["effort"],
        "max_completion_tokens": cfg["max_new_tokens"],
        "store": False,
    }
    if json_schema is not None:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": SCHEMA_NAME, "schema": json_schema, "strict": True},
        }
    return request


def build_openai(cfg: Mapping, json_schema: dict | None = None):
    """Batch prediction function for an `openai_api` config (OPENAI_API_KEY).

    With `usage_log` in the config, token usage, finish reason and duration of every request are appended
    to that JSONL file. A truncated or refused response raises instead of returning partial output.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=cfg.get("timeout", 3600))

    def respond(conversation: list[dict]) -> str:
        start = time.monotonic()
        response = client.chat.completions.create(**openai_request(cfg, conversation, json_schema))
        choice = response.choices[0]
        if cfg.get("usage_log"):
            usage = response.usage
            details = getattr(usage, "completion_tokens_details", None)
            with open(cfg["usage_log"], "a") as f:
                f.write(
                    json.dumps(
                        {
                            "input_tokens": usage.prompt_tokens,
                            "output_tokens": usage.completion_tokens,
                            "thinking_tokens": getattr(details, "reasoning_tokens", None),
                            "stop_reason": choice.finish_reason,
                            "seconds": round(time.monotonic() - start, 1),
                        }
                    )
                    + "\n"
                )
        if choice.message.refusal or choice.finish_reason != "stop":
            refused = bool(choice.message.refusal)
            raise RuntimeError(f"{cfg['path']} finished with {choice.finish_reason} (refusal: {refused})")
        return choice.message.content

    def predict(conversations: list[list[dict]]) -> list[str]:
        return [respond(c) for c in conversations]

    return predict
