"""Gemini models on Vertex AI through the google-genai client."""

import logging
import os
import time
from collections.abc import Mapping

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


def to_vertex_schema(schema: dict | None) -> dict | None:
    """Convert a JSON schema to Vertex AI's schema format.

    Types are upper-cased; only type, description, properties, required, items and enum are kept
    (``additionalProperties`` is dropped).
    """
    if schema is None:
        return None
    result = {}
    if "type" in schema:
        result["type"] = schema["type"].upper()
    if "description" in schema:
        result["description"] = schema["description"]
    if "properties" in schema:
        result["properties"] = {key: to_vertex_schema(value) for key, value in schema["properties"].items()}
    if "required" in schema:
        result["required"] = schema["required"]
    if "items" in schema:
        result["items"] = to_vertex_schema(schema["items"])
    if "enum" in schema:
        result["enum"] = schema["enum"]
    return result


def vertex_config(cfg: Mapping, json_schema: dict | None = None) -> dict:
    """`GenerateContentConfig` fields shared by every request of a model config."""
    config = {}
    if cfg.get("temperature") is not None:
        config["temperature"] = cfg["temperature"]
    if cfg.get("max_new_tokens") is not None:
        config["max_output_tokens"] = cfg["max_new_tokens"]
    if cfg.get("top_p") is not None:
        config["top_p"] = cfg["top_p"]
    thinking = cfg.get("thinking")
    if thinking is not None:
        if "level" in thinking:
            config["thinking_config"] = {"thinking_level": thinking["level"]}
        elif "budget" in thinking:
            config["thinking_config"] = {"thinking_budget": thinking["budget"]}
        # Any thinking setting, including budget 0, forces temperature 1.0 (as in the paper runs).
        if config.get("temperature", 1.0) != 1.0:
            log.warning("Gemini thinking requires temperature 1.0, overriding %s", config["temperature"])
            config["temperature"] = 1.0
    if json_schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = to_vertex_schema(json_schema)
    return config


def vertex_request(model: str, conversation: list[dict], config: dict) -> dict:
    """Keyword arguments of `generate_content` for one conversation (config as a plain dict)."""
    return {
        "model": model,
        "contents": conversation[1]["content"],
        "config": {"system_instruction": conversation[0]["content"], **config},
    }


def build_vertex(cfg: Mapping, json_schema: dict | None = None):
    """Batch prediction function for a `vertex_ai` config (one request per conversation, with retries)."""
    from google import genai
    from google.genai import errors, types

    client = genai.Client(
        vertexai=True, project=os.getenv("GCP_PROJECT_ID"), location=os.getenv("GCP_LOCATION")
    )
    config = vertex_config(cfg, json_schema)

    def generate(request: dict) -> str:
        kwargs = {**request, "config": types.GenerateContentConfig(**request["config"])}
        for attempt in range(MAX_ATTEMPTS):
            last = attempt == MAX_ATTEMPTS - 1
            try:
                text = client.models.generate_content(**kwargs).text
            except (errors.ServerError, errors.ClientError, TimeoutError) as e:
                if last:
                    raise
                log.warning("Request failed: %s. Retrying (%d/%d)", e, attempt + 2, MAX_ATTEMPTS)
                time.sleep(2**attempt)
                continue
            if text and text.strip():
                return text
            if last:
                raise ValueError("Model returned empty response")
            log.warning("Empty response. Retrying (%d/%d)", attempt + 2, MAX_ATTEMPTS)
            time.sleep(2**attempt)

    def predict(conversations: list[list[dict]]) -> list[str]:
        return [generate(vertex_request(cfg["path"], c, config)) for c in conversations]

    return predict
