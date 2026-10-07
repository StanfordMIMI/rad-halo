import subprocess
import sys

import pytest

from radhalo.llm import build_model
from radhalo.llm.claude import claude_request
from radhalo.llm.local import hf_generation_kwargs, strip_think, thinking_enabled, vllm_sampling_kwargs
from radhalo.llm.vertex import to_vertex_schema, vertex_config, vertex_request

SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {"type": "array", "items": {"type": "string"}, "description": "Flagged sentences"},
        "kind": {"type": "string", "enum": ["a", "b"]},
    },
    "required": ["labels"],
    "additionalProperties": False,
}
CONVERSATION = [
    {"role": "system", "content": "You are a fictional assistant."},
    {"role": "user", "content": "Fictional question."},
]


def test_vertex_schema_upper_cases_types_and_drops_other_keywords():
    assert to_vertex_schema(SCHEMA) == {
        "type": "OBJECT",
        "properties": {
            "labels": {"type": "ARRAY", "description": "Flagged sentences", "items": {"type": "STRING"}},
            "kind": {"type": "STRING", "enum": ["a", "b"]},
        },
        "required": ["labels"],
    }
    assert to_vertex_schema(None) is None


@pytest.mark.parametrize(
    "thinking,expected",
    [({"budget": 0}, {"thinking_budget": 0}), ({"level": "LOW"}, {"thinking_level": "LOW"})],
)
def test_vertex_thinking_forces_temperature_one(thinking, expected):
    cfg = {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 64, "thinking": thinking}
    assert vertex_config(cfg) == {
        "temperature": 1.0,
        "top_p": 0.9,
        "max_output_tokens": 64,
        "thinking_config": expected,
    }


def test_vertex_config_without_thinking_keeps_temperature_and_adds_schema():
    config = vertex_config({"temperature": 0.7}, SCHEMA)
    assert config["temperature"] == 0.7
    assert config["response_mime_type"] == "application/json"
    assert config["response_schema"] == to_vertex_schema(SCHEMA)
    assert vertex_config({}) == {}


def test_vertex_request_splits_system_and_user():
    request = vertex_request("m", CONVERSATION, {"top_p": 0.5})
    assert request == {
        "model": "m",
        "contents": "Fictional question.",
        "config": {"system_instruction": "You are a fictional assistant.", "top_p": 0.5},
    }


def test_claude_request_is_plain_text_without_top_p():
    cfg = {"path": "m", "max_new_tokens": 8, "temperature": 0.7, "top_p": 0.95}
    request = claude_request(cfg, CONVERSATION)
    assert "top_p" not in request and request["temperature"] == 0.7
    assert request["system"] == "You are a fictional assistant."
    assert request["messages"] == [CONVERSATION[1]]
    assert not {"tools", "tool_choice", "thinking", "betas", "output_config"} & set(request)


def test_claude_api_rejects_json_schema():
    from radhalo.llm.claude import build_claude

    with pytest.raises(NotImplementedError):
        build_claude({"path": "m"}, SCHEMA)


def test_strip_think_removes_multiline_blocks():
    text = '<think>\nstep one\nstep two\n</think>\n\n{"a": 1} <think>x</think>'
    assert strip_think(text).strip() == '{"a": 1}'


def test_hf_and_vllm_sampling_arguments():
    cfg = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0, "max_new_tokens": 16}
    assert hf_generation_kwargs(cfg) == {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "num_return_sequences": 1,
        "top_k": 20,
        "min_p": 0,
    }
    assert "top_k" not in hf_generation_kwargs({"temperature": 0.7, "top_p": 0.9})
    assert vllm_sampling_kwargs(cfg) == {"temperature": 0.6, "top_p": 0.95, "max_tokens": 16}
    assert thinking_enabled({"thinking": {"enable": True}}) and not thinking_enabled({})


@pytest.mark.parametrize("backend", ["local", "vllm"])
def test_local_backends_reject_json_schema(backend):
    with pytest.raises(NotImplementedError):
        build_model({"type": backend, "path": "m", "batch_size": 1, "max_length": 8}, json_schema=SCHEMA)


def test_unknown_backend():
    with pytest.raises(ValueError):
        build_model({"type": "other"})


def test_backends_import_without_heavy_dependencies():
    code = (
        "import sys, radhalo.llm.vertex, radhalo.llm.claude, radhalo.llm.local; "
        "heavy = {'torch', 'transformers', 'vllm', 'anthropic', 'google.genai'} & set(sys.modules); "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_vertex_request_uses_effort_and_structured_output():
    from radhalo.llm.claude import vertex_request

    cfg = {"path": "claude-opus-5-5", "max_new_tokens": 128000, "effort": "max"}
    conversation = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Fictional report."}]
    request = vertex_request(cfg, conversation, {"type": "object"})
    assert request["output_config"] == {
        "effort": "max",
        "format": {"type": "json_schema", "schema": {"type": "object"}},
    }
    assert request["system"] == "sys" and request["messages"] == [conversation[1]]
    assert not {"temperature", "top_p", "thinking", "tools", "tool_choice"} & set(request)


def test_openai_request_uses_effort_and_strict_schema():
    from radhalo.llm.openai_api import openai_request

    cfg = {"path": "gpt-6-astra", "max_new_tokens": 128000, "effort": "max"}
    conversation = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Fictional report."}]
    request = openai_request(cfg, conversation, {"type": "object"})
    assert request["reasoning"] == {"effort": "max"} and request["store"] is False
    assert request["text"]["format"]["strict"] is True
    assert request["instructions"] == "sys" and request["input"] == "Fictional report."
    assert not {"temperature", "top_p"} & set(request)
