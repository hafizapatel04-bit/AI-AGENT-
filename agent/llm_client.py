"""
llm_client.py
-------------
Thin wrapper around a local Ollama server's chat endpoint, with support
for Ollama's tool-calling ("function calling") API.

Requires Ollama running locally (https://ollama.com) and a tool-capable
model pulled, e.g.:
    ollama pull llama3.1
    ollama pull qwen2.5

If Ollama isn't reachable, callers get a clear OllamaUnavailableError so
the Streamlit app can degrade gracefully instead of crashing.
"""

from __future__ import annotations

import json
from typing import Any

import requests

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"


class OllamaUnavailableError(RuntimeError):
    pass


def is_ollama_running(host: str = OLLAMA_HOST) -> bool:
    try:
        r = requests.get(f"{host}/api/tags", timeout=2)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def list_models(host: str = OLLAMA_HOST) -> list[str]:
    try:
        r = requests.get(f"{host}/api/tags", timeout=3)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except requests.exceptions.RequestException:
        return []


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str = DEFAULT_MODEL,
    host: str = OLLAMA_HOST,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """
    Calls Ollama's /api/chat endpoint (non-streaming) with optional tool
    definitions. Returns the raw `message` dict from the response, which
    may contain `tool_calls` for the agent loop to execute.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if tools:
        payload["tools"] = tools

    try:
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise OllamaUnavailableError(
            f"Could not reach Ollama at {host}. Is `ollama serve` running?"
        ) from e
    except requests.exceptions.HTTPError as e:
        raise OllamaUnavailableError(
            f"Ollama returned an error (model '{model}' pulled? try `ollama pull {model}`): {e}"
        ) from e

    data = resp.json()
    return data.get("message", {})
