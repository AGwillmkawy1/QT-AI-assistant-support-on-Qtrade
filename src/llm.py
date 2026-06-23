"""
LLM interface — tries Ollama first, falls back to Anthropic API if
ANTHROPIC_API_KEY is set.  Set OLLAMA_MODEL env var to override the
default model (llama3.2).
"""

from __future__ import annotations

import os
import textwrap

import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

_SYSTEM_PROMPT = textwrap.dedent("""
    You are a concise customer-support assistant for QTrade, an online marketplace
    for smart-home devices.

    Rules:
    1. Answer ONLY using the context passages provided. Do not add information from
       your training data.
    2. If the context does not contain a clear answer, respond with exactly:
       "I don't know — this isn't covered in our help docs."
    3. Keep answers short and direct (2–4 sentences max).
    4. Do NOT add a Sources line — citations are handled separately.
""").strip()


def _build_user_message(query: str, context: str) -> str:
    return f"Context:\n{context}\n\nCustomer question: {query}"


def _call_ollama(query: str, context: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(query, context)},
        ],
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _call_anthropic(query: str, context: str) -> str:
    import anthropic  # optional dep

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_message(query, context)}],
    )
    return msg.content[0].text.strip()


def generate(query: str, context: str) -> str:
    """
    Generate a grounded answer.  Tries Ollama; falls back to Anthropic API
    if ANTHROPIC_API_KEY is set and Ollama is unavailable.
    """
    try:
        return _call_ollama(query, context)
    except Exception as ollama_err:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            return _call_anthropic(query, context)
        raise RuntimeError(
            f"Ollama unavailable ({ollama_err}) and ANTHROPIC_API_KEY is not set. "
            "Start Ollama or export ANTHROPIC_API_KEY."
        ) from ollama_err
