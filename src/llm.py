"""
LLM interface — priority order:
  1. Ollama (local, OLLAMA_MODEL, default llama3.2)
  2. Groq   (free tier, GROQ_API_KEY)
  3. Anthropic (paid, ANTHROPIC_API_KEY)
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


def _call_groq(query: str, context: str) -> str:
    from groq import Groq  # optional dep

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    chat = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(query, context)},
        ],
        max_tokens=512,
    )
    return chat.choices[0].message.content.strip()


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
    Generate a grounded answer. Tries Ollama first, then Groq, then Anthropic.
    Set GROQ_API_KEY for the free Groq option (console.groq.com).
    """
    try:
        return _call_ollama(query, context)
    except Exception as ollama_err:
        if os.getenv("GROQ_API_KEY"):
            return _call_groq(query, context)
        if os.getenv("ANTHROPIC_API_KEY"):
            return _call_anthropic(query, context)
        raise RuntimeError(
            f"No LLM available. Ollama error: {ollama_err}\n"
            "Fix: run 'ollama serve', or set GROQ_API_KEY (free at console.groq.com), "
            "or set ANTHROPIC_API_KEY."
        ) from ollama_err
