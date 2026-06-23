#!/usr/bin/env python3
"""
QTrade Support Assistant — interactive CLI.

Usage:
    python cli.py                  # uses docs/ relative to CWD
    python cli.py --docs path/to/docs
    python cli.py --query "How do I reset my SmartHub?"   # single-shot
"""

from __future__ import annotations

import argparse
import sys

from src.assistant import Assistant
from src.escalation import EscalationReason

ESCALATION_TAG = {
    EscalationReason.SAFETY: "[ESCALATED — SAFETY]",
    EscalationReason.EXPLICIT_HUMAN_REQUEST: "[ESCALATED — HUMAN REQUESTED]",
    EscalationReason.NO_GROUNDED_ANSWER: "[ESCALATED — NO GROUNDED ANSWER]",
}

DIVIDER = "─" * 60


def print_response(response) -> None:
    print()
    if response.escalated:
        tag = ESCALATION_TAG.get(response.escalation_reason, "[ESCALATED]")
        print(f"{tag}")
        print(response.answer)
    else:
        print(response.answer)
        if response.sources:
            print(f"\nSources: {', '.join(response.sources)}")
    print(DIVIDER)


def run_interactive(assistant: Assistant) -> None:
    print("QTrade Support Assistant  (type 'exit' or Ctrl-C to quit)")
    print(DIVIDER)
    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        response = assistant.ask(query)
        print_response(response)


def run_single(assistant: Assistant, query: str) -> None:
    response = assistant.ask(query)
    print_response(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="QTrade Support Assistant")
    parser.add_argument("--docs", default="docs", help="Path to help docs directory")
    parser.add_argument("--query", default=None, help="Single query (non-interactive)")
    args = parser.parse_args()

    print("Loading and indexing help docs...", end=" ", flush=True)
    try:
        assistant = Assistant(docs_dir=args.docs)
    except Exception as exc:
        print(f"\nFailed to initialise assistant: {exc}", file=sys.stderr)
        sys.exit(1)
    print("ready.\n")

    if args.query:
        run_single(assistant, args.query)
    else:
        run_interactive(assistant)


if __name__ == "__main__":
    main()
