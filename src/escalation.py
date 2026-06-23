"""
Escalation policy for the QTrade support assistant.

Primary trigger (the one rule required by the spec):
    SAFETY — if the customer's message contains signals of a physical hazard
    (burning smell, overheating, smoke, sparks, fire), the assistant must not
    attempt to answer and must immediately route to a human agent.

Secondary triggers (supporting the primary with common-sense guards):
    - EXPLICIT_HUMAN_REQUEST — customer explicitly asks for a person/manager.
    - NO_GROUNDED_ANSWER    — retrieval found nothing above the similarity
                              threshold, so the assistant cannot cite a source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


class EscalationReason(Enum):
    NONE = auto()
    SAFETY = auto()
    EXPLICIT_HUMAN_REQUEST = auto()
    NO_GROUNDED_ANSWER = auto()


_SAFETY_PATTERNS = re.compile(
    r"\b(burn(ing|t|s)?|hot|overheat(ing)?|smok(e|ing)|spark(s|ing)?|fire|flame|melt(ing)?|smell)\b",
    re.IGNORECASE,
)

_HUMAN_REQUEST_PATTERNS = re.compile(
    r"\b(manager|supervisor|human|agent|person|representative|rep|escalat|speak to someone|talk to someone|real person|call me|phone)\b",
    re.IGNORECASE,
)


@dataclass
class EscalationDecision:
    should_escalate: bool
    reason: EscalationReason
    message: str


def check(query: str, is_grounded: bool) -> EscalationDecision:
    """Return an escalation decision for the given query and retrieval state."""

    if _SAFETY_PATTERNS.search(query):
        return EscalationDecision(
            should_escalate=True,
            reason=EscalationReason.SAFETY,
            message=(
                "This sounds like it may involve a safety hazard. "
                "Please stop using the device immediately and contact our support team directly. "
                "Routing you to a human agent now."
            ),
        )

    if _HUMAN_REQUEST_PATTERNS.search(query):
        return EscalationDecision(
            should_escalate=True,
            reason=EscalationReason.EXPLICIT_HUMAN_REQUEST,
            message=(
                "I understand you'd like to speak with a human agent. "
                "Routing you now — someone will be with you shortly."
            ),
        )

    if not is_grounded:
        return EscalationDecision(
            should_escalate=True,
            reason=EscalationReason.NO_GROUNDED_ANSWER,
            message=(
                "I don't have enough information in our help docs to answer that confidently. "
                "I'm routing you to a human agent who can help further."
            ),
        )

    return EscalationDecision(
        should_escalate=False,
        reason=EscalationReason.NONE,
        message="",
    )
