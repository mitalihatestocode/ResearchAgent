"""
Security Audit Log
------------------
Records security-relevant events (blocked queries, rejected files,
neutralised injections, rate limiting, softened overclaims) so they are
visible to the user in the UI's Security panel and to an operator in the
server log.

PRIVACY BY DESIGN: events never store raw user text or document content.
Queries are referenced by a short SHA-256 fingerprint, so an operator can
correlate repeated attempts without the log becoming a copy of what
users typed.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("researchagent.security")

SEVERITIES = ("info", "low", "medium", "high")


def fingerprint(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


@dataclass
class SecurityEvent:
    event: str          # e.g. "query_blocked", "file_rejected"
    severity: str       # info | low | medium | high
    detail: str         # human-readable, never contains raw user/document text
    ref: str = ""       # fingerprint of the related query/file
    timestamp: float = field(default_factory=time.time)


class AuditLog:
    def __init__(self, max_events: int = 200):
        self.events: list[SecurityEvent] = []
        self.max_events = max_events

    def record(self, event: str, severity: str, detail: str, ref: str = "") -> SecurityEvent:
        if severity not in SEVERITIES:
            severity = "info"
        entry = SecurityEvent(event, severity, detail, ref)
        self.events.append(entry)
        del self.events[:-self.max_events]
        log = logger.warning if severity in ("medium", "high") else logger.info
        log("%s [%s] %s ref=%s", event, severity, detail, ref)
        return entry

    def counts(self) -> dict:
        result = {s: 0 for s in SEVERITIES}
        for e in self.events:
            result[e.severity] += 1
        return result

    def as_dicts(self) -> list[dict]:
        return [asdict(e) for e in reversed(self.events)]
