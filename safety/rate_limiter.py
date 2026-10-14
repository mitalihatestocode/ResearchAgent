"""
Rate Limiter
------------
Per-session sliding-window limiter for agent runs. Protects against
abuse (scripted flooding of the UI) and runaway API cost: every agent
run can make 5-25 LLM calls, so an unbounded loop of questions would
quickly exhaust a Groq quota.

Two limits:
  - burst limit : max_requests within window_seconds (default 6/min)
  - session cap : max_total runs per session (default 120)
"""

import time
from collections import deque


class RateLimiter:
    def __init__(self, max_requests: int = 6, window_seconds: float = 60.0, max_total: int = 120, clock=time.monotonic):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_total = max_total
        self._clock = clock
        self._recent = deque()
        self.total = 0

    def check(self) -> dict:
        """Records the attempt if allowed. Returns {"allowed", "message", "retry_after_s"}."""
        now = self._clock()
        while self._recent and now - self._recent[0] >= self.window_seconds:
            self._recent.popleft()

        if self.total >= self.max_total:
            return {"allowed": False, "retry_after_s": None,
                    "message": f"Session limit reached ({self.max_total} questions). Clear the session or reload the page."}
        if len(self._recent) >= self.max_requests:
            retry = max(1, int(self.window_seconds - (now - self._recent[0])) + 1)
            return {"allowed": False, "retry_after_s": retry,
                    "message": f"Too many questions in a short time ({self.max_requests}/{int(self.window_seconds)}s). "
                               f"Please wait {retry}s."}

        self._recent.append(now)
        self.total += 1
        return {"allowed": True, "retry_after_s": 0, "message": "ok"}
