"""
LLM Client
----------
Thin wrapper around the Groq API. Every other part of the agent (planner,
tools) calls through this one module - so if we ever swap providers
(e.g. to Gemini), this is the only file that needs to change.

WHY WRAP IT INSTEAD OF CALLING GROQ DIRECTLY EVERYWHERE?
1. Single place to handle "API key missing" and "API call failed" errors
   gracefully instead of repeating try/except in every tool.
2. Single place for reliability features:
     - two models with SEPARATE rate-limit budgets (Groq's free tier allows
       8,000 tokens/minute per model): on a 429 the call switches to the
       other model immediately; it only waits (honouring Retry-After) when
       both are cooling down
     - retired/unknown models (404) are skipped automatically
     - native JSON mode (response_format); if the provider's JSON validator
       rejects an output, the call is retried without it and parsed by us
     - reasoning models (gpt-oss) run with low reasoning effort and a
       generous completion budget, so they never run out of tokens
       before finishing the JSON
3. Single place to record usage (calls, tokens, latency) so the UI can
   show the cost of each agent run - useful for the demo and viva.
4. Single place for secret handling:
     - a key pasted in the UI lives in a ContextVar for that session's run
       only (use_api_key), never in os.environ, so concurrent users of a
       deployed app can never use each other's keys
     - every error message is passed through redact_secrets() so a key
       can never leak onto the screen or into logs
"""

import contextlib
import contextvars
import os
import json
import re
import threading
import time
from groq import Groq

from safety.output_security import redact_secrets

# Primary model + fallback. Both can be overridden from .env without
# touching code (GROQ_MODEL / GROQ_FALLBACK_MODEL).
MODEL_NAME = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
FALLBACK_MODEL_NAME = os.environ.get("GROQ_FALLBACK_MODEL", "openai/gpt-oss-120b")

MAX_COMPLETION_TOKENS = int(os.environ.get("GROQ_MAX_COMPLETION_TOKENS", "4096"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("GROQ_TIMEOUT_SECONDS", "45"))
MAX_ATTEMPTS = 6
BASE_BACKOFF_SECONDS = 1.5
MAX_TOTAL_WAIT_SECONDS = 75  # upper bound on rate-limit waiting per call


class LLMError(Exception):
    """Raised when the LLM cannot be reached or returns something unusable."""
    pass


# ---------------- usage tracking ----------------
# Simple module-level counters. The orchestrator resets them at the start
# of each run and reads them at the end to report "this answer took N
# LLM calls / M tokens".
_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}
_usage_lock = threading.Lock()  # tools may call the LLM from parallel threads


def reset_usage():
    for key in _usage:
        _usage[key] = 0 if key != "latency_s" else 0.0


def get_usage() -> dict:
    return dict(_usage)


# ---------------- client ----------------
# Per-run API key (set by the UI for keys pasted into a session). Falls
# back to the server's GROQ_API_KEY from .env.
_session_api_key: contextvars.ContextVar = contextvars.ContextVar("groq_session_api_key", default=None)

GROQ_KEY_RE = re.compile(r"^gsk_[A-Za-z0-9]{20,}$")


def looks_like_groq_key(key: str) -> bool:
    return bool(GROQ_KEY_RE.match((key or "").strip()))


@contextlib.contextmanager
def use_api_key(key):
    """Scopes an API key to the current run (and threads started with copy_context)."""
    token = _session_api_key.set(key.strip() if key else None)
    try:
        yield
    finally:
        _session_api_key.reset(token)


def has_api_key() -> bool:
    return bool(_session_api_key.get() or os.environ.get("GROQ_API_KEY"))


def _get_client() -> Groq:
    api_key = _session_api_key.get() or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMError(
            "GROQ_API_KEY is not set. Create a .env file (copy .env.example) "
            "and add your key from https://console.groq.com/keys, then restart "
            "the app."
        )
    # Strict per-request timeout, and no hidden SDK retries: call_llm's own
    # scheduler handles retries/fallback, so a stalled request can never hang a run
    # (found in evaluation: one request stalled for ~26 minutes with SDK defaults).
    return Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)


def _is_retryable(error: Exception) -> bool:
    """Rate limits (429), server errors (5xx) and timeouts are worth retrying."""
    status = getattr(error, "status_code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    name = type(error).__name__.lower()
    return "timeout" in name or "connection" in name or "ratelimit" in name


def _retry_after_seconds(error: Exception):
    """Reads the provider's Retry-After header from a rate-limit error, if present."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        value = headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _chat(model: str, system_prompt: str, user_prompt: str, temperature: float, json_mode: bool) -> str:
    client = _get_client()
    kwargs = dict(
        model=model,
        temperature=temperature,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    if "gpt-oss" in model:
        # Reasoning model: keep the hidden reasoning short so the visible
        # answer (our JSON) always fits in the completion budget.
        kwargs["reasoning_effort"] = "low"
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    usage = getattr(response, "usage", None)
    if usage is not None:
        with _usage_lock:
            _usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            _usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0

    content = response.choices[0].message.content
    if not content:
        raise LLMError("The LLM returned an empty response.")
    return content


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2, json_mode: bool = False) -> str:
    """
    Send a system+user prompt pair to the LLM and return the raw text.

    Scheduling: models are tried in order; a model that is rate-limited or
    erroring transiently "cools down" (Retry-After or exponential backoff)
    while the other model is used. Only if every model is cooling does the
    call wait, and never longer than MAX_TOTAL_WAIT_SECONDS in total.

    Raises:
        LLMError: if the key is missing or every attempt fails, with a
                  message clear enough to show directly in the UI (secrets
                  redacted).
    """
    models = [MODEL_NAME] + ([FALLBACK_MODEL_NAME] if FALLBACK_MODEL_NAME != MODEL_NAME else [])
    cooling_until = {m: 0.0 for m in models}
    use_json = {m: json_mode for m in models}
    failures = {m: 0 for m in models}
    unavailable, errors = set(), []
    waited = 0.0

    for attempt in range(MAX_ATTEMPTS):
        candidates = [m for m in models if m not in unavailable]
        if not candidates:
            break
        now = time.monotonic()
        ready = [m for m in candidates if cooling_until[m] <= now]
        if not ready:
            pause = min(cooling_until[m] for m in candidates) - now
            if waited + pause > MAX_TOTAL_WAIT_SECONDS:
                break
            time.sleep(max(0.0, pause))
            waited += max(0.0, pause)
            ready = [min(candidates, key=lambda m: cooling_until[m])]
        model = ready[0]

        try:
            start = time.time()
            content = _chat(model, system_prompt, user_prompt, temperature, use_json[model])
            with _usage_lock:
                _usage["calls"] += 1
                _usage["latency_s"] += time.time() - start
            return content
        except LLMError as e:
            if "GROQ_API_KEY" in str(e):
                raise  # no point retrying a missing key
            errors.append(f"{model}: {e}")
            unavailable.add(model)  # empty response - try the other model
        except Exception as e:
            status = getattr(e, "status_code", None)
            text = str(e)
            if status in (401, 403):
                raise LLMError(
                    "The Groq API rejected the API key (401/403). Check GROQ_API_KEY in your "
                    ".env file or paste a valid key in the sidebar."
                )
            errors.append(f"{model}: {text[:300]}")
            if status == 400 and "json_validate_failed" in text and use_json[model]:
                use_json[model] = False  # retry this model without provider-side JSON validation
            elif status == 404 or "model_not_found" in text or "decommissioned" in text:
                unavailable.add(model)
            elif _is_retryable(e):
                failures[model] += 1
                delay = _retry_after_seconds(e) or BASE_BACKOFF_SECONDS * (2 ** (failures[model] - 1))
                cooling_until[model] = time.monotonic() + min(delay, 30.0)
            else:
                unavailable.add(model)  # other bad requests: let the other model try

    raise LLMError(redact_secrets("The LLM API call failed. Details: " + " | ".join(errors[-3:] or ["no attempts made"])))


def parse_json_response(raw: str) -> dict:
    """
    Robustly parse a JSON object out of an LLM response. Handles the
    common failure modes: markdown code fences, and short prose before
    or after the object.
    """
    cleaned = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to the outermost {...} block in the text
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(cleaned[start:end + 1])

    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("Top-level JSON value is not an object", cleaned, 0)
    return parsed


def call_llm_json(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> dict:
    """
    Same as call_llm(), but requests JSON mode and parses a JSON object.
    Used whenever we need structured output (plans, paper analysis, etc.)

    Raises:
        LLMError: if the call fails OR if the response isn't valid JSON
                  (we'd rather fail loudly here than pass broken data
                  downstream to the comparison/report tools).
    """
    json_instruction = (
        "\n\nRespond with ONLY a valid JSON object. No markdown, no code "
        "fences, no explanation text before or after the JSON."
    )
    raw = call_llm(system_prompt + json_instruction, user_prompt, temperature, json_mode=True)

    try:
        return parse_json_response(raw)
    except json.JSONDecodeError:
        raise LLMError(
            f"The LLM did not return valid JSON, so this step could not "
            f"complete safely. Raw response started with: {raw[:150]!r}"
        )
