"""
LLM Client
----------
Thin wrapper around the Groq API. Every other part of the agent (planner,
tools) calls through this one module - so if we ever swap providers
(e.g. to Gemini), this is the only file that needs to change.

WHY WRAP IT INSTEAD OF CALLING GROQ DIRECTLY EVERYWHERE?
1. Single place to handle "API key missing" and "API call failed" errors
   gracefully (required by the error-handling spec) instead of repeating
   try/except in every tool.
2. Single place to change the model name or add retry logic later.
3. Makes the rest of the codebase provider-agnostic and easier to explain
   in viva: "call_llm() is our one interface to the LLM."
"""

import os
import json
from groq import Groq

MODEL_NAME = "openai/gpt-oss-20b"


class LLMError(Exception):
    """Raised when the LLM cannot be reached or returns something unusable."""
    pass


def _get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMError(
            "GROQ_API_KEY is not set. Create a .env file (copy .env.example) "
            "and add your key from https://console.groq.com/keys, then restart "
            "the app."
        )
    return Groq(api_key=api_key)


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    """
    Send a system+user prompt pair to the LLM and return the raw text
    response.

    Args:
        system_prompt: sets the LLM's role/rules for this call
        user_prompt: the actual task/question/content for this call
        temperature: lower = more focused/deterministic (good for
                     factual extraction tasks, which is most of what
                     this project does)

    Raises:
        LLMError: if the key is missing or the API call fails, with a
                  message clear enough to show directly in the UI.
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    except LLMError:
        raise  # already a clear message, just pass it through

    except Exception as e:
        # Covers network errors, invalid key (401), rate limits (429),
        # model errors, etc. We deliberately catch broadly since the
        # Groq SDK can raise several different exception types.
        raise LLMError(f"The LLM API call failed. Details: {e}")


def call_llm_json(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> dict:
    """
    Same as call_llm(), but expects and parses a JSON object response.
    Used whenever we need structured output (plans, paper analysis, etc.)

    Raises:
        LLMError: if the call fails OR if the response isn't valid JSON
                  (this matters - we'd rather fail loudly here than pass
                  broken data downstream to the comparison/report tools).
    """
    # Nudge the model firmly toward JSON-only output. Some models still
    # wrap it in markdown fences sometimes, so we also strip those below.
    json_instruction = (
        "\n\nRespond with ONLY a valid JSON object. No markdown, no code "
        "fences, no explanation text before or after the JSON."
    )
    raw = call_llm(system_prompt + json_instruction, user_prompt, temperature)

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Strip ```json ... ``` or ``` ... ``` fences if the model added them anyway
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMError(
            f"The LLM did not return valid JSON, so this step could not "
            f"complete safely. Raw response started with: {raw[:150]!r}"
        )
