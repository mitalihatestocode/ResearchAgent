"""
Guardrails
----------
Deterministic (non-LLM) safety checks that run BEFORE and AFTER the LLM.
Prompt instructions alone are not a safety mechanism - a model can ignore
them - so the checks here are plain Python that cannot be talked out of
their job.

Layers (rubric: Safety, Ethics & Responsible AI):

1. INPUT GUARDRAILS - validate_query()
   - empty / overly long queries
   - prompt-injection attempts typed by the user
   - academic-integrity misuse (fabricating results, fake citations,
     plagiarism / detector evasion) -> refused with an explanation

2. DOCUMENT GUARDRAILS - validate_upload(), scan_for_injection(),
   sanitize_evidence()
   - file size / page count / number-of-papers limits
   - uploaded PDFs are UNTRUSTED input: text like "ignore previous
     instructions" hidden inside a paper is detected and neutralised
     before it reaches the LLM (indirect prompt injection)

3. OUTPUT GUARDRAILS - soften_overclaims()
   - research gap suggestions must never claim guaranteed novelty;
     words like "novel", "first ever", "groundbreaking" are rewritten
     to hedged language and the event is logged in the agent trace

4. FILE SECURITY - inspect_pdf_bytes(), safe_filename()
   - magic-byte check (a renamed .exe is not a PDF), encrypted PDFs
     rejected, active content (/JavaScript, /Launch) rejected, embedded
     files flagged; filenames stripped of paths and control characters

OBFUSCATION RESISTANCE: every pattern check runs on normalize_text(),
which applies Unicode NFKC normalisation (full-width / stylised letters
-> ASCII), removes zero-width and bidi-control characters, and collapses
whitespace - so "ig\u200bnore previous instructions" is still caught.
"""

import hashlib
import os
import re
import unicodedata

# ---------------- limits ----------------
MAX_QUERY_CHARS = 1000
MAX_FILE_MB = 25
MAX_PAGES_PER_PAPER = 80
MAX_PAPERS = 5

# ---------------- pattern lists ----------------
# Phrases that try to override the system's instructions. Used both on
# the user's query and on text extracted from uploaded PDFs.
INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts|rules)",
    r"disregard (all |any )?(the )?(previous|prior|above|earlier|system) (instructions|prompts|rules)",
    r"forget (all |your )?(previous |prior )?(instructions|rules)",
    r"you are now (a|an|in) ",
    r"(reveal|print|show|repeat|output) (me )?(your |the )?(system prompt|hidden instructions|instructions above)",
    # NOTE: a bare "system prompt" match was removed - LLM papers legitimately
    # discuss system prompts (false positive found on the Self-RAG paper).
    r"\bjailbreak\b",
    r"\bdeveloper mode\b",
    r"act as (an? )?(unfiltered|unrestricted)",
    r"(give|rate) (this|the) paper (a )?(positive|perfect|10/10|high) (review|score|rating)",
]

# Requests that would turn the assistant into a tool for academic misconduct.
MISUSE_PATTERNS = [
    (r"\b(fabricate|make up|invent|fake)\b.{0,40}\b(results?|data|dataset|citations?|references?|numbers|experiments?)\b",
     "fabricating research data or citations"),
    (r"\b(fake|fabricated|made[- ]up)\b.{0,20}\b(citations?|references?|results?|data)\b",
     "fabricating research data or citations"),
    (r"\b(bypass|evade|avoid|beat|fool)\b.{0,30}\b(plagiarism|turnitin|ai[- ]?detect\w*|detector)\b",
     "evading plagiarism or AI-detection checks"),
    (r"\b(paraphrase|reword|rewrite|spin)\b.{0,40}\bso (it|that it)?\s*(isn'?t|is not|won'?t be) (detected|flagged|plagiari[sz]ed)\b",
     "disguising plagiarised text"),
    (r"\bwrite (my|the|a) (entire |whole |full )?(thesis|dissertation)\b.{0,30}\b(for me|submit)\b",
     "ghost-writing work for submission"),
]

# Overclaiming language that must not appear in research-gap suggestions.
OVERCLAIM_REPLACEMENTS = [
    (r"\b(completely |entirely |truly )?novel\b", "potentially under-explored"),
    (r"\bfirst[- ]ever\b", "possibly early"),
    (r"\bnever (been )?(studied|explored|investigated|addressed)( before)?\b", "not addressed in the analysed papers"),
    (r"\bno one has (ever )?(studied|explored|investigated|addressed)\b", "the analysed papers do not address"),
    (r"\bgroundbreaking\b", "potentially valuable"),
    (r"\brevolutionary\b", "potentially impactful"),
    (r"\b(will|would) (definitely|certainly|surely)\b", "may"),
    (r"\bguaranteed?\b", "possible"),
]


# Zero-width, joiner, bidi-override and other invisible "format" characters
_INVISIBLE_RE = re.compile("[\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180e\u200b-\u200f"
                           "\u202a-\u202e\u2060-\u2064\u2066-\u206f\u3164\ufeff\uffa0]")


def normalize_text(text: str) -> str:
    """Canonical form used for all security pattern matching."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _INVISIBLE_RE.sub("", text)
    return re.sub(r"\s+", " ", text).lower()


def _matches(patterns, text):
    normalised = normalize_text(text)
    return [p for p in patterns if re.search(p, normalised)]


# ---------------- 1. input guardrails ----------------
def validate_query(query: str) -> dict:
    """
    Returns:
        {"allowed": bool, "category": str, "message": str}
        category is one of: ok, empty, too_long, prompt_injection, misuse
    """
    q = (query or "").strip()

    if not q:
        return {"allowed": False, "category": "empty",
                "message": "Please type a question about the uploaded papers."}

    if len(q) > MAX_QUERY_CHARS:
        return {"allowed": False, "category": "too_long",
                "message": f"Your question is {len(q)} characters long. Please keep it "
                           f"under {MAX_QUERY_CHARS} characters."}

    if _matches(INJECTION_PATTERNS, q):
        return {"allowed": False, "category": "prompt_injection",
                "message": "This request looks like an attempt to override the agent's "
                           "instructions, so it was blocked. Please ask a question about "
                           "the content of the uploaded papers."}

    for pattern, reason in MISUSE_PATTERNS:
        if re.search(pattern, normalize_text(q)):
            return {"allowed": False, "category": "misuse",
                    "message": f"I can't help with {reason} - that would violate academic "
                               f"integrity. I can help you understand, compare, and critically "
                               f"analyse the uploaded papers instead."}

    return {"allowed": True, "category": "ok", "message": "Query passed input guardrails."}


# ---------------- 2. document guardrails ----------------
def validate_upload(file_name: str, size_bytes: int, current_paper_count: int) -> dict:
    """Checks performed before a PDF is even opened."""
    if not file_name.lower().endswith(".pdf"):
        return {"allowed": False, "message": f"{file_name}: only PDF files are supported."}
    if size_bytes > MAX_FILE_MB * 1024 * 1024:
        return {"allowed": False,
                "message": f"{file_name}: file is {size_bytes / 1024 / 1024:.1f} MB; the limit is {MAX_FILE_MB} MB."}
    if current_paper_count >= MAX_PAPERS:
        return {"allowed": False,
                "message": f"{file_name}: at most {MAX_PAPERS} papers can be analysed at once. "
                           f"Remove a paper first."}
    return {"allowed": True, "message": "ok"}


def scan_for_injection(text: str) -> list[str]:
    """Returns the injection patterns found in a piece of (untrusted) text."""
    return _matches(INJECTION_PATTERNS, text or "")


def sanitize_evidence(text: str) -> str:
    """
    Neutralises instruction-like phrases inside paper text before it is
    placed into a prompt. The surrounding content is kept so the paper is
    still usable; only the injected instruction is replaced. Invisible
    characters are stripped first so obfuscated injections are caught,
    and our own evidence delimiters are defused so a document cannot
    "close" its <evidence> block and pose as instructions.
    """
    cleaned = _INVISIBLE_RE.sub("", unicodedata.normalize("NFKC", text or ""))
    cleaned = re.sub(r"</?\s*evidence\b[^>]*>", "[removed: delimiter-like text]", cleaned, flags=re.IGNORECASE)
    for pattern in INJECTION_PATTERNS:
        cleaned = re.sub(pattern, "[removed: instruction-like text in document]", cleaned, flags=re.IGNORECASE)
    return cleaned


# ---------------- 3. output guardrails ----------------
def soften_overclaims(text: str) -> tuple[str, list[str]]:
    """
    Rewrites overclaiming language. Returns (new_text, list_of_changes)
    so the agent trace can show exactly what the guardrail changed.
    """
    changes = []
    new_text = text
    for pattern, replacement in OVERCLAIM_REPLACEMENTS:
        for match in re.finditer(pattern, new_text, flags=re.IGNORECASE):
            changes.append(f"'{match.group(0)}' -> '{replacement}'")
        new_text = re.sub(pattern, replacement, new_text, flags=re.IGNORECASE)
    return new_text, changes


# ---------------- 4. file security ----------------
# PDF name objects that mean "run something" - a research paper never
# needs them. We never execute PDF content (pdfplumber only parses text),
# so this is defence in depth: such files are rejected outright.
_ACTIVE_CONTENT = {b"/JavaScript": "embedded JavaScript", b"/JS": "embedded JavaScript",
                   b"/Launch": "a launch action (can start programs)"}
_FLAG_CONTENT = {b"/EmbeddedFile": "embedded file attachments", b"/RichMedia": "rich media (Flash/video)"}


def _contains_name(data: bytes, name: bytes) -> bool:
    # Match a whole PDF name: "/JS" must not match "/JSONData"
    return re.search(re.escape(name) + rb"(?![A-Za-z0-9])", data) is not None


def inspect_pdf_bytes(data: bytes) -> dict:
    """
    Static inspection of an uploaded file BEFORE parsing.

    Returns:
        {"allowed": bool, "message": str, "warnings": [str], "sha256": str}
    """
    digest = hashlib.sha256(data).hexdigest()
    if not data.lstrip()[:5].startswith(b"%PDF-"):
        return {"allowed": False, "sha256": digest, "warnings": [],
                "message": "File content is not a PDF (missing %PDF header), so it was rejected even though the name ends in .pdf."}
    if _contains_name(data, b"/Encrypt"):
        return {"allowed": False, "sha256": digest, "warnings": [],
                "message": "The PDF is encrypted / password-protected and cannot be analysed safely."}
    for name, label in _ACTIVE_CONTENT.items():
        if _contains_name(data, name):
            return {"allowed": False, "sha256": digest, "warnings": [],
                    "message": f"The PDF contains {label}. Research papers do not need active content, so it was rejected."}
    warnings = [f"contains {label} (ignored - only text is extracted)"
                for name, label in _FLAG_CONTENT.items() if _contains_name(data, name)]
    return {"allowed": True, "sha256": digest, "warnings": warnings, "message": "ok"}


def safe_filename(name: str, max_len: int = 80) -> str:
    """Strips directories, control/markup characters and limits length (used as a display name and ID)."""
    base = os.path.basename((name or "").replace("\\", "/"))
    base = unicodedata.normalize("NFKC", base)
    base = re.sub(r"[^\w.\- ()]", "_", base).strip(" ._") or "document"
    stem, ext = os.path.splitext(base)
    return (stem[: max_len - len(ext)] + ext) if len(base) > max_len else base
