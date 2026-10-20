import hashlib
import re
import unicodedata

INJECTION = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts|rules)",
    r"disregard (all |any )?(the )?(previous|prior|above|earlier|system) (instructions|prompts|rules)",
    r"forget (all |your )?(previous |prior )?(instructions|rules)",
    r"you are now (a|an|in) ",
    r"(reveal|print|show|repeat|output) (me )?(your |the )?(system prompt|hidden instructions|instructions above)",
    r"\bjailbreak\b", r"\bdeveloper mode\b",
    r"(give|rate) (this|the) paper (a )?(positive|perfect|10/10|high) (review|score|rating)",
]
MISUSE = {
    r"\b(fabricate|make up|invent|fake)\b.{0,40}\b(results?|data|dataset|citations?|references?|experiments?)\b":
        "fabricating research data or citations",
    r"\b(bypass|evade|avoid|beat|fool)\b.{0,30}\b(plagiarism|turnitin|ai[- ]?detect\w*|detector)\b":
        "getting around plagiarism or AI-detection checks",
    r"\bwrite (my|the|a) (entire |whole |full )?(thesis|dissertation)\b": "ghost-writing work for submission",
}
OVERCLAIMS = {
    r"\b(completely |entirely |truly )?novel\b": "potentially under-explored",
    r"\bfirst[- ]ever\b": "possibly early",
    r"\bnever (been )?(studied|explored|investigated|addressed)( before)?\b": "not addressed in these papers",
    r"\bgroundbreaking\b": "potentially valuable",
    r"\brevolutionary\b": "potentially impactful",
    r"\b(will|would) (definitely|certainly|surely)\b": "may",
    r"\bguaranteed?\b": "possible",
}
INVISIBLE = re.compile("[­​-‏‪-‮⁠-⁤﻿]")


def normalize(text):
    text = INVISIBLE.sub("", unicodedata.normalize("NFKC", text or ""))
    return re.sub(r"\s+", " ", text).lower()


def check_question(question):
    q = (question or "").strip()
    text = normalize(q)
    if not q:
        return "Please type a question about the papers."
    if len(q) > 1000:
        return "Please keep questions under 1000 characters."
    if any(re.search(p, text) for p in INJECTION):
        return ("This looks like an attempt to override the agent's instructions, so it was blocked. "
                "Please ask about the content of the papers.")
    for pattern, reason in MISUSE.items():
        if re.search(pattern, text):
            return f"I can't help with {reason}. I can help you understand, compare and analyse the papers instead."
    return None


def has_injection(text):
    return any(re.search(p, normalize(text)) for p in INJECTION)


def clean_evidence(text):
    text = INVISIBLE.sub("", unicodedata.normalize("NFKC", text or ""))
    text = re.sub(r"</?\s*evidence\b[^>]*>", "[removed]", text, flags=re.I)
    for pattern in INJECTION:
        text = re.sub(pattern, "[removed: instruction-like text]", text, flags=re.I)
    return text


def soften(text):
    for pattern, replacement in OVERCLAIMS.items():
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def check_pdf(data):
    sha = hashlib.sha256(data).hexdigest()
    if not data.lstrip()[:5].startswith(b"%PDF-"):
        return False, "This file is not a real PDF.", sha
    if len(data) > 25 * 1024 * 1024:
        return False, "This PDF is larger than 25 MB.", sha
    for name, why in [(b"/Encrypt", "is password-protected"), (b"/JavaScript", "contains JavaScript"),
                      (b"/JS", "contains JavaScript"), (b"/Launch", "can launch programs")]:
        if re.search(re.escape(name) + rb"(?![A-Za-z0-9])", data):
            return False, f"Rejected: this PDF {why}.", sha
    return True, "ok", sha
