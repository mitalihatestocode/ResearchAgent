"""
Output Security
---------------
Everything the UI renders comes from untrusted sources: text extracted
from uploaded PDFs, LLM output (which can be steered by those PDFs), and
titles/URLs returned by external APIs. These helpers make that content
safe to display.

- escape_html(): for any untrusted text placed inside HTML we build
  ourselves (st.markdown(..., unsafe_allow_html=True)). Prevents XSS such
  as a paper title of  <img src=x onerror=alert(1)>.
- safe_url(): only http(s) links are ever rendered; javascript:, data:,
  file: and malformed URLs become "#".
- md_link(): a Markdown link with an escaped label and a safe URL.
- redact_secrets(): masks API keys / bearer tokens before an error
  message is shown on screen or written to a log.
"""

import html
import re
from urllib.parse import urlparse

_SECRET_PATTERNS = [
    (re.compile(r"gsk_[A-Za-z0-9]{8,}"), "gsk_****"),                       # Groq
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "sk-****"),                      # OpenAI-style
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{16,}"), r"\1****"),
    (re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)['\"]?[A-Za-z0-9._\-]{12,}"), r"\1****"),
]


def escape_html(text) -> str:
    return html.escape(str(text if text is not None else ""), quote=True)


def safe_url(url) -> str:
    url = str(url or "").strip()
    try:
        parsed = urlparse(url)
    except ValueError:
        return "#"
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return "#"
    # Characters that could break out of an HTML attribute are never legitimate here
    if any(ch in url for ch in "\"'<>`\\"):
        return "#"
    # Parentheses/spaces are legal in some DOIs but would end a Markdown link early
    return url.replace(" ", "%20").replace("(", "%28").replace(")", "%29")


def md_escape(text) -> str:
    """Escape characters that would change Markdown structure inside a link label."""
    return re.sub(r"([\[\]\\`*_<>|])", r"\\\1", str(text or ""))


def md_link(label, url) -> str:
    target = safe_url(url)
    label = md_escape(label) or "link"
    return f"[{label}]({target})" if target != "#" else label


def redact_secrets(text) -> str:
    text = str(text or "")
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
