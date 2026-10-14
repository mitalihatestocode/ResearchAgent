"""
HTML building blocks for the iOS-style UI.

SECURITY RULE: every value inserted into HTML here goes through
escape_html(). Paper text, LLM output, filenames and API results are all
untrusted, so a paper titled  <img src=x onerror=alert(1)>  renders as
text, never as markup. Callers pass raw values; escaping happens here,
in one place.
"""

import streamlit as st

from safety.output_security import escape_html as e

STATUS_COLOR = {"ok": "green", "warning": "orange", "error": "red", "skipped": "gray",
                "blocked": "red", "info": "blue"}
STATUS_SYMBOL = {"ok": "✓", "warning": "!", "error": "✕", "skipped": "–", "blocked": "⛔", "info": "i"}


def html(markup: str):
    """Renders trusted, already-escaped markup produced by the helpers below."""
    st.markdown(markup, unsafe_allow_html=True)


def icon(symbol: str, color: str = "blue", small: bool = False) -> str:
    return f'<span class="ra-icon{" sm" if small else ""} bg-{e(color)}">{e(symbol)}</span>'


def pill(text, color: str = "") -> str:
    return f'<span class="ra-pill {e(color)}">{e(text)}</span>'


def pills(items) -> str:
    """items: iterable of (text, color)."""
    return '<div class="ra-pills">' + "".join(pill(t, c) for t, c in items) + "</div>"


def hero(eyebrow: str, title: str, subtitle: str, right: str = "") -> str:
    return (f'<div class="ra-hero"><div><div class="ra-eyebrow">{e(eyebrow)}</div>'
            f'<div class="ra-title">{e(title)}</div><div class="ra-subtitle">{e(subtitle)}</div></div>'
            f'<div>{right}</div></div>')


def widgets(cards) -> str:
    """cards: iterable of (icon_symbol, color, label, value, sub)."""
    out = []
    for symbol, color, label, value, sub in cards:
        out.append(f'<div class="ra-widget"><div class="lbl">{icon(symbol, color, small=True)}{e(label)}</div>'
                   f'<div class="val">{e(value)}</div><div class="sub">{e(sub)}</div></div>')
    return '<div class="ra-widgets">' + "".join(out) + "</div>"


def section(title: str) -> str:
    return f'<div class="ra-section">{e(title)}</div>'


def h2(title: str) -> str:
    return f'<div class="ra-h2">{e(title)}</div>'


def list_rows(rows) -> str:
    """
    rows: iterable of dicts with keys
      icon (symbol), color, title, sub (optional), trail (optional), mono (bool)
    """
    out = []
    for r in rows:
        title_cls = "title mono" if r.get("mono") else "title"
        sub = f'<div class="sub">{e(r["sub"])}</div>' if r.get("sub") else ""
        trail = f'<div class="trail">{e(r["trail"])}</div>' if r.get("trail") else ""
        out.append(f'<div class="ra-row">{icon(r.get("icon", "•"), r.get("color", "gray"))}'
                   f'<div class="body"><div class="{title_cls}">{e(r["title"])}</div>{sub}</div>{trail}</div>')
    return '<div class="ra-list">' + "".join(out) + "</div>"


def user_bubble(text: str) -> str:
    return f'<div class="ra-user-row"><div class="ra-user-bubble">{e(text)}</div></div>'


def ai_label(text: str = "ResearchAgent") -> str:
    return f'<div class="ra-ai-label">{icon("✦", "indigo", small=True)}{e(text)}</div>'


def quote(text: str) -> str:
    return f'<div class="ra-quote">{e(text)}</div>'


def empty_state(symbol: str, title: str, text: str, steps) -> str:
    """steps: iterable of (icon_symbol, color, title, text)."""
    step_html = "".join(f'<div class="ra-step">{icon(s, c, small=True)}<b>{e(t)}</b><div class="desc">{e(d)}</div></div>'
                        for s, c, t, d in steps)
    return (f'<div class="ra-empty"><div class="big">{e(symbol)}</div><h3>{e(title)}</h3><p>{e(text)}</p>'
            f'<div class="ra-steps">{step_html}</div></div>')
