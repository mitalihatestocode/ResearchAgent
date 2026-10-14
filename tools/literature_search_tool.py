"""
Literature Search Tool (external API integration)
--------------------------------------------------
Connects the agent to the outside world: searches public scholarly
databases for publications related to the uploaded papers or to each
suggested research gap.

APIs used (both free, no API key needed):
  1. arXiv API     - https://export.arxiv.org/api/query   (Atom XML)
  2. OpenAlex API  - https://api.openalex.org/works       (JSON, 250M+ works)
arXiv is tried first; OpenAlex is the automatic fallback if arXiv is
down, rate-limited, or returns nothing. Results are cached per query.

WHY THIS MATTERS (viva point - responsible AI):
The research-gap tool can only see the uploaded papers, so it cannot
know whether a "gap" has already been addressed elsewhere. This tool
runs a PRIOR-WORK CHECK for every suggested gap and shows the closest
existing publications next to it, so the user can judge novelty
themselves instead of trusting the LLM.

Only short keyword queries are sent to these APIs - never paper text
or any user data.
"""

import json
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_URL = "https://export.arxiv.org/api/query"
OPENALEX_URL = "https://api.openalex.org/works"
TIMEOUT_SECONDS = 12
USER_AGENT = "ResearchAgent/2.0 (academic project; mailto:research-agent@example.com)"

_ATOM = {"atom": "http://www.w3.org/2005/Atom"}

try:
    # python.org builds on macOS ship without root certificates; certifi's
    # bundle (already installed with our dependencies) fixes HTTPS there.
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = ssl.create_default_context()
_cache: dict[tuple, list[dict]] = {}


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS, context=_SSL_CONTEXT) as response:
        return response.read()


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def _keywords(query: str, limit: int = 5) -> list[str]:
    from rag.bm25 import tokenize  # reuse the same stopword-aware tokenizer
    seen = []
    for token in tokenize(query):
        if len(token) > 2 and token not in seen:
            seen.append(token)
    return seen[:limit]


def search_arxiv(query: str, max_results: int = 3) -> list[dict]:
    keywords = _keywords(query) or query.split()[:5]
    params = urllib.parse.urlencode({
        "search_query": " AND ".join(f"all:{w}" for w in keywords),
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    })
    url = f"{ARXIV_URL}?{params}"
    root = ET.fromstring(_http_get(url))
    results = []
    for entry in root.findall("atom:entry", _ATOM):
        published = entry.findtext("atom:published", default="", namespaces=_ATOM)
        authors = [_clean(a.findtext("atom:name", default="", namespaces=_ATOM))
                   for a in entry.findall("atom:author", _ATOM)]
        results.append({
            "title": _clean(entry.findtext("atom:title", default="", namespaces=_ATOM)),
            "authors": ", ".join(authors[:4]) + (" et al." if len(authors) > 4 else ""),
            "year": published[:4],
            "url": _clean(entry.findtext("atom:id", default="", namespaces=_ATOM)),
            "summary": _clean(entry.findtext("atom:summary", default="", namespaces=_ATOM))[:300],
            "source": "arXiv",
        })
    return results


def search_openalex(query: str, max_results: int = 3) -> list[dict]:
    params = urllib.parse.urlencode({
        "search": query,
        "per-page": max_results,
        "select": "title,publication_year,doi,id,authorships,cited_by_count",
    })
    data = json.loads(_http_get(f"{OPENALEX_URL}?{params}"))
    results = []
    for work in data.get("results", []):
        authors = [a.get("author", {}).get("display_name", "") for a in work.get("authorships", [])]
        results.append({
            "title": _clean(work.get("title") or ""),
            "authors": ", ".join(authors[:4]) + (" et al." if len(authors) > 4 else ""),
            "year": str(work.get("publication_year") or ""),
            "url": work.get("doi") or work.get("id") or "",
            "summary": f"Cited by {work.get('cited_by_count', 0)} works.",
            "source": "OpenAlex",
        })
    return results


def search_literature(query: str, max_results: int = 3) -> dict:
    """
    Tool entry point. Never raises - network problems become a clear status.

    Returns:
        {"status": "ok" | "empty" | "error", "message": str, "query": str,
         "provider": str, "results": [ {title, authors, year, url, summary, source} ]}
    """
    query = _clean(query)
    if not query:
        return {"status": "empty", "message": "Empty search query.", "query": query, "provider": "", "results": []}

    key = (query.lower(), max_results)
    if key in _cache:
        return {"status": "ok" if _cache[key] else "empty", "message": "Served from cache.",
                "query": query, "provider": _cache[key][0]["source"] if _cache[key] else "", "results": _cache[key]}

    errors = []
    for provider, search in (("arXiv", search_arxiv), ("OpenAlex", search_openalex)):
        try:
            results = [r for r in search(query, max_results) if r["title"]]
            if results:
                _cache[key] = results
                return {"status": "ok", "message": f"Found {len(results)} related publication(s) on {provider}.",
                        "query": query, "provider": provider, "results": results}
            errors.append(f"{provider}: no results")
        except Exception as e:  # network errors, timeouts, malformed responses
            errors.append(f"{provider}: {type(e).__name__}")
            time.sleep(0.3)

    all_empty = all(e.endswith("no results") for e in errors)
    return {
        "status": "empty" if all_empty else "error",
        "message": ("No related publications found." if all_empty
                    else "External literature search is unavailable right now (" + "; ".join(errors) + ")."),
        "query": query, "provider": "", "results": [],
    }
