"""
Per-user library: saved papers, analysis cache and chat history
---------------------------------------------------------------
So a signed-in user never has to upload or analyse the same papers twice:

  - papers   : extracted text, chunks and their EMBEDDING VECTORS are saved,
               so the search index is rebuilt instantly on sign-in (no
               re-embedding)
  - cache    : the LLM's metadata/analysis per paper is saved, so the slow
               free-tier analysis step never runs twice for the same paper
  - turns    : every question with the agent's full result (answer, plan,
               trace, sources) is saved and replayed in the chat

Every function takes the user's id and every query filters on it, so one
user can never read or delete another user's data.
"""

import json
import os
import shutil
import time

import numpy as np

from storage import db

MAX_TURNS_LOADED = 60


def _json_default(value):
    if hasattr(value, "model_dump"):        # pydantic models (PaperInfo)
        return value.model_dump()
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _dumps(value) -> str:
    return json.dumps(value, default=_json_default)


# ---------------- papers ----------------
def save_paper(user_id: int, name: str, sha256: str, extraction: dict, chunks: list, vectors, check: dict):
    folder = db.paper_dir(user_id, sha256)
    with open(os.path.join(folder, "extraction.json"), "w") as f:
        f.write(_dumps(extraction))
    with open(os.path.join(folder, "chunks.json"), "w") as f:
        f.write(_dumps(chunks))
    np.save(os.path.join(folder, "vectors.npy"), np.asarray(vectors, dtype="float32"))
    with db.session() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO papers (user_id, sha256, name, num_pages, status, size_kb, warnings, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (int(user_id), sha256, name, extraction.get("num_pages", 0), extraction.get("status", "ok"),
             int(check.get("size_kb", 0)), _dumps(check.get("warnings", [])), time.time()),
        )


def list_papers(user_id: int) -> list[dict]:
    with db.session() as conn:
        rows = conn.execute("SELECT * FROM papers WHERE user_id = ? ORDER BY id", (int(user_id),)).fetchall()
    return [dict(r) for r in rows]


def load_papers(user_id: int) -> list[dict]:
    """Returns [{name, sha256, extraction, chunks, vectors, check}] - skipping any with missing files."""
    loaded = []
    for row in list_papers(user_id):
        folder = db.paper_dir(user_id, row["sha256"])
        try:
            with open(os.path.join(folder, "extraction.json")) as f:
                extraction = json.load(f)
            with open(os.path.join(folder, "chunks.json")) as f:
                chunks = json.load(f)
            vectors = np.load(os.path.join(folder, "vectors.npy"))
        except (OSError, ValueError):
            continue
        loaded.append({
            "name": row["name"], "sha256": row["sha256"], "extraction": extraction, "chunks": chunks,
            "vectors": vectors,
            "check": {"sha256": row["sha256"], "size_kb": row["size_kb"], "warnings": json.loads(row["warnings"])},
        })
    return loaded


def delete_paper(user_id: int, sha256: str):
    with db.session() as conn:
        conn.execute("DELETE FROM papers WHERE user_id = ? AND sha256 = ?", (int(user_id), sha256))
        conn.execute("DELETE FROM paper_cache WHERE user_id = ? AND sha256 = ?", (int(user_id), sha256))
    shutil.rmtree(db.paper_dir(user_id, sha256), ignore_errors=True)


# ---------------- analysis cache ----------------
def save_cache(user_id: int, sha256: str, kind: str, payload: dict):
    if kind not in ("metadata", "analysis"):
        raise ValueError("unknown cache kind")
    with db.session() as conn:
        conn.execute(
            "INSERT INTO paper_cache (user_id, sha256, kind, payload, updated_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, sha256, kind) DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at",
            (int(user_id), sha256, kind, _dumps(payload), time.time()),
        )


def load_cache(user_id: int) -> dict:
    """Returns {sha256: {kind: payload}}"""
    with db.session() as conn:
        rows = conn.execute("SELECT sha256, kind, payload FROM paper_cache WHERE user_id = ?", (int(user_id),)).fetchall()
    cache = {}
    for row in rows:
        cache.setdefault(row["sha256"], {})[row["kind"]] = json.loads(row["payload"])
    return cache


# ---------------- chat history ----------------
def save_turn(user_id: int, result: dict) -> int:
    payload = {k: v for k, v in result.items() if k != "paper_infos"}
    payload["paper_infos"] = {}  # rebuilt from the analysis cache on load
    with db.session() as conn:
        cursor = conn.execute("INSERT INTO turns (user_id, created_at, query, payload) VALUES (?, ?, ?, ?)",
                              (int(user_id), time.time(), result.get("query", ""), _dumps(payload)))
        return cursor.lastrowid


def load_turns(user_id: int, limit: int = MAX_TURNS_LOADED) -> list[dict]:
    with db.session() as conn:
        rows = conn.execute("SELECT payload FROM turns WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                            (int(user_id), int(limit))).fetchall()
    return [json.loads(r["payload"]) for r in reversed(rows)]


def clear_turns(user_id: int):
    with db.session() as conn:
        conn.execute("DELETE FROM turns WHERE user_id = ?", (int(user_id),))


def stats(user_id: int) -> dict:
    with db.session() as conn:
        papers = conn.execute("SELECT COUNT(*) FROM papers WHERE user_id = ?", (int(user_id),)).fetchone()[0]
        turns = conn.execute("SELECT COUNT(*) FROM turns WHERE user_id = ?", (int(user_id),)).fetchone()[0]
        analysed = conn.execute("SELECT COUNT(*) FROM paper_cache WHERE user_id = ? AND kind = 'analysis'",
                                (int(user_id),)).fetchone()[0]
    return {"papers": papers, "turns": turns, "analysed": analysed}
