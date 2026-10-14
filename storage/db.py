"""
Database
--------
A local SQLite database (no server to install) holding user accounts and
each user's saved library and chat history. Large per-paper artefacts
(extracted text, chunks, embedding vectors) live as files next to it:

    data/app/researchagent.db
    data/app/users/<user_id>/<sha256 of the PDF>/{extraction.json, chunks.json, vectors.npy}

Paths are built ONLY from the integer user id and a validated 64-char hex
digest - never from filenames or other user input - so path traversal is
impossible. The data directory is created with owner-only permissions and
is git-ignored. Set RESEARCHAGENT_DATA_DIR to relocate it (tests use a
temporary directory).
"""

import contextlib
import os
import re
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT    NOT NULL UNIQUE,
    password_hash   TEXT    NOT NULL,
    salt            TEXT    NOT NULL,
    iterations      INTEGER NOT NULL,
    created_at      REAL    NOT NULL,
    last_login      REAL,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until    REAL    NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS papers (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    sha256     TEXT    NOT NULL,
    name       TEXT    NOT NULL,
    num_pages  INTEGER NOT NULL,
    status     TEXT    NOT NULL,
    size_kb    INTEGER NOT NULL,
    warnings   TEXT    NOT NULL DEFAULT '[]',
    created_at REAL    NOT NULL,
    UNIQUE (user_id, sha256)
);
CREATE TABLE IF NOT EXISTS paper_cache (
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    sha256     TEXT    NOT NULL,
    kind       TEXT    NOT NULL,          -- 'metadata' | 'analysis'
    payload    TEXT    NOT NULL,          -- JSON
    updated_at REAL    NOT NULL,
    PRIMARY KEY (user_id, sha256, kind)
);
CREATE TABLE IF NOT EXISTS turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at REAL    NOT NULL,
    query      TEXT    NOT NULL,
    payload    TEXT    NOT NULL           -- JSON of the agent result
);
CREATE INDEX IF NOT EXISTS idx_turns_user ON turns(user_id, id);
"""


def data_dir() -> str:
    path = os.environ.get("RESEARCHAGENT_DATA_DIR") or os.path.join(ROOT, "data", "app")
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


def db_path() -> str:
    return os.path.join(data_dir(), "researchagent.db")


def connect() -> sqlite3.Connection:
    path = db_path()
    new = not os.path.exists(path)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    if new:
        os.chmod(path, 0o600)  # owner read/write only
    return conn


@contextlib.contextmanager
def session():
    """Connection that commits on success, rolls back on error and always closes."""
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def user_dir(user_id: int) -> str:
    path = os.path.join(data_dir(), "users", str(int(user_id)))
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


def paper_dir(user_id: int, sha256: str) -> str:
    if not _SHA_RE.match(sha256 or ""):
        raise ValueError("invalid paper id")
    path = os.path.join(user_dir(user_id), sha256)
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path
