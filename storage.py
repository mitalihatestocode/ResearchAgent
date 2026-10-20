"""User accounts plus each user's saved papers, analyses and chat history, all in one SQLite file."""
import contextlib
import hashlib
import hmac
import io
import json
import os
import re
import sqlite3
import time

import numpy as np

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "app.db")
ITERATIONS = 600_000  # PBKDF2 rounds: slow on purpose, so stolen password hashes are hard to crack
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT UNIQUE, salt BLOB, hash BLOB,
                                  failed INTEGER DEFAULT 0, locked_until REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS papers (user INTEGER, sha TEXT, paper TEXT, chunks TEXT, vectors BLOB,
                                   cache TEXT DEFAULT '{}', PRIMARY KEY (user, sha));
CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY, user INTEGER, result TEXT);
"""


@contextlib.contextmanager
def db():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = sqlite3.connect(DB)
    try:
        con.executescript(SCHEMA)
        with con:  # commits at the end, rolls back on errors
            yield con
    finally:
        con.close()


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)

# ---------------- accounts ----------------

def sign_up(name, password, again):
    """Returns (user, message); user is None when the account could not be created."""
    name = name.strip().lower()
    if not re.fullmatch(r"[a-z0-9_.-]{3,32}", name):
        return None, "Username: 3-32 letters, numbers, dots, dashes or underscores."
    if password != again:
        return None, "The passwords don't match."
    if len(password) < 8 or not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        return None, "The password needs at least 8 characters, including a letter and a number."
    if password.lower() in (name, "password1", "password123", "qwerty123", "abc12345", "passw0rd"):
        return None, "That password is too easy to guess."
    salt = os.urandom(16)
    with db() as con:
        if con.execute("SELECT 1 FROM users WHERE name = ?", (name,)).fetchone():
            return None, "That username is taken."
        user_id = con.execute("INSERT INTO users (name, salt, hash) VALUES (?, ?, ?)",
                              (name, salt, hash_password(password, salt))).lastrowid
    return {"id": user_id, "name": name}, "Account created."


def sign_in(name, password):
    """Same message for an unknown user or a wrong password; 5 failed attempts lock the account for 5 minutes."""
    name = name.strip().lower()
    with db() as con:
        row = con.execute("SELECT id, salt, hash, failed, locked_until FROM users WHERE name = ?", (name,)).fetchone()
        if not row:
            hash_password(password, b"0" * 16)  # take the same time as a real check
            return None, "Wrong username or password."
        user_id, salt, stored, failed, locked_until = row
        if locked_until > time.time():
            return None, "Too many failed attempts - try again in a few minutes."
        if not hmac.compare_digest(hash_password(password, salt), stored):
            failed += 1
            locked = failed >= 5
            con.execute("UPDATE users SET failed = ?, locked_until = ? WHERE id = ?",
                        (0 if locked else failed, time.time() + 300 if locked else 0, user_id))
            return None, "Too many failed attempts - the account is locked for 5 minutes." if locked \
                else "Wrong username or password."
        con.execute("UPDATE users SET failed = 0 WHERE id = ?", (user_id,))
    return {"id": user_id, "name": name}, "Signed in."


def delete_account(user, password):
    with db() as con:
        salt, stored = con.execute("SELECT salt, hash FROM users WHERE id = ?", (user["id"],)).fetchone()
        if not hmac.compare_digest(hash_password(password, salt), stored):
            return False
        con.execute("DELETE FROM papers WHERE user = ?", (user["id"],))
        con.execute("DELETE FROM turns WHERE user = ?", (user["id"],))
        con.execute("DELETE FROM users WHERE id = ?", (user["id"],))
    return True

# ---------------- saved papers and chat ----------------
# Every query filters on the signed-in user's id, so nobody can read another user's data.

def save_paper(user, sha, paper, chunks, vectors):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(vectors, dtype="float32"))  # saving the vectors means no re-embedding on sign-in
    with db() as con:
        con.execute("INSERT OR REPLACE INTO papers (user, sha, paper, chunks, vectors) VALUES (?, ?, ?, ?, ?)",
                    (user["id"], sha, json.dumps(paper), json.dumps(chunks), buffer.getvalue()))


def save_cache(user, sha, cache):
    with db() as con:
        con.execute("UPDATE papers SET cache = ? WHERE user = ? AND sha = ?", (json.dumps(cache), user["id"], sha))


def load_papers(user):
    with db() as con:
        rows = con.execute("SELECT sha, paper, chunks, vectors, cache FROM papers WHERE user = ?", (user["id"],))
        return [{"sha": sha, "paper": json.loads(paper), "chunks": json.loads(chunks),
                 "vectors": np.load(io.BytesIO(vectors)), "cache": json.loads(cache)}
                for sha, paper, chunks, vectors, cache in rows.fetchall()]


def delete_paper(user, sha):
    with db() as con:
        con.execute("DELETE FROM papers WHERE user = ? AND sha = ?", (user["id"], sha))


def save_turn(user, result):
    with db() as con:
        con.execute("INSERT INTO turns (user, result) VALUES (?, ?)", (user["id"], json.dumps(result, default=str)))


def load_turns(user, limit=50):
    with db() as con:
        rows = con.execute("SELECT result FROM turns WHERE user = ? ORDER BY id DESC LIMIT ?", (user["id"], limit))
        return [json.loads(r[0]) for r in reversed(rows.fetchall())]


def clear_turns(user):
    with db() as con:
        con.execute("DELETE FROM turns WHERE user = ?", (user["id"],))
