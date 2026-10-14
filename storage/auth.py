"""
Accounts: sign up, sign in, delete
-----------------------------------
Security design:
  - Passwords are never stored. We store PBKDF2-HMAC-SHA256 hashes with a
    random 16-byte salt per user and 600,000 iterations (OWASP 2023
    recommendation), plus the iteration count so it can be raised later.
  - Hashes are compared in constant time (hmac.compare_digest).
  - Sign-in failures return ONE generic message ("Invalid username or
    password") whether the user exists or not, and a dummy hash is computed
    for unknown users so response time doesn't reveal which usernames exist.
  - Brute-force protection: 5 failed attempts lock the account for 5 minutes.
  - Password policy: 8-128 characters, at least one letter and one digit,
    not the username, not a common password.
  - Deleting an account requires the password and removes every row and
    file belonging to the user.
"""

import hashlib
import hmac
import os
import re
import shutil
import time

from storage import db

ITERATIONS = 600_000
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 300

USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,32}$")
COMMON_PASSWORDS = {
    "password", "password1", "password123", "12345678", "123456789", "1234567890", "qwerty123",
    "qwertyuiop", "iloveyou1", "admin123", "welcome1", "abc12345", "letmein1", "11111111", "passw0rd",
}
_DUMMY_SALT = os.urandom(16)


def _hash(password: str, salt: bytes, iterations: int) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations).hex()


def normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def check_password_policy(username: str, password: str):
    """Returns an error message, or None if the password is acceptable."""
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if len(password) > 128:
        return "Password must be at most 128 characters."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        return "Password must contain at least one letter and one number."
    if password.lower() == normalize_username(username) or password.lower() in COMMON_PASSWORDS:
        return "That password is too easy to guess. Choose another one."
    return None


def register(username: str, password: str, confirm: str = None) -> dict:
    """Returns {"ok": bool, "message": str, "user": {"id", "username"} | None}"""
    name = normalize_username(username)
    if not USERNAME_RE.match(name):
        return {"ok": False, "user": None,
                "message": "Username must be 3-32 characters: letters, numbers, dot, dash or underscore."}
    if confirm is not None and password != confirm:
        return {"ok": False, "user": None, "message": "Passwords do not match."}
    problem = check_password_policy(name, password or "")
    if problem:
        return {"ok": False, "user": None, "message": problem}

    salt = os.urandom(16)
    with db.session() as conn:
        if conn.execute("SELECT 1 FROM users WHERE username = ?", (name,)).fetchone():
            return {"ok": False, "user": None, "message": "That username is already taken."}
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash, salt, iterations, created_at, last_login) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name, _hash(password, salt, ITERATIONS), salt.hex(), ITERATIONS, time.time(), time.time()),
        )
        user_id = cursor.lastrowid
    return {"ok": True, "message": "Account created.", "user": {"id": user_id, "username": name}}


def authenticate(username: str, password: str) -> dict:
    """Returns {"ok": bool, "message": str, "user": {...} | None, "locked": bool}"""
    generic = "Invalid username or password."
    name = normalize_username(username)
    now = time.time()
    with db.session() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (name,)).fetchone()
        if row is None:
            _hash(password or "", _DUMMY_SALT, ITERATIONS)  # equalise timing for unknown users
            return {"ok": False, "user": None, "locked": False, "message": generic}

        if row["locked_until"] > now:
            minutes = max(1, round((row["locked_until"] - now) / 60))
            return {"ok": False, "user": None, "locked": True,
                    "message": f"Too many failed attempts. Try again in about {minutes} minute(s)."}

        candidate = _hash(password or "", bytes.fromhex(row["salt"]), row["iterations"])
        if not hmac.compare_digest(candidate, row["password_hash"]):
            failed = row["failed_attempts"] + 1
            if failed >= MAX_FAILED_ATTEMPTS:
                conn.execute("UPDATE users SET failed_attempts = 0, locked_until = ? WHERE id = ?",
                             (now + LOCKOUT_SECONDS, row["id"]))
                return {"ok": False, "user": None, "locked": True,
                        "message": f"Too many failed attempts. The account is locked for {LOCKOUT_SECONDS // 60} minutes."}
            conn.execute("UPDATE users SET failed_attempts = ? WHERE id = ?", (failed, row["id"]))
            return {"ok": False, "user": None, "locked": False, "message": generic}

        conn.execute("UPDATE users SET failed_attempts = 0, locked_until = 0, last_login = ? WHERE id = ?",
                     (now, row["id"]))
    return {"ok": True, "user": {"id": row["id"], "username": row["username"]}, "locked": False,
            "message": "Signed in."}


def delete_account(user_id: int, password: str) -> dict:
    """Deletes the user, all their rows (ON DELETE CASCADE) and all their files."""
    with db.session() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
        if row is None:
            return {"ok": False, "message": "Account not found."}
        candidate = _hash(password or "", bytes.fromhex(row["salt"]), row["iterations"])
        if not hmac.compare_digest(candidate, row["password_hash"]):
            return {"ok": False, "message": "Password is incorrect."}
        conn.execute("DELETE FROM users WHERE id = ?", (int(user_id),))
    shutil.rmtree(db.user_dir(user_id), ignore_errors=True)
    return {"ok": True, "message": "Your account and all saved data were deleted."}
