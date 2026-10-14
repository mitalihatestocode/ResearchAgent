"""Accounts, saved library and chat history (TC32-TC38)."""

import json
import os
import stat

import numpy as np
import pytest

from storage import auth, db, library


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCHAGENT_DATA_DIR", str(tmp_path / "app"))
    monkeypatch.setattr(auth, "ITERATIONS", 1_000)  # keep tests fast; production uses 600k


def make_user(name="alice", password="correct horse 42"):
    result = auth.register(name, password, password)
    assert result["ok"], result["message"]
    return result["user"]


# ---------- TC32: sign up ----------
def test_tc32_register_and_password_policy():
    user = make_user("Alice")
    assert user["username"] == "alice"                                  # normalised
    assert not auth.register("ALICE", "another pass 99")["ok"]           # duplicate (case-insensitive)
    assert "8 characters" in auth.register("bob", "ab1")["message"]
    assert "letter and one number" in auth.register("bob", "onlyletters")["message"]
    assert "easy to guess" in auth.register("bob", "password1")["message"]
    assert "easy to guess" in auth.register("bob12345", "bob12345")["message"]
    assert "do not match" in auth.register("bob", "good pass 1", "good pass 2")["message"]
    assert not auth.register("a", "good pass 1")["ok"]                   # username too short
    assert not auth.register("../etc", "good pass 1")["ok"]              # unsafe characters


# ---------- TC33: password storage ----------
def test_tc33_passwords_are_salted_hashes_never_plaintext():
    make_user("alice", "correct horse 42")
    make_user("bob", "correct horse 42")                                 # same password
    with db.session() as conn:
        rows = conn.execute("SELECT * FROM users").fetchall()
    raw = open(db.db_path(), "rb").read()
    assert b"correct horse 42" not in raw
    assert rows[0]["password_hash"] != rows[1]["password_hash"]          # unique salts
    assert len(rows[0]["salt"]) == 32 and len(rows[0]["password_hash"]) == 64
    assert stat.S_IMODE(os.stat(db.db_path()).st_mode) == 0o600


# ---------- TC34: sign in + brute force ----------
def test_tc34_sign_in_generic_errors_and_lockout():
    make_user("alice", "correct horse 42")
    assert auth.authenticate("ALICE ", "correct horse 42")["ok"]
    wrong_user = auth.authenticate("nobody", "whatever 1")
    wrong_pass = auth.authenticate("alice", "wrong pass 1")
    assert wrong_user["message"] == wrong_pass["message"] == "Invalid username or password."

    for _ in range(auth.MAX_FAILED_ATTEMPTS - 2):
        assert not auth.authenticate("alice", "wrong pass 1")["locked"]
    locked = auth.authenticate("alice", "wrong pass 1")                   # 5th failure
    assert locked["locked"]
    assert not auth.authenticate("alice", "correct horse 42")["ok"]       # even the right password waits
    with db.session() as conn:
        conn.execute("UPDATE users SET locked_until = 0")                 # lock expires
    assert auth.authenticate("alice", "correct horse 42")["ok"]


# ---------- TC35: saved library round trip ----------
def _paper(name="paper_a.pdf"):
    extraction = {"source_name": name, "status": "ok", "message": "ok", "num_pages": 2,
                  "full_text": "text", "pages": [{"page_number": 1, "text": "text"}]}
    chunks = [{"chunk_id": f"{name}_p1_c0", "source_name": name, "page_number": 1, "section": "Results",
               "text": "MIMIC-III results", "injection_flags": []}]
    return extraction, chunks, np.ones((1, 384), dtype="float32")


def test_tc35_papers_vectors_cache_and_chat_survive_sign_out(monkeypatch):
    user = make_user()
    sha = "a" * 64
    extraction, chunks, vectors = _paper()
    library.save_paper(user["id"], "paper_a.pdf", sha, extraction, chunks, vectors, {"size_kb": 12, "warnings": []})
    library.save_cache(user["id"], sha, "analysis", {"status": "ok", "extracted_fields": {"dataset": "MIMIC-III"}})
    library.save_turn(user["id"], {"query": "What dataset?", "final_answer": "MIMIC-III [S1]", "blocked": False,
                                   "paper_infos": {"x": object()}, "usage": {"calls": 2}, "score": np.float32(0.5)})

    # "sign in again": everything comes back from disk
    loaded = library.load_papers(user["id"])
    assert loaded[0]["name"] == "paper_a.pdf" and loaded[0]["chunks"][0]["text"] == "MIMIC-III results"
    assert loaded[0]["vectors"].shape == (1, 384)
    assert library.load_cache(user["id"])[sha]["analysis"]["extracted_fields"]["dataset"] == "MIMIC-III"
    turns = library.load_turns(user["id"])
    assert turns[0]["final_answer"] == "MIMIC-III [S1]" and turns[0]["paper_infos"] == {}

    # saved vectors are reused: the embedding model is never called
    import rag.embeddings
    from rag.vector_store import VectorStore

    def boom(*args, **kwargs):
        raise AssertionError("re-embedding should not happen")
    monkeypatch.setattr(rag.embeddings, "embed_texts", boom)
    monkeypatch.setattr("rag.vector_store.embed_texts", boom)
    store = VectorStore()
    store.add_chunks(loaded[0]["chunks"], loaded[0]["vectors"])
    assert store.index.ntotal == 1


# ---------- TC36: isolation between users ----------
def test_tc36_users_cannot_see_each_others_data():
    alice, bob = make_user("alice"), make_user("bob")
    extraction, chunks, vectors = _paper()
    library.save_paper(alice["id"], "secret.pdf", "b" * 64, extraction, chunks, vectors, {"size_kb": 1})
    library.save_turn(alice["id"], {"query": "alice private question", "final_answer": "x"})
    assert library.load_papers(bob["id"]) == [] and library.load_turns(bob["id"]) == []
    library.delete_paper(bob["id"], "b" * 64)                             # bob cannot delete alice's paper
    assert len(library.load_papers(alice["id"])) == 1


# ---------- TC37: path safety ----------
@pytest.mark.parametrize("bad", ["../../etc", "a" * 63, "A" * 64, "/tmp/x", ""])
def test_tc37_paper_paths_only_accept_hex_digests(bad):
    with pytest.raises(ValueError):
        db.paper_dir(1, bad)


# ---------- TC38: delete account ----------
def test_tc38_delete_account_requires_password_and_removes_everything():
    user = make_user("alice", "correct horse 42")
    extraction, chunks, vectors = _paper()
    library.save_paper(user["id"], "p.pdf", "c" * 64, extraction, chunks, vectors, {"size_kb": 1})
    library.save_turn(user["id"], {"query": "q", "final_answer": "a"})
    folder = db.user_dir(user["id"])

    assert not auth.delete_account(user["id"], "wrong pass 1")["ok"]
    assert auth.delete_account(user["id"], "correct horse 42")["ok"]
    assert not os.path.exists(folder)
    with db.session() as conn:
        assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 0
    assert not auth.authenticate("alice", "correct horse 42")["ok"]


def test_clear_turns_and_stats():
    user = make_user()
    library.save_turn(user["id"], {"query": "q1", "final_answer": "a1"})
    library.save_turn(user["id"], {"query": "q2", "final_answer": "a2"})
    assert [t["query"] for t in library.load_turns(user["id"])] == ["q1", "q2"]   # chronological
    assert library.stats(user["id"])["turns"] == 2
    library.clear_turns(user["id"])
    assert library.load_turns(user["id"]) == []
