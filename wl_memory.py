"""
AA Trade Evaluator - memory layer.

Stores, in one SQLite file (bot_memory.db):
  messages     recent DM messages per user (trimmed to MAX_MESSAGES_PER_USER)
  queries      which units each user looked up (for "frequent lookups")
  learned      unit choices a user made with the buttons (personal memory)
  choices      every user's choice per query key (used for crowd consensus)
  last_trade   the last trade each user evaluated (for the "swap" command)
"""

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(os.environ.get("DATA_DIR", BASE_DIR), "bot_memory.db")

MAX_MESSAGES_PER_USER = 500
CONSENSUS_MIN_USERS = 3      # at least this many different users must agree
CONSENSUS_SHARE = 0.8        # and this share of all users who picked for that key

_LOCK = threading.Lock()


@contextmanager
def memory_conn():
    """One short-lived connection per call, serialized by a lock (safe for threads)."""
    with _LOCK:
        conn = sqlite3.connect(MEMORY_FILE, timeout=10)
        try:
            with conn:
                yield conn
        finally:
            conn.close()


def init():
    with memory_conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uid TEXT NOT NULL,
                ts REAL NOT NULL,
                text TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_uid ON messages(uid, id);

            CREATE TABLE IF NOT EXISTS queries (
                uid TEXT NOT NULL,
                item TEXT NOT NULL,
                ts REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_queries_uid ON queries(uid, item);

            CREATE TABLE IF NOT EXISTS learned (
                uid TEXT NOT NULL,
                query_key TEXT NOT NULL,
                canonical TEXT NOT NULL,
                PRIMARY KEY (uid, query_key)
            );

            CREATE TABLE IF NOT EXISTS choices (
                query_key TEXT NOT NULL,
                canonical TEXT NOT NULL,
                uid TEXT NOT NULL,
                PRIMARY KEY (query_key, canonical, uid)
            );

            CREATE TABLE IF NOT EXISTS last_trade (
                uid TEXT PRIMARY KEY,
                data TEXT NOT NULL
            );
        """)


def log_message(uid, text):
    with memory_conn() as c:
        c.execute("INSERT INTO messages (uid, ts, text) VALUES (?, ?, ?)",
                  (uid, time.time(), text[:1000]))
        c.execute("""DELETE FROM messages WHERE uid = ? AND id NOT IN (
                       SELECT id FROM messages WHERE uid = ? ORDER BY id DESC LIMIT ?)""",
                  (uid, uid, MAX_MESSAGES_PER_USER))


def log_items(uid, items):
    now = time.time()
    with memory_conn() as c:
        c.executemany("INSERT INTO queries (uid, item, ts) VALUES (?, ?, ?)",
                      [(uid, i, now) for i in items])


def top_items(uid, n=3):
    with memory_conn() as c:
        return c.execute("""SELECT item, COUNT(*) AS cnt FROM queries WHERE uid = ?
                            GROUP BY item ORDER BY cnt DESC LIMIT ?""", (uid, n)).fetchall()


def learn(uid, query_key, canonical):
    """Save a user's choice: personal memory plus a vote for the crowd consensus."""
    with memory_conn() as c:
        c.execute("INSERT OR REPLACE INTO learned (uid, query_key, canonical) VALUES (?, ?, ?)",
                  (uid, query_key, canonical))
        c.execute("INSERT OR IGNORE INTO choices (query_key, canonical, uid) VALUES (?, ?, ?)",
                  (query_key, canonical, uid))


def learned_map(uid):
    with memory_conn() as c:
        rows = c.execute("SELECT query_key, canonical FROM learned WHERE uid = ?", (uid,)).fetchall()
    return dict(rows)


def consensus(query_key):
    """
    The canonical unit most users picked for this query key, if enough users agree.
    Returns None when there is no clear majority.
    """
    with memory_conn() as c:
        rows = c.execute(
            "SELECT canonical, COUNT(DISTINCT uid) AS users FROM choices "
            "WHERE query_key = ? GROUP BY canonical ORDER BY users DESC",
            (query_key,)).fetchall()
    if not rows:
        return None
    total = sum(users for _, users in rows)
    top, users = rows[0]
    if users >= CONSENSUS_MIN_USERS and users / total >= CONSENSUS_SHARE:
        return top
    return None


def set_last_trade(uid, give, get):
    with memory_conn() as c:
        c.execute("INSERT OR REPLACE INTO last_trade (uid, data) VALUES (?, ?)",
                  (uid, json.dumps({"give": give, "get": get})))


def get_last_trade(uid):
    with memory_conn() as c:
        row = c.execute("SELECT data FROM last_trade WHERE uid = ?", (uid,)).fetchone()
    return json.loads(row[0]) if row else None


def forget(uid):
    """Delete everything stored about one user, including their votes in the crowd consensus."""
    with memory_conn() as c:
        c.execute("DELETE FROM learned WHERE uid = ?", (uid,))
        c.execute("DELETE FROM choices WHERE uid = ?", (uid,))
        c.execute("DELETE FROM queries WHERE uid = ?", (uid,))
        c.execute("DELETE FROM messages WHERE uid = ?", (uid,))
        c.execute("DELETE FROM last_trade WHERE uid = ?", (uid,))
