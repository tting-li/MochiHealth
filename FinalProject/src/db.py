"""SQLite DB helpers (users, sessions, messages)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.config import DB_PATH


def get_conn(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Path | None = None) -> None:
    conn = get_conn(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT NOT NULL UNIQUE,
              email TEXT,
              password_hash TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS sessions (
              id TEXT PRIMARY KEY,
              user_id INTEGER NOT NULL,
              title TEXT,
              summary TEXT,
              tags_json TEXT,
              created_at TEXT NOT NULL DEFAULT (datetime('now')),
              FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS messages (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL,
              role TEXT NOT NULL CHECK(role IN ('user','assistant')),
              content TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT (datetime('now')),
              FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            """
        )
        # \Lightweight migration for existing DBs
        user_cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "email" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT;")
        #Unique emails when present -- existing accounts = NULL and will be edited later
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL;"
        )
        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "summary" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN summary TEXT;")
        if "tags_json" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN tags_json TEXT;")
        conn.commit()
    finally:
        conn.close()
