"""Conversation persistence for the journal sidebar."""

from __future__ import annotations

import json

from src.db import get_conn


def ensure_session(session_id: str, user_id: int, title: str | None = None) -> None:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT id FROM sessions WHERE id = ? AND user_id = ?",
            (session_id, int(user_id)),
        ).fetchone()
        if row:
            return
        conn.execute(
            "INSERT INTO sessions(id, user_id, title) VALUES(?, ?, ?)",
            (session_id, int(user_id), title),
        )
        conn.commit()
    finally:
        conn.close()


def list_sessions(user_id: int, limit: int = 50) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT id, title, summary, tags_json, created_at FROM sessions WHERE user_id = ? "
            "ORDER BY datetime(created_at) DESC LIMIT ?",
            (int(user_id), int(limit)),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"] or "(untitled)",
                "summary": r["summary"] or "",
                "tags": json.loads(r["tags_json"]) if r["tags_json"] else [],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def load_messages(session_id: str, user_id: int) -> list[dict]:
    conn = get_conn()
    try:
        ok = conn.execute(
            "SELECT 1 FROM sessions WHERE id = ? AND user_id = ?",
            (session_id, int(user_id)),
        ).fetchone()
        if not ok:
            return []
        rows = conn.execute(
            "SELECT id, role, content, created_at FROM messages WHERE session_id = ? "
            "ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"], "db_id": int(r["id"])} for r in rows]
    finally:
        conn.close()


def append_message(session_id: str, role: str, content: str) -> int | None:
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO messages(session_id, role, content) VALUES(?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def bulk_save_messages(session_id: str, user_id: int, messages: list[dict], title: str | None) -> None:
    """
    Save an in-memory transcript into the DB for this user/session.
    If the session already has messages, do nothing (avoid duplicates).
    """
    ensure_session(session_id, user_id, title=title)
    conn = get_conn()
    try:
        existing = conn.execute(
            "SELECT COUNT(1) AS n FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if existing and int(existing["n"]) > 0:
            return
        rows = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role not in ("user", "assistant") or not content:
                continue
            rows.append((session_id, role, str(content)))
        if rows:
            conn.executemany(
                "INSERT INTO messages(session_id, role, content) VALUES(?, ?, ?)",
                rows,
            )
        conn.commit()
    finally:
        conn.close()


def update_message(message_id: int, content: str) -> None:
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            (content, int(message_id)),
        )
        conn.commit()
    finally:
        conn.close()


def delete_messages_after(session_id: str, message_id: int) -> None:
    """Delete messages with id > message_id within this session."""
    conn = get_conn()
    try:
        conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND id > ?",
            (session_id, int(message_id)),
        )
        conn.commit()
    finally:
        conn.close()


def set_session_title(session_id: str, user_id: int, title: str) -> None:
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ? AND user_id = ?",
            (title, session_id, int(user_id)),
        )
        conn.commit()
    finally:
        conn.close()


def get_session_meta(session_id: str, user_id: int) -> dict | None:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT title, summary, tags_json FROM sessions WHERE id = ? AND user_id = ?",
            (session_id, int(user_id)),
        ).fetchone()
        if not row:
            return None
        return {
            "title": row["title"] or "",
            "summary": row["summary"] or "",
            "tags": json.loads(row["tags_json"]) if row["tags_json"] else [],
        }
    finally:
        conn.close()


def set_session_meta(
    session_id: str,
    user_id: int,
    *,
    title: str | None = None,
    summary: str | None = None,
    tags: list[str] | None = None,
) -> None:
    conn = get_conn()
    try:
        fields = []
        vals = []
        if title is not None:
            fields.append("title = ?")
            vals.append(title)
        if summary is not None:
            fields.append("summary = ?")
            vals.append(summary)
        if tags is not None:
            fields.append("tags_json = ?")
            vals.append(json.dumps(tags))
        if not fields:
            return
        vals.extend([session_id, int(user_id)])
        conn.execute(
            f"UPDATE sessions SET {', '.join(fields)} WHERE id = ? AND user_id = ?",
            tuple(vals),
        )
        conn.commit()
    finally:
        conn.close()

