"""Local auth (username + password) stored in SQLite.

This is for a course demo only; do not deploy publicly without hardening.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
import re
from typing import NamedTuple

from src.db import get_conn


class AuthResult(NamedTuple):
    ok: bool
    user_id: int | None
    error: str | None


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _hash_password(password: str, *, salt: bytes | None = None) -> str:
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return f"pbkdf2_sha256${salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, salt_hex, dk_hex = stored.split("$", 2)
        if algo != "pbkdf2_sha256":
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(dk_hex)
    except Exception:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return hmac.compare_digest(actual, expected)


def register_user(username: str, password: str, email: str) -> AuthResult:
    username = (username or "").strip().lower()
    email = (email or "").strip().lower()
    if len(username) < 3:
        return AuthResult(False, None, "Username must be at least 3 characters.")
    if not EMAIL_RE.match(email or ""):
        return AuthResult(False, None, "Please enter a valid email address.")
    if len(password or "") < 6:
        return AuthResult(False, None, "Password must be at least 6 characters.")

    pw_hash = _hash_password(password)
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO users(username, email, password_hash) VALUES(?, ?, ?)",
            (username, email, pw_hash),
        )
        conn.commit()
        return AuthResult(True, int(cur.lastrowid), None)
    except sqlite3.IntegrityError:
        # Could be either username or email uniqueness.
        return AuthResult(False, None, "Username or email already exists.")
    finally:
        conn.close()


def login_user(username: str, password: str) -> AuthResult:
    username = (username or "").strip().lower()
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if not row:
            return AuthResult(False, None, "Invalid username or password.")
        if not _verify_password(password or "", row["password_hash"]):
            return AuthResult(False, None, "Invalid username or password.")
        return AuthResult(True, int(row["id"]), None)
    finally:
        conn.close()


def get_username(user_id: int) -> str | None:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT username FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
        return str(row["username"]) if row else None
    finally:
        conn.close()


def get_user_profile(user_id: int) -> dict | None:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT id, username, email, created_at FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
        if not row:
            return None
        return {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "email": (str(row["email"]) if row["email"] is not None else ""),
            "created_at": str(row["created_at"]),
        }
    finally:
        conn.close()


def update_username(user_id: int, new_username: str) -> AuthResult:
    new_username = (new_username or "").strip().lower()
    if len(new_username) < 3:
        return AuthResult(False, None, "Username must be at least 3 characters.")
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE users SET username = ? WHERE id = ?",
            (new_username, int(user_id)),
        )
        conn.commit()
        return AuthResult(True, int(user_id), None)
    except sqlite3.IntegrityError:
        return AuthResult(False, None, "That username is already taken.")
    finally:
        conn.close()


def update_email(user_id: int, new_email: str) -> AuthResult:
    new_email = (new_email or "").strip().lower()
    if not EMAIL_RE.match(new_email or ""):
        return AuthResult(False, None, "Please enter a valid email address.")
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE users SET email = ? WHERE id = ?",
            (new_email, int(user_id)),
        )
        conn.commit()
        return AuthResult(True, int(user_id), None)
    except sqlite3.IntegrityError:
        return AuthResult(False, None, "That email is already in use.")
    finally:
        conn.close()


def change_password(user_id: int, current_password: str, new_password: str) -> AuthResult:
    if len(new_password or "") < 6:
        return AuthResult(False, None, "New password must be at least 6 characters.")
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
        if not row:
            return AuthResult(False, None, "User not found.")
        if not _verify_password(current_password or "", row["password_hash"]):
            return AuthResult(False, None, "Current password is incorrect.")
        pw_hash = _hash_password(new_password)
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (pw_hash, int(user_id)),
        )
        conn.commit()
        return AuthResult(True, int(user_id), None)
    finally:
        conn.close()

