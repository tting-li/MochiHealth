"""Streamlit entry w/ RAG and Ollama help"""

from __future__ import annotations

import src.config  # noqa: F401  — configure HF cache env before heavy ML imports

import logging
import time
from collections import deque
from pathlib import Path

import streamlit as st

from src.agent_debug import agent_debug
from src.auth import (
    change_password,
    get_user_profile,
    get_username,
    login_user,
    register_user,
    update_email,
    update_username,
)
from src.config import (
    DEFAULT_RETRIEVAL_K,
    LOGS_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RATE_LIMIT_MAX_MESSAGES,
    RATE_LIMIT_WINDOW_SECONDS,
)
from src.db import init_db
from src.journal import (
    append_message,
    bulk_save_messages,
    delete_messages_after,
    ensure_session,
    get_session_meta,
    list_sessions,
    load_messages,
    set_session_meta,
    set_session_title,
    update_message,
)
from src.ollama_client import ollama_health
from src.rag_chain import build_vectorstore, run_turn
from src.session import check_rate_limit, new_session_id, record_message_time
from src.summarize import summarize_session

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("health_journal")

MOCHI_AVATAR = "assets/mochi.svg"


def _init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = new_session_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "symptoms" not in st.session_state:
        st.session_state.symptoms = []
    if "rate_times" not in st.session_state:
        st.session_state.rate_times = deque()
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "vs_error" not in st.session_state:
        st.session_state.vs_error = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    if "loaded_session_id" not in st.session_state:
        st.session_state.loaded_session_id = None
    if "guest_unsaved_chat" not in st.session_state:
        st.session_state.guest_unsaved_chat = False
    if "offer_save_guest_chat" not in st.session_state:
        st.session_state.offer_save_guest_chat = False
    if "regen_after_edit" not in st.session_state:
        st.session_state.regen_after_edit = False
    if "seeded_greeting" not in st.session_state:
        st.session_state.seeded_greeting = False

    #greeting
    if not st.session_state.seeded_greeting and not st.session_state.messages:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi, I'm **Mochi**, your personal healthcare assistant — how may I help you today?",
            }
        ]
        st.session_state.seeded_greeting = True


def _ensure_vectorstore():
    if st.session_state.vectorstore is not None:
        return
    try:
        with st.spinner("Loading knowledge base and embeddings (first run may download models)..."):
            st.session_state.vectorstore = build_vectorstore()
        st.session_state.vs_error = None
    except Exception as e:
        st.session_state.vs_error = str(e)
        logger.exception("vectorstore build failed")


def _reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.symptoms = []
    st.session_state.session_id = new_session_id()
    st.session_state.rate_times = deque()
    st.session_state.loaded_session_id = None
    st.session_state.seeded_greeting = False
    logger.info("session_reset session_id=%s", st.session_state.session_id)


def main() -> None:
    #region agent log
    agent_debug(
        "H5",
        "app.py:main:entry",
        "app_started",
        {"cwd": str(Path.cwd())},
    )
    #endregion agent log
    st.set_page_config(
        page_title="Health Journal Assistant",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_db()
    _init_state()
    _ensure_vectorstore()

    st.markdown(
        """
<style>
/* Hide Streamlit chrome (Deploy bar / toolbar / menu) */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDeployButton"] { display: none; }

.block-container { padding-top: 1.2rem; }
.stChatMessage { border-radius: 8px; }
.triage-badge { display:inline-block; padding: 2px 10px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; }
.triage-routine { background: #e7f6ea; color: #14532d; border: 1px solid #b7e4c7; }
.triage-monitor { background: #fff7cc; color: #7a5c00; border: 1px solid #ffe08a; }
.triage-urgent { background: #ffe6cc; color: #7a2e00; border: 1px solid #ffb570; }
.triage-emergency { background: #ffe0e0; color: #7a0010; border: 1px solid #ff9aa2; }
.mochiIcon { width: 38px; height: 38px; border-radius: 14px; background: #ffffff; border: 1px solid rgba(0,0,0,0.08); display: inline-flex; align-items: center; justify-content: center; margin-right: 10px; }
.mochiFace { position: relative; width: 28px; height: 22px; background: #ffffff; border: 2px solid rgba(0,0,0,0.08); border-radius: 14px; }
.mochiEye { position: absolute; top: 50%; width: 6px; height: 6px; background: #111827; border-radius: 999px; transform: translateY(-50%); }
.mochiEye.left { left: 6px; }
.mochiEye.right { right: 6px; }
.mochiSparkle { position: absolute; top: 3px; left: 1px; width: 2px; height: 2px; background: #ffffff; border-radius: 999px; opacity: 0.9; }
.mochiBlush { position: absolute; top: 65%; width: 6px; height: 3px; background: #ffb4c8; border-radius: 999px; opacity: 0.55; }
.mochiBlush.left { left: 3px; }
.mochiBlush.right { right: 3px; }
</style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Account")
        if not st.session_state.user_id:
            tab_login, tab_register = st.tabs(["Log in", "Register"])
            with tab_login:
                u = st.text_input("Username", key="login_user")
                p = st.text_input("Password", type="password", key="login_pass")
                if st.button("Log in", type="primary"):
                    res = login_user(u, p)
                    if res.ok:
                        st.session_state.user_id = res.user_id
                        prof = get_user_profile(res.user_id or 0) or {}
                        st.session_state.username = prof.get("username") or get_username(res.user_id or 0)
                        st.session_state.user_email = prof.get("email") or ""
                        if st.session_state.messages:
                            st.session_state.offer_save_guest_chat = True
                        st.success("Logged in.")
                        st.rerun()
                    else:
                        st.error(res.error or "Login failed.")
            with tab_register:
                u = st.text_input("New username", key="reg_user")
                e = st.text_input("Email", key="reg_email")
                p = st.text_input("New password", type="password", key="reg_pass")
                if st.button("Create account", type="primary"):
                    res = register_user(u, p, e)
                    if res.ok:
                        st.session_state.user_id = res.user_id
                        prof = get_user_profile(res.user_id or 0) or {}
                        st.session_state.username = prof.get("username") or get_username(res.user_id or 0)
                        st.session_state.user_email = prof.get("email") or ""
                        if st.session_state.messages:
                            st.session_state.offer_save_guest_chat = True
                        st.success("Account created.")
                        st.rerun()
                    else:
                        st.error(res.error or "Registration failed.")
        else:
            #profile updates
            prof = get_user_profile(st.session_state.user_id) or {}
            if prof.get("username"):
                st.session_state.username = prof.get("username")
            st.session_state.user_email = prof.get("email") if prof is not None else st.session_state.user_email

            st.success(f"Logged in as `{st.session_state.username or 'user'}`")
            if st.session_state.user_email:
                st.caption(f"Email: `{st.session_state.user_email}`")
            else:
                st.warning("No email on file yet — please add one in Account settings.")

            with st.expander("Account settings", expanded=False):
                st.subheader("Profile")
                new_u = st.text_input(
                    "Change username",
                    value=st.session_state.username or "",
                    key="acct_new_username",
                )
                if st.button("Update username", key="acct_update_username"):
                    r = update_username(int(st.session_state.user_id), new_u)
                    if r.ok:
                        st.session_state.username = new_u.strip().lower()
                        st.success("Username updated.")
                        st.rerun()
                    else:
                        st.error(r.error or "Could not update username.")

                new_e = st.text_input(
                    "Change email",
                    value=st.session_state.user_email or "",
                    key="acct_new_email",
                )
                if st.button("Update email", key="acct_update_email"):
                    r = update_email(int(st.session_state.user_id), new_e)
                    if r.ok:
                        st.session_state.user_email = new_e.strip().lower()
                        st.success("Email updated.")
                        st.rerun()
                    else:
                        st.error(r.error or "Could not update email.")

                st.divider()
                st.subheader("Password")
                cur_p = st.text_input("Current password", type="password", key="acct_cur_pass")
                new_p1 = st.text_input("New password", type="password", key="acct_new_pass")
                new_p2 = st.text_input("Confirm new password", type="password", key="acct_new_pass2")
                if st.button("Change password", key="acct_change_pass", type="primary"):
                    if new_p1 != new_p2:
                        st.error("New passwords do not match.")
                    else:
                        r = change_password(int(st.session_state.user_id), cur_p, new_p1)
                        if r.ok:
                            st.success("Password updated.")
                        else:
                            st.error(r.error or "Could not change password.")

            if st.button("Log out", type="secondary"):
                st.session_state.user_id = None
                st.session_state.username = None
                st.session_state.user_email = None
                _reset_chat()
                st.rerun()

        if st.session_state.get("force_new_conversation"):
            _reset_chat()
            st.session_state.loaded_session_id = None
            st.session_state.journal_pick = "(new conversation)"
            st.session_state.force_new_conversation = False
            st.rerun()

        if st.session_state.user_id and st.session_state.offer_save_guest_chat:
            st.info("Would you like to save your current guest conversation to this account?")
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Save conversation", type="primary"):
                    title = None
                    #first user message as a lightweight title
                    for m in st.session_state.messages:
                        if m.get("role") == "user" and m.get("content"):
                            t = str(m["content"]).strip()
                            title = t[:48] + ("…" if len(t) > 48 else "")
                            break
                    bulk_save_messages(
                        st.session_state.session_id,
                        st.session_state.user_id,
                        st.session_state.messages,
                        title=title,
                    )
                    st.session_state.loaded_session_id = st.session_state.session_id
                    st.session_state.offer_save_guest_chat = False
                    st.session_state.guest_unsaved_chat = False
                    st.success("Saved to your journal.")
                    st.rerun()
            with c2:
                if st.button("Don't save", type="secondary"):
                    st.session_state.offer_save_guest_chat = False
                    st.session_state.guest_unsaved_chat = False
                    st.rerun()

        st.divider()
        st.header("Settings")
        model = st.text_input("Ollama model", value=OLLAMA_MODEL)
        k = st.slider("Retrieval top-k", 1, 8, DEFAULT_RETRIEVAL_K)
        prompt_variant = st.selectbox(
            "Prompt style",
            options=["zero_shot", "few_shot", "chain_of_thought"],
            format_func=lambda x: {
                "zero_shot": "Zero-shot",
                "few_shot": "Few-shot",
                "chain_of_thought": "Chain-of-thought",
            }[x],
        )
        if st.button("Reset conversation", type="secondary"):
            _reset_chat()
            st.rerun()

        st.divider()
        st.markdown(
            """
<div style="display:flex; align-items:center; gap:8px;">
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path d="M7 3h11a2 2 0 0 1 2 2v16a2 2 0 0 1-2 2H7a3 3 0 0 1-3-3V6a3 3 0 0 1 3-3z" stroke="currentColor" stroke-width="1.5"/>
    <path d="M7 3v18" stroke="currentColor" stroke-width="1.5" opacity="0.6"/>
    <path d="M10 8h7M10 12h7M10 16h7" stroke="currentColor" stroke-width="1.5" opacity="0.7"/>
  </svg>
  <div style="font-weight: 800; font-size: 1.1rem;">Journal</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.user_id:
            #Ensure the current session exists in DB so title/summary/tags are editable immediately.
            try:
                ensure_session(
                    st.session_state.session_id,
                    st.session_state.user_id,
                    title="(new conversation)",
                )
            except Exception:
                pass

            sessions = list_sessions(st.session_state.user_id)
            options = ["(new conversation)"] + [
                f"{s['created_at']} — {s['title']} — {s['id']}" for s in sessions
            ]
            pick = st.selectbox("Open a past conversation", options=options, key="journal_pick")
            if pick != "(new conversation)":
                session_id = pick.split(" — ")[-1]
                if session_id and session_id != st.session_state.loaded_session_id:
                    msgs = load_messages(session_id, st.session_state.user_id)
                    st.session_state.messages = msgs
                    st.session_state.symptoms = []
                    st.session_state.session_id = session_id
                    st.session_state.loaded_session_id = session_id
                    st.rerun()

            meta_session_id = st.session_state.loaded_session_id or st.session_state.session_id
            if meta_session_id:
                #Use per-session widget keys so the sidebar refreshes when DB values change.
                _jk = str(meta_session_id)
                title_key = f"journal_title__{_jk}"
                summary_key = f"journal_summary__{_jk}"
                tags_key = f"journal_tags__{_jk}"
                save_title_key = f"journal_save_title__{_jk}"
                save_summary_key = f"journal_save_summary__{_jk}"
                save_tags_key = f"journal_save_tags__{_jk}"
                regen_key = f"journal_regen__{_jk}"

                #If we have pending regenerated values, apply them BEFORE widgets instantiate.
                pending = (st.session_state.get("pending_journal_prefill") or {}).get(_jk)
                if isinstance(pending, dict):
                    if pending.get("title") is not None:
                        st.session_state[title_key] = pending.get("title") or ""
                    if pending.get("summary") is not None:
                        st.session_state[summary_key] = pending.get("summary") or ""
                    if pending.get("tags") is not None:
                        st.session_state[tags_key] = pending.get("tags") or []
                    #clear once applied
                    try:
                        del st.session_state["pending_journal_prefill"][_jk]
                    except Exception:
                        pass

                meta = get_session_meta(meta_session_id, st.session_state.user_id) or {}
                title_val = meta.get("title") or ""
                summary_val = meta.get("summary") or ""
                tags_val = meta.get("tags") or []

                st.caption("Entry title")
                new_title = st.text_input("Title", value=title_val, key=title_key)
                if new_title != title_val and st.button("Save title", key=save_title_key):
                    set_session_meta(
                        meta_session_id,
                        st.session_state.user_id,
                        title=new_title.strip(),
                    )
                    st.rerun()

                st.caption("Summary (auto-generated, editable)")
                new_summary = st.text_area(
                    "Summary",
                    value=summary_val,
                    height=110,
                    key=summary_key,
                )
                if new_summary != summary_val and st.button("Save summary", key=save_summary_key):
                    set_session_meta(
                        meta_session_id,
                        st.session_state.user_id,
                        summary=new_summary.strip(),
                    )
                    st.rerun()

                st.caption("Triage tags mentioned (editable)")
                all_tags = ["routine", "monitor", "urgent", "emergency"]
                new_tags = st.multiselect(
                    "Tags",
                    options=all_tags,
                    default=[t for t in tags_val if t in all_tags],
                    key=tags_key,
                )
                if sorted(new_tags) != sorted([t for t in tags_val if t in all_tags]) and st.button("Save tags", key=save_tags_key):
                    set_session_meta(
                        meta_session_id,
                        st.session_state.user_id,
                        tags=sorted(new_tags),
                    )
                    st.rerun()

                if st.button("Generate summary + tags (LLM)", type="primary", key=regen_key):
                    try:
                        with st.spinner("Summarizing..."):
                            s = summarize_session(st.session_state.messages, model=model)
                        set_session_meta(
                            meta_session_id,
                            st.session_state.user_id,
                            title=s.get("title") or title_val,
                            summary=s.get("summary") or summary_val,
                            tags=s.get("triage_tags") or tags_val,
                        )
                        #Defer widget prefill until next rerun
                        st.session_state.setdefault("pending_journal_prefill", {})[_jk] = {
                            "title": s.get("title") or title_val,
                            "summary": s.get("summary") or summary_val,
                            "tags": [t for t in (s.get("triage_tags") or tags_val or []) if t in all_tags],
                        }
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not generate summary: {type(e).__name__}: {e}")
            if st.button("Start new conversation", type="secondary"):
                st.session_state.force_new_conversation = True
                st.rerun()
        else:
            st.caption("Guest mode: you can chat, but history is not saved.")

        st.divider()
        st.caption("Ollama base URL")
        st.code(OLLAMA_BASE_URL, language="text")
        ok = ollama_health(model)
        st.success("Ollama reachable") if ok else st.error("Cannot reach Ollama tags API — is `ollama serve` running?")

        st.divider()
        st.subheader("Tracked symptoms (this session)")
        if st.session_state.symptoms:
            for s in st.session_state.symptoms:
                st.write(f"- {s}")
        else:
            st.caption("None yet — describe how you feel in the chat.")

 #icon design for mochi
    st.markdown(
        """
<div style="display:flex; align-items:center; gap:10px; margin-bottom: 0.5rem;">
  <div class="mochiIcon">
    <div class="mochiFace">
      <div class="mochiEye left"><div class="mochiSparkle"></div></div>
      <div class="mochiEye right"><div class="mochiSparkle"></div></div>
      <div class="mochiBlush left"></div>
      <div class="mochiBlush right"></div>
    </div>
  </div>
  <div>
    <div style="font-size: 1.8rem; font-weight: 800;">Mochi: Your personal health assistant</div>
    <div style="color: rgba(17,24,39,0.70); margin-top: -2px;">Journal + grounded health info (educational prototype)</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "**Educational prototype — not medical advice.** "
        "Not a replacement for a clinician or emergency services. "
        "Answers use a small MedlinePlus-style knowledge file plus a local LLM; "
        "they may be wrong or incomplete. For emergencies, call your local emergency number."
    )

    if st.session_state.vs_error:
        st.error(f"Could not load RAG index: {st.session_state.vs_error}")
        return
    if not st.session_state.user_id:
        st.caption("You are chatting as **Guest**. Log in to save conversations to your journal.")

    #allow editing the most recent user message
    if "edit_idx" not in st.session_state:
        st.session_state.edit_idx = None
    if "edit_text" not in st.session_state:
        st.session_state.edit_text = ""

    last_user_idx = None
    for i in range(len(st.session_state.messages) - 1, -1, -1):
        if st.session_state.messages[i].get("role") == "user":
            last_user_idx = i
            break

    for i, m in enumerate(st.session_state.messages):
        role = m["role"]
        avatar = MOCHI_AVATAR if role == "assistant" else None
        with st.chat_message(role, avatar=avatar):
            st.markdown(m["content"], unsafe_allow_html=True)
            if m["role"] == "assistant" and m.get("sources"):
                with st.expander("Sources used"):
                    for d in m["sources"]:
                        st.markdown(
                            f"**{d['title']}** (`{d['id']}`) — triage tag in KB: `{d['triage_level']}`"
                        )
                        st.caption(d["snippet"][:400] + ("…" if len(d["snippet"]) > 400 else ""))
            if m.get("role") == "user":
                # Only allow editing the most recent user message
                is_last_user = (last_user_idx is not None and i == last_user_idx)
                if is_last_user and st.button("Edit", key=f"edit_btn_{i}"):
                    st.session_state.edit_idx = i
                    st.session_state.edit_text = m.get("content", "")
                    st.rerun()

    if st.session_state.edit_idx is not None and last_user_idx is not None:
        st.info("Editing last message. Saving will regenerate the assistant response.")
        st.session_state.edit_text = st.text_area(
            "Edit your message",
            value=st.session_state.edit_text,
            height=90,
            key="edit_box",
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            resend = st.button("Save + Resend", type="primary")
        with c2:
            cancel_edit = st.button("Cancel", type="secondary")
        if cancel_edit:
            st.session_state.edit_idx = None
            st.session_state.edit_text = ""
            st.rerun()
        if resend:
            idx = int(st.session_state.edit_idx)
            new_text = st.session_state.edit_text.strip()
            if new_text:
                st.session_state.messages[idx]["content"] = new_text
                #Remove everything after this user message 
                st.session_state.messages = st.session_state.messages[: idx + 1]
                #If logged in and message exists in database, update it and delete later rows.
                if st.session_state.user_id and st.session_state.session_id:
                    db_id = st.session_state.messages[idx].get("db_id")
                    if db_id:
                        update_message(int(db_id), new_text)
                        delete_messages_after(st.session_state.session_id, int(db_id))
                #Trigger regeneration on next rerun.
                st.session_state.regen_after_edit = True
                st.session_state.edit_idx = None
                st.session_state.edit_text = ""
                st.rerun()

    #Edit to regenerate assistant response
    if st.session_state.get("regen_after_edit") and st.session_state.messages:
        if st.session_state.messages[-1].get("role") == "user":
            user_text = st.session_state.messages[-1].get("content", "")
            prior_messages = st.session_state.messages[:-1]
            vs = st.session_state.vectorstore
            try:
                with st.chat_message("assistant", avatar=MOCHI_AVATAR):
                    placeholder = st.empty()
                    placeholder.markdown("**Mochi** is thinking…")
                    with st.spinner("Regenerating response..."):
                        result = run_turn(
                            vs,
                            user_text,
                            prior_messages,
                            prompt_variant=prompt_variant,
                            model=model,
                            retrieval_k=int(k),
                        )
            except Exception as e:
                st.error(f"The model request failed: {e}")
                st.session_state.regen_after_edit = False
                return

            sources = [
                {
                    "id": d.metadata.get("id", ""),
                    "title": d.metadata.get("title", ""),
                    "triage_level": d.metadata.get("triage_level", ""),
                    "snippet": d.page_content[:600],
                }
                for d in result.retrieved
            ]

            display = result.answer_text
            if result.triage_level:
                tri = result.triage_level
                display += "\n\n" + f"<span class='triage-badge triage-{tri}'>Triage: {tri}</span>"
                display += f"\n\n*Informational triage tag (not a diagnosis): **{tri}***"

            st.session_state.messages.append(
                {"role": "assistant", "content": display, "sources": sources}
            )
            if st.session_state.user_id:
                mid = append_message(st.session_state.session_id, "assistant", display)
                if mid:
                    st.session_state.messages[-1]["db_id"] = int(mid)

            st.session_state.regen_after_edit = False
            st.rerun()

    user_text = st.chat_input("Message Personal Health Assistant…")
    if not user_text:
        return

    allowed, err = check_rate_limit(
        st.session_state.rate_times,
        max_messages=RATE_LIMIT_MAX_MESSAGES,
        window_seconds=float(RATE_LIMIT_WINDOW_SECONDS),
    )
    if not allowed:
        st.warning(err)
        logger.warning(
            "rate_limited session_id=%s len_times=%s",
            st.session_state.session_id,
            len(st.session_state.rate_times),
        )
        return

    record_message_time(st.session_state.rate_times)
    st.session_state.messages.append({"role": "user", "content": user_text})
    if not st.session_state.user_id:
        st.session_state.guest_unsaved_chat = True
    prior_messages = st.session_state.messages[:-1]
    #immediate user message
    with st.chat_message("user"):
        st.markdown(user_text)

    #only when logged in
    if st.session_state.user_id:
        ensure_session(
            st.session_state.session_id,
            st.session_state.user_id,
            title=(user_text[:48] + ("…" if len(user_text) > 48 else "")),
        )
        mid = append_message(st.session_state.session_id, "user", user_text)
        if mid:
            st.session_state.messages[-1]["db_id"] = int(mid)

    vs = st.session_state.vectorstore
    t0 = time.perf_counter()
    try:
        with st.chat_message("assistant", avatar=MOCHI_AVATAR):
            placeholder = st.empty()
            placeholder.markdown("**Mochi** is thinking…")
            with st.spinner("Generating response..."):
                result = run_turn(
                    vs,
                    user_text,
                    prior_messages,
                    prompt_variant=prompt_variant,
                    model=model,
                    retrieval_k=int(k),
                )
    except Exception as e:
        logger.exception(
            "run_turn_failed session_id=%s prompt=%s",
            st.session_state.session_id,
            prompt_variant,
        )
        #region agent log
        agent_debug(
            "H6",
            "app.py:main:run_turn_exception",
            "run_turn_failed",
            {
                "exc_type": type(e).__name__,
                "exc_str": str(e)[:500],
                "model": model,
                "prompt_variant": prompt_variant,
                "user_len": len(user_text),
            },
        )
        #endregion agent log
        st.error(f"The model request failed: {e}")
        st.session_state.messages.pop()
        return

    elapsed = time.perf_counter() - t0
    for s in result.symptoms_this_turn:
        if s and s not in st.session_state.symptoms:
            st.session_state.symptoms.append(s)

    sources = [
        {
            "id": d.metadata.get("id", ""),
            "title": d.metadata.get("title", ""),
            "triage_level": d.metadata.get("triage_level", ""),
            "snippet": d.page_content[:600],
        }
        for d in result.retrieved
    ]

    display = result.answer_text
    if result.triage_level:
        tri = result.triage_level
        display += "\n\n" + f"<span class='triage-badge triage-{tri}'>Triage: {tri}</span>"
        display += f"\n\n*Informational triage tag (not a diagnosis): **{tri}***"

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": display,
            "sources": sources,
        }
    )
    if st.session_state.user_id:
        mid = append_message(st.session_state.session_id, "assistant", display)
        if mid:
            st.session_state.messages[-1]["db_id"] = int(mid)

    logger.info(
        "turn_ok session_id=%s prompt_variant=%s ms=%.0f triage=%s retrieval_ids=%s msg_len=%s",
        st.session_state.session_id,
        prompt_variant,
        elapsed * 1000,
        result.triage_level,
        [d.metadata.get("id") for d in result.retrieved],
        len(user_text),
    )

    st.rerun()


if __name__ == "__main__":
    main()
