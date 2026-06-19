"""
app.py
auron -- a local, code-RAG chat assistant.

This file is UI orchestration only; all RAG logic lives in core/. The CSS
block below is a plain Python string passed to st.markdown -- there are no
separate .html/.js/.css files anywhere in this project. It exists purely
to fade the two UI states in smoothly; Streamlit's rerun-the-whole-script
execution model doesn't give true persistent animation between states the
way a JS framework would, so this is a deliberately modest fade-in, not a
continuous transition.
"""

from __future__ import annotations

import streamlit as st

from core.cleanup import start_cleanup_scheduler
from core.ingest import (
    CloneError,
    InvalidRepoURLError,
    NoPythonFilesError,
    ingest_repository,
)
from core.retrieval import stream_answer

st.set_page_config(page_title="auron", page_icon="🔎", layout="centered")

# -- One-time, process-wide setup --------------------------------------
# st.cache_resource guarantees this body runs exactly once per server
# process, regardless of how many browser sessions connect or how many
# times the script reruns on user interaction. Without it, a scheduler
# would be (re)started on every rerun.
@st.cache_resource
def _scheduler():
    return start_cleanup_scheduler()


_scheduler()

st.markdown(
    """
    <style>
    @keyframes fade-in {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .auron-fade { animation: fade-in 0.5s ease-out; }
    .auron-heading { font-size: 3rem; font-weight: 700; letter-spacing: -0.04em; margin-bottom: 0; }
    .auron-heading-small { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.03em; margin-bottom: 0; }
    .auron-tagline { color: var(--text-color-secondary, #888); margin-top: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -- Session state defaults ----------------------------------------------
st.session_state.setdefault("indexed", False)
st.session_state.setdefault("repo_stats", None)
st.session_state.setdefault("messages", [])


def reset_app() -> None:
    st.session_state.indexed = False
    st.session_state.repo_stats = None
    st.session_state.messages = []


# ==========================================================================
# State 1 -- repository submission
# ==========================================================================
if not st.session_state.indexed:
    st.markdown('<p class="auron-heading auron-fade">auron</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="auron-tagline auron-fade">A local assistant for understanding your codebase</p>',
        unsafe_allow_html=True,
    )
    st.write("")

    with st.form("repo_form"):
        repo_url = st.text_input(
            "Enter GitHub Repository URL",
            placeholder="https://github.com/user/repo",
        )
        submitted = st.form_submit_button("Index repository", use_container_width=True)

    if submitted:
        if not repo_url.strip():
            st.error("Enter a repository URL first.")
            st.stop()

        status_box = st.status("Indexing repository...", expanded=True)
        try:
            stats = ingest_repository(
                repo_url, progress=lambda msg: status_box.update(label=msg)
            )
        except InvalidRepoURLError as exc:
            status_box.update(label="Invalid URL", state="error")
            st.error(str(exc))
            st.stop()
        except CloneError as exc:
            status_box.update(label="Clone failed", state="error")
            st.error(str(exc))
            st.stop()
        except NoPythonFilesError as exc:
            # Strict check from the spec: halt cleanly, no partial index.
            status_box.update(label="No Python files found", state="error")
            st.error(str(exc))
            st.stop()
        else:
            status_box.update(label="Indexing complete", state="complete")
            st.session_state.indexed = True
            st.session_state.repo_stats = stats
            st.rerun()

# ==========================================================================
# State 2 -- chat over the indexed repository
# ==========================================================================
else:
    stats = st.session_state.repo_stats
    header_col, reset_col = st.columns([5, 1])
    with header_col:
        st.markdown(
            '<p class="auron-heading-small auron-fade">auron</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Indexed `{stats['repo_name']}` -- "
            f"{stats['file_count']} files, {stats['chunk_count']} chunks"
        )
    with reset_col:
        st.button("New repo", on_click=reset_app, use_container_width=True)

    st.markdown('<div class="auron-fade">', unsafe_allow_html=True)
    st.subheader("Write your question about this repository")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask about the indexed code...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if m["role"] in ("user", "assistant")
            ]
            answer = st.write_stream(stream_answer(question, history=history))

        st.session_state.messages.append({"role": "assistant", "content": answer})

    st.markdown("</div>", unsafe_allow_html=True)
