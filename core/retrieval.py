"""
core/retrieval.py
Query-time retrieval against ChromaDB and streaming generation via Ollama.
"""

from __future__ import annotations

from typing import Generator, Iterable, Optional

import ollama

from core.ingest import get_collection

OLLAMA_CHAT_MODEL = "llama3"
TOP_K = 6

SYSTEM_PROMPT = (
    "You are auron, a local assistant that answers questions about the "
    "indexed codebase using only the provided context. Cite the relevant "
    "file and line range for every claim, like (file.py:12-30). If the "
    "context does not contain the answer, say so plainly instead of "
    "guessing."
)


def retrieve_context(question: str, k: int = TOP_K) -> list[dict]:
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return [{"content": doc, **meta} for doc, meta in zip(docs, metas)]


def _format_context(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        loc = f"{h['file_path']}:{h['start_line']}-{h['end_line']}"
        blocks.append(f"# {loc} ({h['node_type']} {h['name']})\n{h['content']}")
    return "\n\n".join(blocks)


def _extract_delta(chunk) -> str:
    """Pull the content delta out of a streamed chat chunk regardless of
    whether the installed `ollama` version returns plain dicts or
    attribute-style response objects."""
    message = chunk.get("message") if isinstance(chunk, dict) else getattr(chunk, "message", None)
    if message is None:
        return ""
    if isinstance(message, dict):
        return message.get("content", "") or ""
    return getattr(message, "content", "") or ""


def stream_answer(
    question: str,
    history: Optional[Iterable[dict]] = None,
    model: str = OLLAMA_CHAT_MODEL,
) -> Generator[str, None, None]:
    """Retrieve context for `question`, build a grounded prompt, and yield
    the model's response incrementally -- pass straight to st.write_stream."""
    hits = retrieve_context(question)
    context = _format_context(hits) if hits else "(no matching code found)"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"Context from the codebase:\n\n{context}\n\nQuestion: {question}",
    })

    for chunk in ollama.chat(model=model, messages=messages, stream=True):
        delta = _extract_delta(chunk)
        if delta:
            yield delta
