"""
core/ingest.py
Clones a GitHub repository, validates it contains Python source, and
(re)populates the local ChromaDB collection via AST-based chunking.

Note: GitPython shells out to the system `git` binary -- it's a pure
Python *package*, but it still requires `git` to be installed on the
host. If you need zero non-Python system dependencies for cloning,
swap this for `dulwich` (a pure-Python git implementation) instead.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from git import GitCommandError, Repo

from core.chunker import chunk_file, discover_python_files

REPOS_DIR = Path("data/repos")
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "codebase"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"
BATCH_SIZE = 64


class InvalidRepoURLError(Exception):
    pass


class NoPythonFilesError(Exception):
    pass


class CloneError(Exception):
    pass


def _validate_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise InvalidRepoURLError(
            "Enter a valid repository URL, e.g. https://github.com/user/repo"
        )
    return url


def _local_dir_for(url: str) -> Path:
    repo_name = url.rstrip("/").split("/")[-1].removesuffix(".git")
    digest = hashlib.sha1(url.encode()).hexdigest()[:8]
    return REPOS_DIR / f"{repo_name}-{digest}"


def clone_repo(url: str) -> Path:
    url = _validate_url(url)
    dest = _local_dir_for(url)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        Repo.clone_from(url, dest, depth=1)  # shallow clone -- history isn't needed
    except GitCommandError as exc:
        raise CloneError(f"Could not clone '{url}': {exc}") from exc
    return dest


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection(client=None):
    client = client or get_chroma_client()
    embed_fn = OllamaEmbeddingFunction(
        url=f"{OLLAMA_BASE_URL}/api/embeddings",
        model_name=OLLAMA_EMBED_MODEL,
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    """Drop and recreate the collection. Used both by the 12h cleanup job
    and when swapping in a freshly submitted repository -- auron indexes
    one active repository at a time, so a new submission replaces the
    previous index rather than mixing chunks from two codebases."""
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # nothing to delete on first run
    get_collection(client)


def ingest_repository(url: str, progress: Optional[Callable[[str], None]] = None) -> dict:
    """
    Clone `url`, validate it contains Python source, chunk it via AST, and
    populate ChromaDB. Returns ingest stats. Raises InvalidRepoURLError /
    CloneError / NoPythonFilesError on failure.
    """
    def report(msg: str) -> None:
        if progress:
            progress(msg)

    report("Cloning repository...")
    repo_path = clone_repo(url)

    report("Scanning for Python files...")
    py_files = discover_python_files(repo_path)
    if not py_files:
        shutil.rmtree(repo_path, ignore_errors=True)
        raise NoPythonFilesError(
            "No .py files found in this repository. "
            "auron currently indexes Python codebases only."
        )

    report(f"Chunking {len(py_files)} files by function and class...")
    reset_collection()
    collection = get_collection()

    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_meta: list[dict] = []
    total_chunks = 0

    def flush():
        nonlocal batch_ids, batch_docs, batch_meta
        if batch_ids:
            collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)
            batch_ids, batch_docs, batch_meta = [], [], []

    for f in py_files:
        for chunk in chunk_file(f):
            if not chunk.content.strip():
                continue
            batch_ids.append(chunk.id)
            batch_docs.append(chunk.content)
            batch_meta.append({
                "file_path": str(f.relative_to(repo_path)),
                "node_type": chunk.node_type,
                "name": chunk.name,
                "parent_class": chunk.parent_class or "",
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            })
            total_chunks += 1
            if len(batch_ids) >= BATCH_SIZE:
                report(f"Embedding chunks locally via Ollama... ({total_chunks})")
                flush()
    flush()

    return {
        "repo_name": repo_path.name,
        "file_count": len(py_files),
        "chunk_count": total_chunks,
        "indexed_at": time.time(),
    }
