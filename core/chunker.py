"""
core/chunker.py
AST-based semantic chunking for Python source files. A function, class, or
method is the atomic chunk unit -- never a raw character-count slice -- so
retrieved context always contains a complete signature and body.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MAX_CHUNK_CHARS = 3000
IGNORE_DIRS = {".git", "venv", ".venv", "__pycache__", "node_modules", "dist", "build"}


@dataclass
class CodeChunk:
    id: str
    content: str
    file_path: str
    node_type: str          # "function" | "class" | "method" | "module_fallback"
    name: str
    parent_class: Optional[str]
    start_line: int
    end_line: int
    docstring: Optional[str] = None


def discover_python_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.py"):
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


class ASTChunker(ast.NodeVisitor):
    def __init__(self, source: str, file_path: str):
        self.source = source
        self.file_path = file_path
        self.chunks: list[CodeChunk] = []

    def _segment(self, node: ast.AST) -> str:
        return ast.get_source_segment(self.source, node) or ""

    def _chunk_id(self, name: str, lineno: int) -> str:
        raw = f"{self.file_path}:{name}:{lineno}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.chunks.append(CodeChunk(
            id=self._chunk_id(node.name, node.lineno),
            content=self._segment(node),
            file_path=self.file_path,
            node_type="class",
            name=node.name,
            parent_class=None,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
        ))
        # Index methods individually too, but keep them tagged with their
        # owning class -- don't generic_visit() or they'd also be picked
        # up as bare top-level functions and double-counted.
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.chunks.append(CodeChunk(
                    id=self._chunk_id(f"{node.name}.{child.name}", child.lineno),
                    content=self._segment(child),
                    file_path=self.file_path,
                    node_type="method",
                    name=child.name,
                    parent_class=node.name,
                    start_line=child.lineno,
                    end_line=child.end_lineno or child.lineno,
                    docstring=ast.get_docstring(child),
                ))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node)

    def _handle_function(self, node) -> None:
        self.chunks.append(CodeChunk(
            id=self._chunk_id(node.name, node.lineno),
            content=self._segment(node),
            file_path=self.file_path,
            node_type="function",
            name=node.name,
            parent_class=None,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
        ))


def _split_oversized(chunk: CodeChunk, overlap: int = 200) -> list[CodeChunk]:
    """Window a chunk that exceeds MAX_CHUNK_CHARS so it stays within the
    embedding model's effective context size."""
    parts = []
    text = chunk.content
    step = max(MAX_CHUNK_CHARS - overlap, 1)
    for i, start in enumerate(range(0, len(text), step)):
        parts.append(CodeChunk(
            id=f"{chunk.id}_{i}",
            content=text[start:start + MAX_CHUNK_CHARS],
            file_path=chunk.file_path,
            node_type=chunk.node_type,
            name=f"{chunk.name}[part {i}]",
            parent_class=chunk.parent_class,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            docstring=chunk.docstring,
        ))
    return parts


def chunk_file(path: Path) -> list[CodeChunk]:
    source = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return [CodeChunk(
            id=hashlib.sha1(str(path).encode()).hexdigest()[:16],
            content=source[:MAX_CHUNK_CHARS],
            file_path=str(path),
            node_type="module_fallback",
            name=path.stem,
            parent_class=None,
            start_line=1,
            end_line=len(source.splitlines()),
        )]

    chunker = ASTChunker(source, str(path))
    for node in tree.body:  # top-level only -- don't re-enter nested defs
        chunker.visit(node)

    final_chunks: list[CodeChunk] = []
    for c in chunker.chunks:
        if len(c.content) <= MAX_CHUNK_CHARS:
            final_chunks.append(c)
        else:
            final_chunks.extend(_split_oversized(c))
    return final_chunks
