"""
Smart chunker: AST-based for code files, sentence-boundary for text/PDF.
This is the core differentiator — chunks respect logical boundaries,
not arbitrary token windows.
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Any


# ─── Text / PDF chunker ────────────────────────────────────────────────────

def chunk_text(text: str, source: str, chunk_size: int = 400, overlap: int = 80) -> List[Dict]:
    """
    Splits text at sentence boundaries within a target chunk size.
    Overlap ensures context isn't lost at chunk edges.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current, current_len = [], [], 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_len + word_count > chunk_size and current:
            chunk_text_str = " ".join(current)
            chunks.append({
                "id": f"{source}::chunk_{len(chunks)}",
                "source": source,
                "type": "text",
                "content": chunk_text_str,
                "metadata": {"char_count": len(chunk_text_str)}
            })
            # keep overlap words for next chunk
            overlap_words = " ".join(current).split()[-overlap:]
            current = [" ".join(overlap_words)]
            current_len = len(overlap_words)

        current.append(sentence)
        current_len += word_count

    if current:
        chunk_text_str = " ".join(current)
        chunks.append({
            "id": f"{source}::chunk_{len(chunks)}",
            "source": source,
            "type": "text",
            "content": chunk_text_str,
            "metadata": {"char_count": len(chunk_text_str)}
        })

    return chunks


# ─── AST-based code chunker ────────────────────────────────────────────────

def chunk_code(source_path: str) -> List[Dict]:
    """
    Parses Python files with the ast module and extracts functions/classes
    as individual chunks. Each chunk includes name, docstring, line number,
    and full source — making retrieval explainable and precise.
    """
    path = Path(source_path)
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        # fallback to plain text chunking if parse fails
        return chunk_text(path.read_text(errors="ignore"), source_path)

    chunks = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        code_segment = ast.get_source_segment(source, node)
        if not code_segment:
            continue

        docstring = ast.get_docstring(node) or ""
        node_type = type(node).__name__

        # Build a rich content string for embedding — name + docstring + code
        # This makes semantic search far more accurate than embedding raw code alone
        content = f"{node_type}: {node.name}\n"
        if docstring:
            content += f"Description: {docstring}\n"
        content += f"\n{code_segment}"

        chunks.append({
            "id": f"{source_path}::{node.name}::{node.lineno}",
            "source": source_path,
            "type": node_type,
            "name": node.name,
            "lineno": node.lineno,
            "content": content,
            "raw_code": code_segment,
            "docstring": docstring,
            "metadata": {
                "file": path.name,
                "node_type": node_type,
                "lineno": node.lineno,
            }
        })

    # If no functions/classes found, fall back to text chunking
    if not chunks:
        return chunk_text(source, source_path)

    return chunks


# ─── Router ───────────────────────────────────────────────────────────────

def chunk_document(filepath: str) -> List[Dict]:
    """
    Auto-detects file type and routes to the right chunker.
    Extend this with PDF support via PyMuPDF as needed.
    """
    ext = Path(filepath).suffix.lower()

    if ext == ".py":
        return chunk_code(filepath)

    elif ext == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(filepath)
            full_text = "\n".join(page.get_text() for page in doc)
            return chunk_text(full_text, filepath)
        except ImportError:
            raise ImportError("Install PyMuPDF: pip install pymupdf")

    elif ext in (".txt", ".md", ".rst"):
        text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
        return chunk_text(text, filepath)

    else:
        # Try as plain text
        try:
            text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
            return chunk_text(text, filepath)
        except Exception:
            raise ValueError(f"Unsupported file type: {ext}")
