"""
RAG-powered tools that drop into the existing StudyBuddy ActionRegistry.
Each function is self-contained and can be registered as an Action.
"""

import os
from typing import List
from litellm import completion

from rag.chunker import chunk_document
from rag.retriever import DualRetriever

# Singleton retriever — shared across all tool calls
_retriever: DualRetriever | None = None


def get_retriever() -> DualRetriever:
    global _retriever
    if _retriever is None:
        _retriever = DualRetriever()
    return _retriever


# ─── Tool: ingest a file or folder ────────────────────────────────────────

def ingest_material(path: str) -> str:
    """
    Ingests a file or all supported files in a folder into the vector store.
    Supports .py, .pdf, .txt, .md files.
    """
    import os
    from pathlib import Path

    retriever = get_retriever()
    target = Path(path)
    total_added = 0
    skipped = []

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = list(target.rglob("*.py")) + \
                list(target.rglob("*.pdf")) + \
                list(target.rglob("*.txt")) + \
                list(target.rglob("*.md"))
    else:
        return f"Path not found: {path}"

    for f in files:
        try:
            chunks = chunk_document(str(f))
            added = retriever.index_chunks(chunks)
            total_added += added
        except Exception as e:
            skipped.append(f"{f.name}: {e}")

    result = f"Indexed {total_added} chunks from {len(files)} file(s)."
    if skipped:
        result += f"\nSkipped: {', '.join(skipped)}"
    result += f"\nTotal chunks in store: {retriever.count()}"
    return result


# ─── Tool: answer a question using RAG ────────────────────────────────────

def rag_answer(question: str) -> str:
    """
    Retrieves relevant chunks from indexed material and generates
    a grounded answer. Shows sources so answers are verifiable.
    """
    retriever = get_retriever()

    if retriever.count() == 0:
        return (
            "No study material indexed yet. "
            "Use ingest_material to add your notes, PDFs, or code files first."
        )

    # Dual retrieval + RRF
    chunks = retriever.retrieve(question, top_k=5)

    if not chunks:
        return "No relevant material found for that question."

    # Build context block
    context_parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        source_label = meta.get("name") or meta.get("source", "unknown")
        context_parts.append(f"[{i+1}] {source_label}\n{chunk['content'][:600]}")

    context = "\n\n---\n\n".join(context_parts)

    # Generate grounded answer
    model = os.getenv("DEFAULT_MODEL", "groq/llama-3.1-8b-instant")
    messages = [
        {
            "role": "system",
            "content": (
                "You are StudyBuddy, an AI study assistant. "
                "Answer the student's question using ONLY the provided context. "
                "Be concise and cite the source numbers [1], [2] etc. "
                "If the answer isn't in the context, say so honestly."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    response = completion(model=model, messages=messages, max_tokens=1024)
    answer = response.choices[0].message.content

    # Append sources
    sources = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        name = meta.get("name", "")
        source = meta.get("source", "")
        lineno = meta.get("lineno", "")
        label = name if name else source
        if lineno:
            label += f" (line {lineno})"
        sources.append(f"[{i+1}] {label}")

    return answer + "\n\nSources:\n" + "\n".join(sources)


# ─── Tool: semantic quiz generation from material ──────────────────────────

def generate_quiz(topic: str, num_questions: int = 5) -> str:
    """
    Generates quiz questions from indexed material on a given topic.
    Uses RAG to ground questions in actual content — not generic templates.
    """
    retriever = get_retriever()
    model = os.getenv("DEFAULT_MODEL", "groq/llama-3.1-8b-instant")

    if retriever.count() == 0:
        # Fallback to LLM-only quiz if no material indexed
        messages = [
            {"role": "system", "content": "You are a quiz generator. Create clear, educational questions."},
            {"role": "user", "content": f"Generate {num_questions} quiz questions about {topic}. Format as a numbered list with answers."}
        ]
        response = completion(model=model, messages=messages, max_tokens=1024)
        return response.choices[0].message.content

    # Retrieve relevant content first
    chunks = retriever.retrieve(topic, top_k=5)
    context = "\n\n".join(c["content"][:400] for c in chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are StudyBuddy. Generate quiz questions grounded in the provided study material. "
                "Questions should test understanding, not just recall. "
                "Format: numbered questions followed by 'Answers:' section."
            )
        },
        {
            "role": "user",
            "content": f"Study material:\n{context}\n\nGenerate {num_questions} quiz questions about: {topic}"
        }
    ]

    response = completion(model=model, messages=messages, max_tokens=1024)
    return response.choices[0].message.content


# ─── Tool: summarize indexed material on a topic ──────────────────────────

def summarize_topic(topic: str) -> str:
    """
    Retrieves and summarizes indexed material on a given topic.
    Replaces the old fake text-truncation summarizer.
    """
    retriever = get_retriever()
    model = os.getenv("DEFAULT_MODEL", "groq/llama-3.1-8b-instant")

    if retriever.count() == 0:
        return "No material indexed yet. Please ingest your study files first."

    chunks = retriever.retrieve(topic, top_k=6)
    if not chunks:
        return f"No content found about '{topic}' in your study material."

    context = "\n\n".join(c["content"][:500] for c in chunks)

    messages = [
        {"role": "system", "content": "You are StudyBuddy. Summarize study material clearly and concisely for a student."},
        {"role": "user", "content": f"Summarize the following material about '{topic}':\n\n{context}"}
    ]

    response = completion(model=model, messages=messages, max_tokens=512)
    return response.choices[0].message.content


# ─── Tool: list indexed sources ───────────────────────────────────────────

def list_sources() -> str:
    """Shows all files currently indexed in the knowledge base."""
    retriever = get_retriever()
    sources = retriever.list_sources()
    if not sources:
        return "No material indexed yet."
    lines = [f"  {i+1}. {s}" for i, s in enumerate(sources)]
    return f"Indexed sources ({retriever.count()} total chunks):\n" + "\n".join(lines)
