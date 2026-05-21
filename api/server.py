"""
FastAPI REST API for StudyBuddy v2.
Exposes the agent as a stateless HTTP service with session management.

Endpoints:
  POST /ingest          — index a file or folder
  POST /chat            — send a message to the agent
  GET  /sources         — list indexed files
  GET  /health          — health check
  DELETE /sources/{src} — remove a source
"""

import os
import uuid
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from main import Agent, build_registry, Goal
from rag.retriever import DualRetriever
from tools.rag_tools import ingest_material, list_sources, get_retriever


# ─── Session store (in-memory; use Redis for production) ──────────────────

sessions: Dict[str, Agent] = {}


def get_or_create_session(session_id: str) -> Agent:
    if session_id not in sessions:
        goals = [
            Goal(1, "Answer from material", "Use RAG to answer from indexed files"),
            Goal(2, "Generate content", "Create grounded quizzes and summaries"),
            Goal(3, "Manage tasks", "Track study tasks"),
        ]
        sessions[session_id] = Agent(goals, build_registry())
    return sessions[session_id]


# ─── App setup ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the retriever on startup
    get_retriever()
    yield

app = FastAPI(
    title="StudyBuddy v2",
    description="RAG-powered study assistant with dual retrieval and AST-aware chunking",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response models ────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    response: str

class IngestRequest(BaseModel):
    path: str

class IngestResponse(BaseModel):
    result: str
    chunks_total: int


# ─── Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    retriever = get_retriever()
    return {
        "status": "ok",
        "chunks_indexed": retriever.count(),
        "active_sessions": len(sessions)
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """Index a file or directory into the knowledge base."""
    try:
        result = ingest_material(req.path)
        retriever = get_retriever()
        return IngestResponse(result=result, chunks_total=retriever.count())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Send a message to the StudyBuddy agent.
    Pass the same session_id to maintain conversation history.
    """
    session_id = req.session_id or str(uuid.uuid4())
    agent = get_or_create_session(session_id)

    # Capture printed output from agent.run()
    import io, sys
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer

    try:
        agent.run(req.message)
    finally:
        sys.stdout = old_stdout

    output = buffer.getvalue()

    # Extract the StudyBuddy response line
    response_lines = [
        line.replace("StudyBuddy: ", "").strip()
        for line in output.split("\n")
        if line.startswith("StudyBuddy:")
    ]
    response = response_lines[-1] if response_lines else output.strip() or "Done."

    return ChatResponse(session_id=session_id, response=response)


@app.get("/sources")
def sources():
    """List all indexed sources and chunk count."""
    retriever = get_retriever()
    return {
        "sources": retriever.list_sources(),
        "total_chunks": retriever.count()
    }


@app.delete("/sources/{source_name}")
def delete_source(source_name: str):
    """Remove all chunks from a specific source."""
    retriever = get_retriever()
    retriever.delete_source(source_name)
    return {"deleted": source_name, "chunks_remaining": retriever.count()}
