# StudyBuddy v2 — RAG-Powered Agentic Study Assistant

An agentic AI study assistant that ingests your lecture notes, PDFs, and code files and answers questions grounded in your actual material — not hallucinated from training data.

## What makes this different

Most "chat with your docs" projects split text into fixed-size chunks and call it a day. StudyBuddy uses:

- **AST-aware chunking** — Python files are parsed at the function/class boundary level using Python's `ast` module. Each chunk carries the function name, docstring, and source location, making retrieval explainable.
- **Dual retrieval** — semantic search (ChromaDB + sentence-transformers) runs in parallel with BM25 keyword search. Semantic finds conceptually similar content; BM25 finds exact function/class names.
- **Reciprocal Rank Fusion** — results from both retrievers are merged using RRF, a well-studied fusion algorithm that requires no score normalization and consistently outperforms either retriever alone.
- **Grounded generation** — answers cite source numbers. The LLM is instructed to only use retrieved context, reducing hallucination.

## Architecture

```
Indexing pipeline (run once per file):
  File → AST parser / sentence splitter → Chunks
       → sentence-transformers → ChromaDB (semantic)
       → BM25Okapi              → BM25 store (keyword)

Query pipeline (every request):
  Question → LLM query rewrite
           → Semantic search (top-10) ─┐
           → BM25 search (top-10)      ├─ RRF fusion → top-5 chunks → LLM → Answer + sources
```

## Stack

- **LLM**: Groq (Llama-3.1-8b-instant) via LiteLLM
- **Embeddings**: sentence-transformers `all-MiniLM-L6-v2`
- **Vector store**: ChromaDB (persistent)
- **Keyword search**: rank-bm25
- **API**: FastAPI + Uvicorn
- **PDF support**: PyMuPDF

## Setup

```bash
# 1. Clone and install
git clone https://github.com/Zaid0205/Study_Buddy
cd Study_Buddy
pip install -r requirements.txt

# 2. Add your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (get one free at console.groq.com)

# 3. Run the CLI
python main.py

# 4. Or run the API server
uvicorn api.server:app --reload --port 8000
```

## Usage (CLI)

```
You: Ingest my notes: ./notes/
You: What is the attention mechanism?
You: Generate a 5-question quiz on transformers
You: Summarize backpropagation
You: Show my indexed sources
You: Add task: Review CNN architectures
```

## Usage (API)

```bash
# Index a file
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "./notes/"}'

# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is softmax?", "session_id": "abc123"}'

# List indexed sources
curl http://localhost:8000/sources
```

## Evaluation

```bash
# Run retrieval evaluation (hit@k and MRR)
python eval.py

# With your own questions
python eval.py --questions my_eval_set.json --k 1 3 5
```

## Project structure

```
StudyBuddy/
├── main.py              # Agent, ActionRegistry, entry point
├── eval.py              # Retrieval evaluation (hit@k, MRR)
├── requirements.txt
├── .env.example
├── rag/
│   ├── chunker.py       # AST chunker (code) + sentence chunker (text/PDF)
│   └── retriever.py     # DualRetriever: ChromaDB + BM25 + RRF fusion
├── tools/
│   └── rag_tools.py     # RAG-powered tool functions for the agent
└── api/
    └── server.py        # FastAPI REST API with session management
```
