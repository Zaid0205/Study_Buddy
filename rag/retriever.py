"""
Dual retrieval: semantic (ChromaDB) + keyword (BM25), fused with
Reciprocal Rank Fusion (RRF).

Why dual retrieval?
- Semantic search finds conceptually similar chunks even with different wording
- BM25 finds exact function/class names the user types
- RRF combines both rankings without needing score normalization
"""

import json
from typing import List, Dict, Any
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi


CHROMA_PATH = "./study_buddy_db"
COLLECTION_NAME = "study_material"
EMBED_MODEL = "all-MiniLM-L6-v2"  # fast, free, good quality


class DualRetriever:

    def __init__(self):
        self.embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )

        # BM25 is rebuilt in memory from stored chunks on each init
        # For production: persist the BM25 corpus to disk with pickle
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: List[Dict] = []
        self._rebuild_bm25()

    # ─── Indexing ──────────────────────────────────────────────────────────

    def index_chunks(self, chunks: List[Dict]) -> int:
        """
        Adds chunks to ChromaDB and rebuilds BM25 index.
        Skips duplicates by ID.
        Returns count of newly added chunks.
        """
        existing_ids = set(self.collection.get()["ids"])
        new_chunks = [c for c in chunks if c["id"] not in existing_ids]

        if not new_chunks:
            return 0

        self.collection.add(
            ids=[c["id"] for c in new_chunks],
            documents=[c["content"] for c in new_chunks],
            metadatas=[{
                "source": c["source"],
                "type": c["type"],
                "name": c.get("name", ""),
                "lineno": str(c.get("lineno", "")),
                "raw_code": c.get("raw_code", ""),
            } for c in new_chunks]
        )

        self._rebuild_bm25()
        return len(new_chunks)

    def _rebuild_bm25(self):
        """Rebuilds in-memory BM25 index from all stored ChromaDB documents."""
        result = self.collection.get()
        if not result["ids"]:
            self._bm25 = None
            self._bm25_docs = []
            return

        self._bm25_docs = [
            {"id": id_, "content": doc, "metadata": meta}
            for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
        ]
        tokenized = [doc["content"].lower().split() for doc in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized)

    # ─── Retrieval ─────────────────────────────────────────────────────────

    def semantic_search(self, query: str, k: int = 8) -> List[Dict]:
        """Dense retrieval via cosine similarity on embeddings."""
        if self.collection.count() == 0:
            return []
        results = self.collection.query(query_texts=[query], n_results=min(k, self.collection.count()))
        docs = []
        for i, id_ in enumerate(results["ids"][0]):
            docs.append({
                "id": id_,
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # cosine sim
            })
        return docs

    def bm25_search(self, query: str, k: int = 8) -> List[Dict]:
        """Sparse retrieval via BM25 — great for exact function/class names."""
        if not self._bm25:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {**self._bm25_docs[i], "bm25_score": scores[i]}
            for i in top_indices if scores[i] > 0
        ]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Dual retrieval + Reciprocal Rank Fusion.

        RRF formula: score(d) = Σ 1 / (k + rank(d))
        where k=60 is a smoothing constant. No score normalization needed —
        works across heterogeneous retrieval systems naturally.
        """
        semantic = self.semantic_search(query, k=10)
        bm25 = self.bm25_search(query, k=10)

        # RRF fusion
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}

        K = 60
        for rank, doc in enumerate(semantic):
            rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0) + 1 / (K + rank + 1)
            doc_map[doc["id"]] = doc

        for rank, doc in enumerate(bm25):
            rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0) + 1 / (K + rank + 1)
            doc_map[doc["id"]] = doc

        ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        results = []
        for id_ in ranked_ids:
            doc = doc_map[id_]
            doc["rrf_score"] = rrf_scores[id_]
            results.append(doc)

        return results

    # ─── Inspection helpers ────────────────────────────────────────────────

    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> List[str]:
        result = self.collection.get()
        sources = {m["source"] for m in result["metadatas"]}
        return sorted(sources)

    def delete_source(self, source: str):
        """Remove all chunks from a specific file."""
        result = self.collection.get(where={"source": source})
        if result["ids"]:
            self.collection.delete(ids=result["ids"])
            self._rebuild_bm25()
