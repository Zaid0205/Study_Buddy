"""
Retrieval evaluation script.
Measures hit@k — what percentage of the time does the correct
source appear in the top-k retrieved chunks.

Usage:
    python eval.py --k 3 --questions eval_questions.json

Output:
    hit@1: 0.65
    hit@3: 0.82     ← this number goes on your resume
    hit@5: 0.88
    MRR:   0.74     (Mean Reciprocal Rank)
"""

import json
import argparse
from typing import List, Dict
from rag.retriever import DualRetriever


# ─── Sample eval set ──────────────────────────────────────────────────────
# Format: list of {question, expected_source} dicts
# expected_source: substring of the correct chunk's source/name

SAMPLE_QUESTIONS = [
    {"question": "How do I add a task?", "expected_source": "add_task"},
    {"question": "How does the agent loop work?", "expected_source": "Agent"},
    {"question": "What tools are available?", "expected_source": "ActionRegistry"},
    {"question": "How is the quiz generated?", "expected_source": "generate_quiz"},
    {"question": "How does summarize work?", "expected_source": "summarize_topic"},
    {"question": "How is retrieval done?", "expected_source": "retrieve"},
    {"question": "What embedding model is used?", "expected_source": "EMBED_MODEL"},
    {"question": "How does BM25 search work?", "expected_source": "bm25_search"},
    {"question": "How is RRF fusion computed?", "expected_source": "reciprocal_rank"},
    {"question": "How are chunks indexed?", "expected_source": "index_chunks"},
]


def evaluate(questions: List[Dict], k_values=(1, 3, 5)) -> Dict:
    retriever = DualRetriever()

    if retriever.count() == 0:
        print("No material indexed. Run main.py and ingest some files first.")
        return {}

    hits = {k: 0 for k in k_values}
    reciprocal_ranks = []

    for item in questions:
        question = item["question"]
        expected = item["expected_source"].lower()

        results = retriever.retrieve(question, top_k=max(k_values))

        # Find rank of correct answer
        rank = None
        for i, chunk in enumerate(results):
            chunk_id = chunk["id"].lower()
            chunk_name = chunk["metadata"].get("name", "").lower()
            if expected in chunk_id or expected in chunk_name:
                rank = i + 1
                break

        if rank is not None:
            reciprocal_ranks.append(1 / rank)
            for k in k_values:
                if rank <= k:
                    hits[k] += 1
        else:
            reciprocal_ranks.append(0)

    n = len(questions)
    results_out = {}
    print(f"\nEvaluation results ({n} questions):")
    print("-" * 30)
    for k in k_values:
        score = hits[k] / n
        results_out[f"hit@{k}"] = round(score, 3)
        print(f"  hit@{k}: {score:.2%}")

    mrr = sum(reciprocal_ranks) / n
    results_out["MRR"] = round(mrr, 3)
    print(f"  MRR:   {mrr:.2%}")
    print("-" * 30)

    return results_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=str, default=None,
                        help="Path to JSON file with eval questions")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()

    if args.questions:
        with open(args.questions) as f:
            questions = json.load(f)
    else:
        print("Using built-in sample questions (index StudyBuddy's own source files first)")
        questions = SAMPLE_QUESTIONS

    evaluate(questions, k_values=tuple(args.k))
