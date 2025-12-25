import json
import os
import random
import re
import pickle
import sys
from pathlib import Path
from pprint import pprint
from statistics import mean

from dotenv import load_dotenv

# ---- Paths / environment ----
# Repo root: .../LLM_Team
REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure imports like `app.*` work when running this file directly
sys.path.insert(0, str(REPO_ROOT))

# Load environment variables from repo root `.env`
load_dotenv(dotenv_path=REPO_ROOT / ".env")

from app.backend.core.rag_graph import run_rag
from evaluation.metrics import recall_at_k, answer_relevance, faithfulness

DATASET_PATH = "evaluation/datasets/validation.json"
RESULTS_PATH = "evaluation/results/results.json"

QUESTION_TEMPLATES = [
    "О чём говорится в этом отрывке?",
    "Что здесь описывается?",
    "Какую ситуацию описывает текст?",
    "О каких событиях идёт речь?",
    "Что можно сказать по этому фрагменту?",
    "Какие действия совершают персонажи?",
    "Что является основной мыслью отрывка?",
    "Какой смысл заложен в этом тексте?"
]


def _load_index_chunks(store_pkl_path: Path):
    """Load chunks (+ optional metadata) from the FAISS sidecar pickle."""
    if not store_pkl_path.exists():
        return [], []
    with open(store_pkl_path, "rb") as f:
        meta = pickle.load(f)
    chunks = meta.get("chunks", []) or []
    metadata = meta.get("metadata", []) or []
    return chunks, metadata


def _pick_quote_from_chunk(text: str, min_len: int = 30, max_len: int = 120) -> str:
    """Pick a short quote from a chunk to make the question retrievable."""
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) <= max_len:
        return t

    # Try to take a sentence-like piece
    parts = re.split(r"(?<=[\.!\?…])\s+", t)
    parts = [p.strip() for p in parts if len(p.strip()) >= min_len]
    if parts:
        cand = random.choice(parts)
        return cand[:max_len].strip()

    # Fallback: random window
    start = random.randint(0, max(0, len(t) - max_len))
    return t[start : start + max_len].strip()


def generate_validation_dataset(num_samples: int = 100):
    # Prefer sampling from the built index so questions match the corpus.
    store_pkl = REPO_ROOT / "data" / "indexes" / "store.pkl"
    chunks, metadata = _load_index_chunks(store_pkl)

    # Fallback (если индекс ещё не собран)
    fallback_texts = [
        "лишь на одной руке",
        "Кольцом, а все загадочные подгорные тайны на поверку оказались ничего не стоящими",
        "49 считался среди них самым сведущим",
        "ни страха, ни поклонения нет в моих чувствах",
        "падалью и наживается на чужой беде",
        "старые легенды об энтах оказались правдой",
        "он вдруг запнулся и смолк",
        "выбросить такое кольцо оказалось невозможно",
        "они были далеко от дома и очень устали",
        "Смеагорл ничего не видел",
    ]

    dataset = []
    for i in range(num_samples):
        if chunks:
            idx = random.randrange(0, len(chunks))
            chunk_text = chunks[idx]
            quote = _pick_quote_from_chunk(chunk_text)
            meta = metadata[idx] if idx < len(metadata) else {}

            # Make a concrete, retrievable question. We keep it simple and uniform.
            # The quote acts as the retrieval anchor; the task is to explain context.
            book = meta.get("book_name") or meta.get("filename") or "книги"
            page = meta.get("page_number")
            loc = f" ({book}, стр. {page})" if page is not None else f" ({book})"

            question = (
                f"Найди в тексте фрагмент, где встречается фраза: «{quote}». "
                f"Кратко опиши, что происходит вокруг этой фразы{loc}. #{i+1}"
            )
            ground_truth = quote
        else:
            ground_truth = random.choice(fallback_texts)
            question = (
                f"Найди в тексте фрагмент, где встречается фраза: «{ground_truth}». "
                f"Кратко опиши, что происходит вокруг этой фразы. #{i+1}"
            )

        dataset.append({
            "question": question,
            "ground_truth": ground_truth,
        })

    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Validation dataset generated with {len(dataset)} examples.")


def main():
    if not os.path.exists(DATASET_PATH) or os.path.getsize(DATASET_PATH) == 0:
        print("Validation dataset not found → generating automatically")
        generate_validation_dataset(num_samples=100)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    for sample in dataset:
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        print("\n==============================")
        print("QUESTION:", question)

        state = run_rag(question, top_k=12)
        answer = state.get("answer", "")
        passages = state.get("passages", [])

        contexts = [p[0] if isinstance(p, tuple) else str(p) for p in passages]

        metrics = {
            "question": question,
            "answer": answer,
            "recall@k": recall_at_k(contexts, ground_truth),
            "answer_relevance": answer_relevance(question, answer),
            "faithfulness": faithfulness(answer, contexts),
        }

        pprint(metrics)
        results.append(metrics)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n========== AVERAGE METRICS ==========")
    print("Recall@K:", mean(r["recall@k"] for r in results))
    print("Answer relevance:", mean(r["answer_relevance"] for r in results))
    print("Faithfulness:", mean(r["faithfulness"] for r in results))


if __name__ == "__main__":
    main()
