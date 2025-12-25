from typing import List

def recall_at_k(contexts: List[str], ground_truth: str) -> float:
    gt = ground_truth.lower()
    for ctx in contexts:
        if gt in ctx.lower():
            return 1.0
    return 0.0

def answer_relevance(question: str, answer: str) -> float:
    if not answer:
        return 0.0
    answer_len = len(answer.split())
    return min(1.0, answer_len / 20)

def faithfulness(answer: str, contexts: List[str]) -> float:
    if not answer or not contexts:
        return 0.0
    context_text = " ".join(contexts).lower()
    answer_tokens = answer.lower().split()
    if not answer_tokens:
        return 0.0
    supported = sum(1 for t in answer_tokens if t in context_text)
    return supported / len(answer_tokens)
