"""
Baseline: BM25 / Dense retriever + Reader (no RL)
Greedy multi-hop: retrieve top-K, take top-1, repeat for max_hops steps
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from environment import exact_match_score, f1_score

logger = logging.getLogger(__name__)


def run_baseline(
    retriever,
    reader,
    dataset,
    evaluator,
    top_k: int = 10,
    max_hops: int = 3,
    max_samples: Optional[int] = None,
) -> Dict:
    """
    Greedy iterative retrieval baseline.
    At each hop: query = question + titles of selected docs so far.
    Select top-1 document. After max_hops, read answer.
    """
    data = dataset.data
    if max_samples is not None:
        import random
        data = random.sample(data, min(max_samples, len(data)))

    all_metrics = defaultdict(list)

    for item in data:
        result = _baseline_single(item, retriever, reader, top_k, max_hops, evaluator.recall_k_list)
        for k, v in result.items():
            all_metrics[k].append(v)

    return {k: float(np.mean(vs)) for k, vs in all_metrics.items()}


def _baseline_single(item, retriever, reader, top_k, max_hops, recall_k_list) -> Dict:
    question = item["question"]
    gold_answer = item["answer"]
    gold_titles = set(sf[0] for sf in item.get("supporting_facts", []))

    selected_titles = []
    selected_docs = []
    retrieved_in_order = []

    for hop in range(max_hops):
        # Build query
        if selected_titles:
            query = f"{question} {selected_titles[-1]}"
        else:
            query = question

        candidates = retriever.retrieve(
            query=query,
            top_k=top_k,
            exclude_titles=selected_titles
        )
        if not candidates:
            break

        # Take top-1 greedily
        top_doc = candidates[0]
        selected_docs.append(top_doc)
        selected_titles.append(top_doc["title"])
        retrieved_in_order.append(top_doc["title"])

    # Read answer
    predicted_answer = reader.predict(question, selected_docs)
    em = exact_match_score(predicted_answer, gold_answer)
    f1 = f1_score(predicted_answer, gold_answer)

    # Supporting fact metrics
    selected_set = set(selected_titles)
    if gold_titles:
        sf_prec = len(selected_set & gold_titles) / max(len(selected_set), 1)
        sf_rec = len(selected_set & gold_titles) / len(gold_titles)
        sf_f1 = 2 * sf_prec * sf_rec / (sf_prec + sf_rec + 1e-9)
    else:
        sf_prec = sf_f1 = sf_rec = 0.0

    # Recall@k
    recall_at_k = {}
    for k in recall_k_list:
        top_k_set = set(retrieved_in_order[:k])
        recall_at_k[f"recall@{k}"] = float(gold_titles.issubset(top_k_set)) if gold_titles else 0.0

    return {
        "em": em,
        "f1": f1,
        "sf_f1": sf_f1,
        "sf_recall": sf_rec,
        "n_hops": len(selected_docs),
        **recall_at_k,
    }
