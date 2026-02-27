"""
检索上限分析脚本
运行：python analyze_retrieval.py

分析：
1. Oracle recall：如果每步都选最优文档，recall@k 能到多少
2. DPR 单跳 recall@K：第一步就能召回 SF 的比例
3. DPR 两跳 recall@K：两步内能召回所有 SF 的比例
4. Context oracle：distractor 模式下 SF 是否一定在 top-10 context 里
"""

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_dev(path="data/hotpotqa/hotpot_dev_distractor_v1.json"):
    with open(path) as f:
        return json.load(f)


def analyze_context_coverage(dev):
    """distractor 模式：SF 是否都在给定的 10 篇 context 里"""
    print("=" * 60)
    print("1. Context Coverage Analysis (distractor mode)")
    print("=" * 60)
    
    sf_in_context = []
    for item in dev:
        sf_titles = set(sf[0] for sf in item["supporting_facts"])
        ctx_titles = set(t for t, _ in item["context"])
        coverage = len(sf_titles & ctx_titles) / len(sf_titles)
        sf_in_context.append(coverage)
    
    full_coverage = sum(1 for c in sf_in_context if c == 1.0)
    print(f"SF fully in context: {full_coverage}/{len(dev)} = {full_coverage/len(dev):.3f}")
    print(f"Avg SF coverage in context: {np.mean(sf_in_context):.3f}")
    print()


def analyze_bm25_recall(dev, index_dir="data/index", top_ks=[1, 2, 5, 10]):
    """BM25 检索的 recall"""
    print("=" * 60)
    print("2. BM25 Recall Analysis")
    print("=" * 60)
    
    try:
        with open(f"{index_dir}/bm25_index.pkl", "rb") as f:
            data = pickle.load(f)
        bm25 = data["bm25"]
        doc_titles = data["titles"]
    except FileNotFoundError:
        print("BM25 index not found, skipping")
        return

    recall_at_k = defaultdict(list)
    recall_both_at_k = defaultdict(list)  # 两篇 SF 都召回

    for item in dev[:500]:  # 采样500条
        question = item["question"]
        sf_titles = set(sf[0] for sf in item["supporting_facts"])

        tokens = question.lower().split()
        scores = bm25.get_scores(tokens)
        ranked = sorted(range(len(doc_titles)), key=lambda i: -scores[i])

        for k in top_ks:
            top_k_set = set(doc_titles[i] for i in ranked[:k])
            hit_count = len(top_k_set & sf_titles)
            recall_at_k[k].append(hit_count / len(sf_titles))
            recall_both_at_k[k].append(float(hit_count == len(sf_titles)))

    print(f"{'K':<6}", end="")
    for k in top_ks:
        print(f"recall@{k:<6}", end="")
    print(f"{'both@k':<8}" * len(top_ks))
    
    print(f"{'BM25':<6}", end="")
    for k in top_ks:
        print(f"{np.mean(recall_at_k[k]):<13.3f}", end="")
    print()
    print()


def analyze_dpr_recall(dev, index_dir="data/index", top_ks=[1, 2, 5, 10],
                        device="cuda", n_samples=500):
    """DPR 单次检索的 recall（用保存的 FAISS index）"""
    print("=" * 60)
    print("3. DPR Single-hop Recall (first retrieval step only)")
    print("=" * 60)

    try:
        import faiss
        import torch
        from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

        index = faiss.read_index(f"{index_dir}/dpr_index.faiss")
        with open(f"{index_dir}/dpr_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        doc_titles = meta["titles"]

        tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        encoder = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        ).to(device)
        encoder.eval()

    except Exception as e:
        print(f"DPR index load failed: {e}")
        return

    recall_at_k = defaultdict(list)
    recall_both_at_k = defaultdict(list)

    for item in dev[:n_samples]:
        sf_titles = set(sf[0] for sf in item["supporting_facts"])
        question = item["question"]

        enc = tokenizer(question, max_length=128, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = encoder(**enc).pooler_output.cpu().numpy().astype("float32")
        faiss.normalize_L2(emb)

        max_k = max(top_ks)
        _, indices = index.search(emb, max_k)
        retrieved = [doc_titles[i] for i in indices[0] if i >= 0]

        for k in top_ks:
            top_k_set = set(retrieved[:k])
            hit = len(top_k_set & sf_titles)
            recall_at_k[k].append(hit / len(sf_titles))
            recall_both_at_k[k].append(float(hit == len(sf_titles)))

    print(f"Single-hop retrieval (query=question only):")
    for k in top_ks:
        r = np.mean(recall_at_k[k])
        rb = np.mean(recall_both_at_k[k])
        print(f"  recall@{k}: {r:.3f}  (both SF recalled: {rb:.3f})")
    print()


def analyze_oracle_2hop(dev, index_dir="data/index", top_ks=[2, 5, 10],
                         device="cuda", n_samples=500):
    """
    Oracle 两跳：
    第1步用问题检索，第2步用"问题+第1步召回标题"检索
    假设每步都选了 SF 中的一篇（oracle 选择）
    """
    print("=" * 60)
    print("4. Oracle 2-hop Upper Bound")
    print("=" * 60)

    try:
        import faiss, torch
        from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

        index = faiss.read_index(f"{index_dir}/dpr_index.faiss")
        with open(f"{index_dir}/dpr_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        doc_titles = meta["titles"]
        doc_texts  = meta["texts"]
        title2text = dict(zip(doc_titles, doc_texts))

        tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        encoder = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        ).to(device)
        encoder.eval()
    except Exception as e:
        print(f"Failed: {e}")
        return

    both_recalled = []

    for item in dev[:n_samples]:
        sf_titles = list(set(sf[0] for sf in item["supporting_facts"]))
        question = item["question"]

        def retrieve(query, exclude=set(), k=10):
            enc = tokenizer(query, max_length=128, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = encoder(**enc).pooler_output.cpu().numpy().astype("float32")
            faiss.normalize_L2(emb)
            _, indices = index.search(emb, k + len(exclude))
            results = [doc_titles[i] for i in indices[0] if i >= 0 and doc_titles[i] not in exclude]
            return results[:k]

        # Hop 1
        hop1_results = retrieve(question, k=10)

        # Oracle: 选 hop1 中最好的那篇（如果有 SF 就选 SF，否则选 top-1）
        hop1_sf = [t for t in hop1_results if t in sf_titles]
        hop1_choice = hop1_sf[0] if hop1_sf else hop1_results[0]

        # Hop 2: query = question + hop1_choice title
        query2 = f"{question} {hop1_choice}"
        hop2_results = retrieve(query2, exclude={hop1_choice}, k=10)

        all_retrieved = set([hop1_choice] + hop2_results)
        recalled = set(sf_titles) & all_retrieved
        both_recalled.append(float(len(recalled) == len(sf_titles)))

    print(f"Oracle 2-hop (best possible with DPR, n={n_samples}):")
    print(f"  Both SF recalled: {np.mean(both_recalled):.3f}")
    print(f"  → This is the upper bound for RL with current DPR index")
    print()


def main():
    print("Loading dev data...")
    dev = load_dev()
    print(f"Dev size: {len(dev)}\n")

    analyze_context_coverage(dev)
    analyze_bm25_recall(dev)
    analyze_dpr_recall(dev, n_samples=500)
    analyze_oracle_2hop(dev, n_samples=500)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("If oracle 2-hop recall is low (<0.4), the DPR index itself is the bottleneck.")
    print("Consider: use a better DPR model, or switch to iterative DPR (IDRQA/MDR style).")


if __name__ == "__main__":
    main()
