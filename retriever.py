"""
Retriever 实现：
- BM25Retriever     : BM25 稀疏检索（主力，recall 最强）
- DenseRetriever    : DPR 稠密检索（备用/对比）
- HybridRetriever   : BM25 + DPR 线性融合（RRF），用于第一跳
- MDRRetriever      : 多跳感知检索，第二跳用"问题+已选文档"联合查询

数据诊断结论：
  BM25 recall@10 = 0.510  >  DPR recall@10 = 0.294
  DPR oracle 2-hop = 0.082（太低，不能作为主检索器）
→ 以 BM25 为主检索器，DPR 辅助重排
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BM25 Retriever（主力）
# ─────────────────────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    BM25 稀疏检索。recall@10=0.510，作为主检索器。
    多跳时查询改写：q2 = "question title_of_hop1"
    """

    def __init__(self, index_dir: str = "data/index"):
        self.index_dir = Path(index_dir)
        self._bm25 = None
        self.doc_titles: List[str] = []
        self.doc_texts:  List[str] = []

    def build_index(self, corpus: Dict[str, str]):
        cache_path = self.index_dir / "bm25_index.pkl"
        if cache_path.exists():
            logger.info("Loading BM25 index from cache")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self._bm25      = data["bm25"]
            self.doc_titles = data["titles"]
            self.doc_texts  = data["texts"]
            return

        from rank_bm25 import BM25Okapi
        logger.info(f"Building BM25 index for {len(corpus)} docs")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.doc_titles = list(corpus.keys())
        self.doc_texts  = [corpus[t] for t in self.doc_titles]
        tokenized = [text.lower().split() for text in self.doc_texts]
        self._bm25 = BM25Okapi(tokenized)

        with open(cache_path, "wb") as f:
            pickle.dump({"bm25": self._bm25,
                         "titles": self.doc_titles,
                         "texts":  self.doc_texts}, f)
        logger.info("BM25 index built")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        exclude_titles: Optional[List[str]] = None,
    ) -> List[Dict]:
        assert self._bm25 is not None, "Call build_index() first"
        tokens   = query.lower().split()
        scores   = self._bm25.get_scores(tokens)
        excluded = set(exclude_titles or [])

        ranked = sorted(
            [(scores[i], i) for i in range(len(self.doc_titles))
             if self.doc_titles[i] not in excluded],
            reverse=True,
        )
        results = []
        for score, idx in ranked[:top_k]:
            results.append({
                "title": self.doc_titles[idx],
                "text":  self.doc_texts[idx],
                "score": float(score),
                "idx":   int(idx),
            })
        return results

    def encode_documents(self, docs: List[Dict]) -> torch.Tensor:
        """
        BM25 没有 embedding，返回随机向量占位。
        策略网络的 encode_documents 调用需要 tensor，
        实际打分由 policy 的 CrossAttention 完成（依赖 state_vec）。
        使用 DenseRetriever.encode_documents 效果更好，
        HybridRetriever 会覆盖此方法。
        """
        raise NotImplementedError(
            "BM25Retriever 没有 embedding，请用 HybridRetriever 或 DenseRetriever.encode_documents"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dense Retriever（DPR，用于文档 embedding）
# ─────────────────────────────────────────────────────────────────────────────

class DenseRetriever:
    """
    DPR 稠密检索。recall@10=0.294，单独使用效果差。
    主要用途：encode_documents() 给策略网络提供语义向量。
    """

    def __init__(
        self,
        model_name: str = "facebook/dpr-question_encoder-single-nq-base",
        index_dir:  str = "data/index",
        device:     str = "cpu",
        cache_dir:  str = "cache",
        batch_size: int = 64,
    ):
        self.device     = device
        self.index_dir  = Path(index_dir)
        self.batch_size = batch_size

        from transformers import (DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
                                   DPRContextEncoder,  DPRContextEncoderTokenizer)
        logger.info(f"Loading DPR encoders: {model_name}")
        self.q_encoder   = DPRQuestionEncoder.from_pretrained(model_name).to(device)
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
        ctx_model        = model_name.replace("question_encoder", "ctx_encoder")
        self.ctx_encoder  = DPRContextEncoder.from_pretrained(ctx_model).to(device)
        self.ctx_tokenizer= DPRContextEncoderTokenizer.from_pretrained(ctx_model)
        self.q_encoder.eval()
        self.ctx_encoder.eval()

        self.doc_titles: List[str] = []
        self.doc_texts:  List[str] = []
        self._index = None

    def build_index(self, corpus: Dict[str, str]):
        import faiss
        index_path = self.index_dir / "dpr_index.faiss"
        meta_path  = self.index_dir / "dpr_meta.pkl"

        if index_path.exists() and meta_path.exists():
            logger.info("Loading existing DPR FAISS index")
            self._index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.doc_titles = meta["titles"]
            self.doc_texts  = meta["texts"]
            return

        logger.info(f"Building DPR FAISS index for {len(corpus)} docs")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.doc_titles = list(corpus.keys())
        self.doc_texts  = [corpus[t] for t in self.doc_titles]

        embeddings = []
        for i in range(0, len(self.doc_texts), self.batch_size):
            batch_texts   = self.doc_texts[i: i + self.batch_size]
            batch_titles  = self.doc_titles[i: i + self.batch_size]
            enc = self.ctx_tokenizer(
                batch_titles, batch_texts,
                max_length=256, padding=True,
                truncation="only_second", return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                emb = self.ctx_encoder(**enc).pooler_output
            embeddings.append(emb.cpu().numpy())
            if i % (self.batch_size * 20) == 0:
                logger.info(f"  Encoded {i}/{len(self.doc_texts)}")

        embs = np.vstack(embeddings).astype(np.float32)
        faiss.normalize_L2(embs)
        dim = embs.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embs)

        faiss.write_index(self._index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump({"titles": self.doc_titles, "texts": self.doc_texts}, f)
        logger.info("DPR FAISS index built")

    def encode_documents(self, docs: List[Dict]) -> torch.Tensor:
        """返回文档语义向量 (N, D)，供策略网络打分用。"""
        titles = [d["title"] for d in docs]
        texts  = [d["text"]  for d in docs]
        enc = self.ctx_tokenizer(
            titles, texts,
            max_length=256, padding=True,
            truncation="only_second", return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            emb = self.ctx_encoder(**enc).pooler_output
        return F.normalize(emb, dim=-1)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        exclude_titles: Optional[List[str]] = None,
    ) -> List[Dict]:
        assert self._index is not None, "Call build_index() first"
        enc = self.q_tokenizer(
            query, max_length=128, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            emb = self.q_encoder(**enc).pooler_output.cpu().numpy().astype(np.float32)
        import faiss
        faiss.normalize_L2(emb)
        scores, indices = self._index.search(
            emb, top_k + len(exclude_titles or []) + 10
        )
        excluded = set(exclude_titles or [])
        results  = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or self.doc_titles[idx] in excluded:
                continue
            results.append({
                "title": self.doc_titles[idx],
                "text":  self.doc_texts[idx],
                "score": float(score),
                "idx":   int(idx),
            })
            if len(results) >= top_k:
                break
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Retriever（BM25 + DPR，RRF 融合）— 推荐用于训练
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Reciprocal Rank Fusion (RRF) 融合 BM25 和 DPR。
    - 召回阶段：两路各取 top_k*2，RRF 合并后取 top_k
    - encode_documents：用 DPR ctx encoder 给策略网络提供向量
    
    根据诊断数据，BM25 权重应更高。
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        bm25_weight: float = 0.7,   # BM25 recall 更强，权重更高
        dense_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        self.bm25    = bm25_retriever
        self.dense   = dense_retriever
        self.bm25_w  = bm25_weight
        self.dense_w = dense_weight
        self.rrf_k   = rrf_k

    def build_index(self, corpus: Dict[str, str]):
        self.bm25.build_index(corpus)
        self.dense.build_index(corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        exclude_titles: Optional[List[str]] = None,
    ) -> List[Dict]:
        fetch_k  = top_k * 3
        excluded = set(exclude_titles or [])

        bm25_results  = self.bm25.retrieve(query,  fetch_k, exclude_titles)
        dense_results = self.dense.retrieve(query, fetch_k, exclude_titles)

        # RRF 打分
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}

        for rank, doc in enumerate(bm25_results):
            t = doc["title"]
            scores[t] = scores.get(t, 0) + self.bm25_w / (self.rrf_k + rank + 1)
            doc_map[t] = doc

        for rank, doc in enumerate(dense_results):
            t = doc["title"]
            scores[t] = scores.get(t, 0) + self.dense_w / (self.rrf_k + rank + 1)
            if t not in doc_map:
                doc_map[t] = doc

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        results = []
        for title, score in ranked:
            if title in excluded:
                continue
            d = doc_map[title]
            results.append({
                "title": title,
                "text":  d["text"],
                "score": float(score),
                "idx":   d.get("idx", -1),
            })
            if len(results) >= top_k:
                break
        return results

    def encode_documents(self, docs: List[Dict]) -> torch.Tensor:
        """用 DPR ctx encoder 编码，供策略网络打分。"""
        return self.dense.encode_documents(docs)
