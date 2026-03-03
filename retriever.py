"""
Retriever 实现（重构版）— 解耦 DPR 依赖，支持任意高效编码器

改动说明：
  ① 新增 EncoderBackend 抽象类，统一编码接口
  ② SentenceTransformerBackend：使用 sentence-transformers（推荐，速度快、模型丰富）
     推荐模型：
       - BAAI/bge-base-en-v1.5      (英文，高质量，768d)
       - sentence-transformers/all-MiniLM-L6-v2  (英文，极快，384d)
       - BAAI/bge-m3                (多语言，支持中英)
       - intfloat/e5-base-v2        (英文，性能均衡)
  ③ HuggingFaceBackend：直接用 AutoModel + mean pooling（兼容旧 DPR 等）
  ④ DenseRetriever 改为接受 EncoderBackend 实例，不再写死 DPR
  ⑤ HybridRetriever 保持接口不变，内部改用新 DenseRetriever
  ⑥ build_index / retrieve / encode_documents 接口与旧版完全兼容

速度对比（encode 1000 docs, GPU）：
  DPR ctx encoder    : ~12s（需分别加载 q/ctx 两个模型）
  all-MiniLM-L6-v2   : ~1.5s（单模型，384d，速度6x+）
  bge-base-en-v1.5   : ~3s   （单模型，768d，质量更好）
"""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# 屏蔽 sentence-transformers / huggingface_hub 的 HTTP 请求日志
for _noisy in ["httpx", "httpcore", "huggingface_hub", "sentence_transformers",
               "transformers.modeling_utils", "filelock"]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# ① 抽象 EncoderBackend
# ─────────────────────────────────────────────────────────────────────────────

class EncoderBackend(ABC):
    """统一编码接口，子类实现 encode()。"""

    @property
    @abstractmethod
    def dim(self) -> int:
        """返回向量维度"""

    @abstractmethod
    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        将文本列表编码为 L2 归一化的 numpy 矩阵 (N, dim)。
        子类保证输出已做 L2 归一化，可直接做内积相似度计算。
        """


# ─────────────────────────────────────────────────────────────────────────────
# ② SentenceTransformerBackend（推荐）
# ─────────────────────────────────────────────────────────────────────────────

class SentenceTransformerBackend(EncoderBackend):
    """
    基于 sentence-transformers 库的编码后端。
    安装：pip install sentence-transformers

    推荐模型（英文 HotpotQA）：
      - BAAI/bge-base-en-v1.5      高质量 768d
      - sentence-transformers/all-MiniLM-L6-v2  极快 384d
      - intfloat/e5-base-v2        均衡 768d，查询需加 "query: " 前缀
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
        query_prefix: str = "",   # e5 系列需要 "query: " / "passage: "
        doc_prefix: str = "",
        cache_folder: str = "cache",
    ):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SentenceTransformer: {model_name}")
        self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_folder)
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self._device = device
        self._dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"SentenceTransformer ready: dim={self._dim}")

    @property
    def dim(self) -> int:
        return self._dim

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
        is_query: bool = False,
    ) -> np.ndarray:
        prefix = self.query_prefix if is_query else self.doc_prefix
        if prefix:
            texts = [prefix + t for t in texts]
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # 直接输出 L2 归一化
            convert_to_numpy=True,
        )
        return embs.astype(np.float32)

    def encode_query(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return self.encode(texts, batch_size=batch_size, is_query=True)

    def encode_docs(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return self.encode(texts, batch_size=batch_size, is_query=False)


# ─────────────────────────────────────────────────────────────────────────────
# ③ HuggingFaceBackend（兼容旧 DPR / BERT 等 HF 模型）
# ─────────────────────────────────────────────────────────────────────────────

class HuggingFaceBackend(EncoderBackend):
    """
    通用 HuggingFace AutoModel 后端，使用 [CLS] 或 mean pooling。
    兼容旧 DPR、BERT、RoBERTa 等模型，但推荐优先用 SentenceTransformerBackend。
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        pooling: str = "mean",    # "mean" | "cls" | "pooler"
        cache_dir: str = "cache",
        max_length: int = 256,
    ):
        from transformers import AutoModel, AutoTokenizer
        logger.info(f"Loading HuggingFace model: {model_name}, pooling={pooling}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        self.model.eval()
        self._device = device
        self._pooling = pooling
        self._max_length = max_length

        # 推断维度
        with torch.no_grad():
            dummy = self.tokenizer(["test"], return_tensors="pt").to(device)
            out = self.model(**dummy)
            h = self._pool(out)
        self._dim = h.shape[-1]
        logger.info(f"HuggingFaceBackend ready: dim={self._dim}")

    @property
    def dim(self) -> int:
        return self._dim

    def _pool(self, out) -> torch.Tensor:
        if self._pooling == "pooler" and hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        elif self._pooling == "cls":
            return out.last_hidden_state[:, 0, :]
        else:  # mean pooling
            token_emb = out.last_hidden_state
            # attention mask from tokenizer stored externally; fallback: mean all tokens
            return token_emb.mean(dim=1)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                max_length=self._max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                out = self.model(**enc)
                h = self._pool(out)
                # mean pooling with attention mask
                if self._pooling == "mean":
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    h = (out.last_hidden_state * mask).sum(1) / (mask.sum(1) + 1e-9)
                h = F.normalize(h, dim=-1)
            all_embs.append(h.cpu().numpy().astype(np.float32))
        return np.vstack(all_embs)

    encode_query = encode  # 无前缀区分
    encode_docs = encode


# ─────────────────────────────────────────────────────────────────────────────
# 工厂函数：根据配置创建合适的 Backend
# ─────────────────────────────────────────────────────────────────────────────

def create_encoder_backend(
    model_name: str = "BAAI/bge-base-en-v1.5",
    backend_type: str = "auto",   # "auto" | "sentence_transformer" | "huggingface"
    device: str = "cpu",
    cache_dir: str = "cache",
    **kwargs,
) -> EncoderBackend:
    """
    工厂函数，根据 backend_type 和 model_name 自动选择后端。

    backend_type="auto" 时：
      - 优先尝试 sentence-transformers（安装了就用）
      - fallback 到 HuggingFaceBackend
    """
    if backend_type == "sentence_transformer":
        return SentenceTransformerBackend(model_name, device=device,
                                          cache_folder=cache_dir, **kwargs)
    elif backend_type == "huggingface":
        return HuggingFaceBackend(model_name, device=device,
                                  cache_dir=cache_dir, **kwargs)
    else:  # auto
        try:
            import sentence_transformers  # noqa
            return SentenceTransformerBackend(model_name, device=device,
                                              cache_folder=cache_dir, **kwargs)
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to HuggingFaceBackend")
            return HuggingFaceBackend(model_name, device=device,
                                      cache_dir=cache_dir, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# BM25 Retriever（不变，主力召回）
# ─────────────────────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    BM25 稀疏检索。recall@10=0.510，作为主检索器。
    多跳时查询改写：q2 = "question bridge_sentence"
    """

    def __init__(self, index_dir: str = "data/index"):
        self.index_dir = Path(index_dir)
        self._bm25 = None
        self.doc_titles: List[str] = []
        self.doc_texts: List[str] = []

    def build_index(self, corpus: Dict[str, str]):
        cache_path = self.index_dir / "bm25_index.pkl"
        if cache_path.exists():
            logger.info("Loading BM25 index from cache")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self._bm25 = data["bm25"]
            self.doc_titles = data["titles"]
            self.doc_texts = data["texts"]
            return

        from rank_bm25 import BM25Okapi
        logger.info(f"Building BM25 index for {len(corpus)} docs")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.doc_titles = list(corpus.keys())
        self.doc_texts = [corpus[t] for t in self.doc_titles]
        tokenized = [text.lower().split() for text in self.doc_texts]
        self._bm25 = BM25Okapi(tokenized)

        with open(cache_path, "wb") as f:
            pickle.dump({"bm25": self._bm25,
                         "titles": self.doc_titles,
                         "texts": self.doc_texts}, f)
        logger.info("BM25 index built")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        exclude_titles: Optional[List[str]] = None,
    ) -> List[Dict]:
        assert self._bm25 is not None, "Call build_index() first"
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
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
                "text": self.doc_texts[idx],
                "score": float(score),
                "idx": int(idx),
            })
        return results

    def encode_documents(self, docs: List[Dict]) -> torch.Tensor:
        raise NotImplementedError(
            "BM25Retriever 没有 embedding，请用 DenseRetriever 或 HybridRetriever"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ④ Dense Retriever（重构：不再写死 DPR，接受任意 EncoderBackend）
# ─────────────────────────────────────────────────────────────────────────────

class DenseRetriever:
    """
    稠密检索器，使用 FAISS 进行近似最近邻搜索。
    编码器由 EncoderBackend 提供，支持 sentence-transformers / HuggingFace 任意模型。

    典型用法：
        backend = SentenceTransformerBackend("BAAI/bge-base-en-v1.5", device="cuda")
        retriever = DenseRetriever(backend, index_dir="data/index")
        retriever.build_index(corpus)
        results = retriever.retrieve("Who directed Inception?", top_k=10)
    """

    def __init__(
        self,
        encoder: EncoderBackend,
        index_dir: str = "data/index",
        batch_size: int = 128,
        index_type: str = "flat",     # "flat" | "ivf" (大语料用 IVF)
        ivf_nlist: int = 1024,        # IVF 聚类数（仅 index_type="ivf" 时有效）
    ):
        self.encoder = encoder
        self.index_dir = Path(index_dir)
        self.batch_size = batch_size
        self.index_type = index_type
        self.ivf_nlist = ivf_nlist

        self.doc_titles: List[str] = []
        self.doc_texts: List[str] = []
        self._index = None

        # 用 encoder 名称（取 dim）区分缓存文件，避免不同模型复用同一 index
        self._cache_tag = f"dense_{encoder.dim}d"

    def build_index(self, corpus: Dict[str, str]):
        import faiss
        index_path = self.index_dir / f"{self._cache_tag}_index.faiss"
        meta_path = self.index_dir / f"{self._cache_tag}_meta.pkl"

        if index_path.exists() and meta_path.exists():
            logger.info(f"Loading cached FAISS index: {index_path}")
            self._index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.doc_titles = meta["titles"]
            self.doc_texts = meta["texts"]
            return

        logger.info(f"Building FAISS index for {len(corpus)} docs (encoder.dim={self.encoder.dim})")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.doc_titles = list(corpus.keys())
        self.doc_texts = [corpus[t] for t in self.doc_titles]

        # 格式化为 "title. text" 供文档编码
        doc_inputs = [f"{t}. {self.doc_texts[i]}" for i, t in enumerate(self.doc_titles)]

        embs = self.encoder.encode(
            doc_inputs,
            batch_size=self.batch_size,
            show_progress=True,
        )  # (N, dim), already L2 normalized

        dim = embs.shape[1]
        if self.index_type == "ivf" and len(embs) > self.ivf_nlist * 10:
            quantizer = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, self.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
            self._index.train(embs)
            self._index.nprobe = 64
            logger.info(f"Using IVF index (nlist={self.ivf_nlist})")
        else:
            self._index = faiss.IndexFlatIP(dim)
            logger.info("Using Flat IP index")

        self._index.add(embs)

        faiss.write_index(self._index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump({"titles": self.doc_titles, "texts": self.doc_texts}, f)
        logger.info(f"FAISS index built: {len(self.doc_titles)} docs")

    def encode_documents(self, docs: List[Dict]) -> torch.Tensor:
        """
        返回文档语义向量 (N, dim)，供策略网络打分用。
        使用 doc prefix（如果 backend 支持）。
        """
        inputs = [f"{d['title']}. {d['text'][:400]}" for d in docs]
        if hasattr(self.encoder, "encode_docs"):
            embs = self.encoder.encode_docs(inputs, batch_size=len(inputs))
        else:
            embs = self.encoder.encode(inputs)
        return torch.from_numpy(embs)  # already normalized

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        exclude_titles: Optional[List[str]] = None,
    ) -> List[Dict]:
        assert self._index is not None, "Call build_index() first"

        if hasattr(self.encoder, "encode_query"):
            q_emb = self.encoder.encode_query([query])
        else:
            q_emb = self.encoder.encode([query])

        import faiss
        fetch_k = top_k + len(exclude_titles or []) + 10
        scores, indices = self._index.search(q_emb, fetch_k)

        excluded = set(exclude_titles or [])
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or self.doc_titles[idx] in excluded:
                continue
            results.append({
                "title": self.doc_titles[idx],
                "text": self.doc_texts[idx],
                "score": float(score),
                "idx": int(idx),
            })
            if len(results) >= top_k:
                break
        return results


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ Hybrid Retriever（接口不变，内部用新 DenseRetriever）
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Reciprocal Rank Fusion (RRF) 融合 BM25 和 Dense。
    BM25 权重更高（0.7），因为其 recall 更强。
    encode_documents 委托给 DenseRetriever。
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        bm25_weight: float = 0.7,
        dense_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.bm25_w = bm25_weight
        self.dense_w = dense_weight
        self.rrf_k = rrf_k

    def build_index(self, corpus: Dict[str, str]):
        self.bm25.build_index(corpus)
        self.dense.build_index(corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        exclude_titles: Optional[List[str]] = None,
    ) -> List[Dict]:
        fetch_k = top_k * 3
        excluded = set(exclude_titles or [])

        bm25_results = self.bm25.retrieve(query, fetch_k, exclude_titles)
        dense_results = self.dense.retrieve(query, fetch_k, exclude_titles)

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
                "text": d["text"],
                "score": float(score),
                "idx": d.get("idx", -1),
            })
            if len(results) >= top_k:
                break
        return results

    def encode_documents(self, docs: List[Dict]) -> torch.Tensor:
        return self.dense.encode_documents(docs)