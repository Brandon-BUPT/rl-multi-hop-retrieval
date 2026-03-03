"""
Policy Network — 重构版（解耦 DPR 依赖）

变更说明：
  ① encoder 不再写死为 DPR，改为接受任意 EncoderBackend（sentence-transformers / HF）
  ② PolicyNetwork 构造函数接受 encoder_backend 对象，或通过 model_name + backend_type 自动创建
  ③ 去除对 DPRQuestionEncoder 特有 API 的依赖（question_encoder 属性等）
  ④ 编码维度自动从 backend.dim 推断，无需手动指定
  ⑤ 保持 forward / get_action_and_value / get_probs 接口完全不变，ppo_trainer.py 无需修改
  ⑥ 新增 freeze_ratio 参数（按比例 freeze），同时保留旧 freeze_layers 整数参数

推荐配置：
  - 快速实验：all-MiniLM-L6-v2 (384d, 速度极快)
  - 高质量：BAAI/bge-base-en-v1.5 (768d)
  - 多语言：BAAI/bge-m3
"""
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message=".*overflowing tokens.*", category=UserWarning)
logger = logging.getLogger(__name__)

_STATE_MAX_CHARS = 400
_BRIDGE_MAX_CHARS = 800


def build_state_text(question: str, selected_docs: List[Dict]) -> str:
    if not selected_docs:
        return f"Question: {question}"
    parts = []
    for i, doc in enumerate(selected_docs):
        max_c = _BRIDGE_MAX_CHARS if i == len(selected_docs) - 1 else _STATE_MAX_CHARS
        parts.append(f"{doc['title']}. {doc['text'][:max_c].strip()}")
    return f"Question: {question} [SEP] Known: {' | '.join(parts)} [SEP] Find: {question}"

_build_state_text = build_state_text  # 兼容旧导入


class CrossAttentionScorer(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, state_vec: torch.Tensor, cand_vecs: torch.Tensor) -> torch.Tensor:
        q = state_vec.unsqueeze(1)
        attended, _ = self.attn(q, cand_vecs, cand_vecs)
        state_exp = state_vec.unsqueeze(1).expand_as(cand_vecs)
        interaction = state_exp * cand_vecs + state_exp - cand_vecs
        return self.score_proj(interaction).squeeze(-1)  # (B, K)


class PolicyNetwork(nn.Module):
    """
    策略网络，单塔架构。
    state 和候选文档在同一嵌入空间，cross-attention 打分。

    参数：
        encoder_backend : EncoderBackend 实例（推荐）
            或通过 encoder_model + backend_type 自动构建
        encoder_model   : 模型名称（当 encoder_backend=None 时使用）
        backend_type    : "auto" | "sentence_transformer" | "huggingface"
        encoder_dim     : 若 None 则从 backend.dim 自动推断
        hidden_dim      : 投影后的隐藏维度，None 则与 encoder_dim 一致
        freeze_layers   : 冻结底部 N 层（int）；-1 = 不冻结；None = 用 freeze_ratio
        freeze_ratio    : 按比例冻结（0.0~1.0），freeze_layers 优先级更高
    """

    def __init__(
        self,
        encoder_backend=None,           # EncoderBackend 实例（优先）
        encoder_model: str = "BAAI/bge-base-en-v1.5",
        backend_type: str = "auto",
        encoder_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        n_heads: int = 8,
        dropout: float = 0.1,
        freeze_layers: Optional[int] = 6,   # 整数层数，-1=不冻结
        freeze_ratio: float = 0.5,           # 按比例冻结（仅 freeze_layers=None 时生效）
        device: str = "cpu",
        cache_dir: str = "cache",
        # ── 以下为兼容旧调用的参数，忽略 ──
        ctx_encoder_model: str = None,
    ):
        super().__init__()
        self.device = device

        # ── ① 初始化 Encoder Backend ──────────────────────────────────────
        if encoder_backend is not None:
            self._backend = encoder_backend
        else:
            from retriever import create_encoder_backend
            self._backend = create_encoder_backend(
                model_name=encoder_model,
                backend_type=backend_type,
                device=device,
                cache_dir=cache_dir,
            )

        enc_dim = encoder_dim if encoder_dim is not None else self._backend.dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else enc_dim

        # ── ② 获取底层 transformer 模型（用于层冻结和梯度更新）─────────────
        self._transformer = self._extract_transformer()
        self.tokenizer = self._extract_tokenizer()

        # 冻结层
        self._apply_freeze(freeze_layers, freeze_ratio)

        # ── ③ Hop embedding ───────────────────────────────────────────────
        self.hop_embedding = nn.Embedding(4, self.hidden_dim)
        nn.init.normal_(self.hop_embedding.weight, std=0.02)

        # ── ④ 共享投影层（state 和文档都走这个）─────────────────────────
        self.state_proj = nn.Sequential(
            nn.Linear(enc_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim),
        )

        # ── ⑤ Cross-Attention Scorer ──────────────────────────────────────
        self.cross_attn = CrossAttentionScorer(self.hidden_dim, n_heads, dropout)
        self.score_gate = nn.Parameter(torch.tensor(0.5))

        # ── STOP ──────────────────────────────────────────────────────────
        self.stop_emb = nn.Parameter(torch.randn(self.hidden_dim) * 0.02)
        self.stop_proj = nn.Linear(self.hidden_dim * 2, 1)

        # ── Value Head ────────────────────────────────────────────────────
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2), nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        self._init_weights()
        self._log_params()

    # ── Backend 解析 ──────────────────────────────────────────────────────

    def _extract_transformer(self):
        """从 backend 中提取底层 transformer 模型（用于冻结和前向传播）"""
        backend = self._backend
        # SentenceTransformerBackend
        if hasattr(backend, "model"):
            m = backend.model
            # sentence_transformers.SentenceTransformer
            if hasattr(m, "_modules"):
                for mod in m.modules():
                    if hasattr(mod, "embeddings") and hasattr(mod, "encoder"):
                        return mod  # BERT/RoBERTa 核心
                # 找不到就用整个 model
            return m
        # HuggingFaceBackend
        if hasattr(backend, "_transformer"):
            return backend._transformer
        return None

    def _extract_tokenizer(self):
        """从 backend 提取 tokenizer（用于 _encode_text）"""
        backend = self._backend
        if hasattr(backend, "tokenizer"):
            return backend.tokenizer
        # SentenceTransformer: 第一个 module 是 Transformer，有 tokenizer
        if hasattr(backend, "model") and hasattr(backend.model, "tokenizer"):
            return backend.model.tokenizer
        # 尝试从 sentence_transformers modules
        if hasattr(backend, "model"):
            for mod in vars(backend.model).values():
                if hasattr(mod, "encode") and hasattr(mod, "tokenizer"):
                    return mod.tokenizer
        return None

    def _apply_freeze(self, freeze_layers: Optional[int], freeze_ratio: float):
        """冻结 transformer 层"""
        t = self._transformer
        if t is None:
            return

        # 冻结 embedding
        if hasattr(t, "embeddings"):
            for p in t.embeddings.parameters():
                p.requires_grad = False

        # 获取 encoder 层列表
        layers = self._get_encoder_layers(t)
        if not layers:
            return

        n_total = len(layers)
        if freeze_layers is not None and freeze_layers >= 0:
            n_freeze = min(freeze_layers, n_total)
        elif freeze_layers == -1:
            n_freeze = 0
        else:
            n_freeze = int(n_total * freeze_ratio)

        for i, layer in enumerate(layers):
            if i < n_freeze:
                for p in layer.parameters():
                    p.requires_grad = False
        logger.info(f"Frozen bottom {n_freeze}/{n_total} transformer layers + embeddings")

    def _get_encoder_layers(self, model) -> list:
        """尝试各种常见 transformer 架构获取层列表"""
        # BERT/RoBERTa
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return list(model.encoder.layer)
        # DPR
        if hasattr(model, "question_encoder"):
            try:
                return list(model.question_encoder.bert_model.encoder.layer)
            except AttributeError:
                pass
        # GPT-style
        if hasattr(model, "h"):
            return list(model.h)
        # 通用 transformer blocks
        for attr in ["layers", "blocks", "transformer_layers"]:
            if hasattr(model, attr):
                return list(getattr(model, attr))
        return []

    def _init_weights(self):
        for mod in [self.state_proj, self.value_head]:
            for layer in mod:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.stop_proj.weight)
        nn.init.constant_(self.stop_proj.bias, 0.1)

    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"PolicyNetwork: {trainable:,} trainable / {total:,} total "
            f"({100 * trainable / total:.1f}%) | hidden_dim={self.hidden_dim}"
        )

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ── Encoding ──────────────────────────────────────────────────────────

    def _encode_text(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """
        通用文本编码，返回 (N, hidden_dim)。
        优先走 backend 的 encode 方法（支持 sentence-transformers 的高效批处理）；
        若 tokenizer 可用则支持梯度传播（训练时用）。
        """
        # 训练时需要梯度，走 transformer forward
        if self._transformer is not None and self.tokenizer is not None:
            return self._encode_with_grad(texts, max_length)

        # 评测/no_grad 时直接用 backend.encode（更高效）
        with torch.no_grad():
            embs = self._backend.encode(texts)
            h = torch.from_numpy(embs).to(self.device)
            return self.state_proj(h)

    def _encode_with_grad(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """支持梯度的编码路径（训练时 encoder 可微）"""
        enc = self.tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        out = self._transformer(**enc)

        # mean pooling（通用，无需 pooler_output）
        if hasattr(out, "last_hidden_state"):
            mask = enc["attention_mask"].unsqueeze(-1).float()
            h = (out.last_hidden_state * mask).sum(1) / (mask.sum(1) + 1e-9)
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            h = out.pooler_output
        else:
            h = out[0][:, 0, :]

        h = F.normalize(h, dim=-1)
        return self.state_proj(h)

    def encode_state(
        self,
        question: str,
        selected_docs: List[Dict],
        hop: Optional[int] = None,
    ) -> torch.Tensor:
        """Returns (1, hidden_dim)"""
        text = build_state_text(question, selected_docs)
        vec = self._encode_text([text], max_length=512)  # (1, D)
        hop_n = len(selected_docs) if hop is None else hop
        vec = vec + self.hop_embedding(torch.tensor([min(hop_n, 3)], device=self.device))
        return vec

    def encode_docs(self, docs: List[Dict]) -> torch.Tensor:
        """Returns (N, hidden_dim) — 单塔，与 state 同空间"""
        texts = [f"{d['title']}. {d['text'][:400]}" for d in docs]
        return self._encode_text(texts, max_length=256)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        state_vec: torch.Tensor,
        cand_embs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state_vec.dim() == 1:
            state_vec = state_vec.unsqueeze(0)
        if cand_embs.dim() == 2:
            cand_embs = cand_embs.unsqueeze(0)
        B, K, D = cand_embs.shape

        dot = torch.bmm(cand_embs, state_vec.unsqueeze(-1)).squeeze(-1)   # (B, K)
        ca = self.cross_attn(state_vec, cand_embs)                         # (B, K)
        gate = torch.sigmoid(self.score_gate)
        doc_scores = gate * dot + (1 - gate) * ca                          # (B, K)

        stop_score = self.stop_proj(
            torch.cat([state_vec, self.stop_emb.unsqueeze(0).expand(B, -1)], dim=-1)
        )                                                                   # (B, 1)
        logits = torch.cat([doc_scores, stop_score], dim=-1)               # (B, K+1)
        value = self.value_head(state_vec)                                  # (B, 1)
        return logits, value

    def get_action_and_value(self, state_vec, cand_embs, action=None):
        logits, value = self.forward(state_vec, cand_embs)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_probs(self, state_vec, cand_embs):
        logits, _ = self.forward(state_vec, cand_embs)
        return F.softmax(logits, dim=-1)
