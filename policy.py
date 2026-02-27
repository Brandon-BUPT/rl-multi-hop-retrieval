"""
Policy Network — Single Tower

架构（回归v4验证有效的设计）：
  encoder     : dpr-question_encoder（query侧，上6层可训练）
  state_proj  : Linear projection，state 和文档共享同一投影层
  hop_emb     : 区分第1/2跳检索意图
  cross_attn  : Cross-Attention scorer
  value_head  : PPO Critic

state 和候选文档在同一嵌入空间，cross-attention 有效。
checkpoint 完整保存所有参数。
"""
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore", message=".*overflowing tokens.*", category=UserWarning)
logger = logging.getLogger(__name__)

_STATE_MAX_CHARS  = 400
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

    def __init__(
        self,
        encoder_model: str = "facebook/dpr-question_encoder-single-nq-base",
        encoder_dim: int = 768,
        hidden_dim: int = 768,
        n_heads: int = 8,
        dropout: float = 0.1,
        freeze_layers: int = 6,
        device: str = "cpu",
        # 以下参数兼容旧调用，忽略
        ctx_encoder_model: str = None,
    ):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim

        # ── 单塔 Encoder（state 和文档共用）──────────────────────────────
        logger.info(f"Loading encoder: {encoder_model}, freeze_layers={freeze_layers}")
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        self.state_encoder = AutoModel.from_pretrained(encoder_model)
        self._freeze_layers(freeze_layers)

        # ── Hop embedding ─────────────────────────────────────────────────
        self.hop_embedding = nn.Embedding(4, hidden_dim)
        nn.init.normal_(self.hop_embedding.weight, std=0.02)

        # ── 共享投影层（state 和文档都走这个）───────────────────────────
        self.state_proj = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )

        # ── Cross-Attention Scorer ────────────────────────────────────────
        self.cross_attn = CrossAttentionScorer(hidden_dim, n_heads, dropout)
        self.score_gate = nn.Parameter(torch.tensor(0.5))

        # ── STOP ──────────────────────────────────────────────────────────
        self.stop_emb  = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.stop_proj = nn.Linear(hidden_dim * 2, 1)

        # ── Value Head ────────────────────────────────────────────────────
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()
        self._log_params()

    def _freeze_layers(self, n: int):
        if n < 0: return
        if hasattr(self.state_encoder, "embeddings"):
            for p in self.state_encoder.embeddings.parameters():
                p.requires_grad = False
        layers = self._encoder_layers()
        for i, layer in enumerate(layers):
            if i < n:
                for p in layer.parameters(): p.requires_grad = False
        logger.info(f"Frozen bottom {n}/{len(layers)} layers + embeddings")

    def _encoder_layers(self):
        enc = self.state_encoder
        if hasattr(enc, "question_encoder"):
            return enc.question_encoder.bert_model.encoder.layer
        if hasattr(enc, "encoder") and hasattr(enc.encoder, "layer"):
            return enc.encoder.layer
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
        logger.info(f"PolicyNetwork: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ── Encoding ──────────────────────────────────────────────────────────

    def _encode_text(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """通用文本编码，返回 (N, hidden_dim)"""
        enc = self.tokenizer(texts, max_length=max_length, truncation=True,
                             padding=True, return_tensors="pt").to(self.device)
        out = self.state_encoder(**enc)
        h = out.pooler_output if (hasattr(out, "pooler_output") and out.pooler_output is not None) \
            else out.last_hidden_state[:, 0, :]
        return self.state_proj(h)

    def encode_state(self, question: str, selected_docs: List[Dict],
                     hop: Optional[int] = None) -> torch.Tensor:
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

    def forward(self, state_vec: torch.Tensor,
                cand_embs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if state_vec.dim() == 1: state_vec = state_vec.unsqueeze(0)
        if cand_embs.dim() == 2: cand_embs = cand_embs.unsqueeze(0)
        B, K, D = cand_embs.shape

        dot  = torch.bmm(cand_embs, state_vec.unsqueeze(-1)).squeeze(-1)  # (B,K)
        ca   = self.cross_attn(state_vec, cand_embs)                       # (B,K)
        gate = torch.sigmoid(self.score_gate)
        doc_scores = gate * dot + (1 - gate) * ca                          # (B,K)

        stop_score = self.stop_proj(
            torch.cat([state_vec, self.stop_emb.unsqueeze(0).expand(B, -1)], dim=-1)
        )                                                                   # (B,1)
        logits = torch.cat([doc_scores, stop_score], dim=-1)               # (B,K+1)
        value  = self.value_head(state_vec)                                 # (B,1)
        return logits, value

    def get_action_and_value(self, state_vec, cand_embs, action=None):
        logits, value = self.forward(state_vec, cand_embs)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None: action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_probs(self, state_vec, cand_embs):
        logits, _ = self.forward(state_vec, cand_embs)
        return F.softmax(logits, dim=-1)