"""
Multi-hop Retrieval MDP Environment
retrieval_mode: "context" (oracle distractor pool, 推荐), "bm25", "dpr"

改动说明：
  ① query 改写：第二跳用文档内容提取桥接实体，而非仅拼标题
  ② 奖励解耦：检索奖励（SF-based）与答案奖励（EM/F1）分离
     - 主奖励 = sf_f1 + joint_recall_bonus（完全不依赖 reader）
     - 答案 bonus = em/f1（仅在检索正确时才有意义，权重低）
  ⑥ 去掉 STOP action：HotpotQA 固定 2 跳，STOP 是噪声
     - 动作空间从 K+1 缩减为 K
     - 移除 min_hops_before_stop / early_stop_penalty 逻辑
     - STOP_ACTION 保留常量以兼容 evaluator 的导入，但 env 不再产生 STOP
"""
import logging
import random
import re
import string
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
STOP_ACTION = -1  # 保留常量供外部兼容，env 内部不再使用


# ─────────────────────────────────────────────────────────────────────────────
# 文本工具
# ─────────────────────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())

def exact_match_score(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))

def f1_score(pred: str, gold: str) -> float:
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    common = set(pt) & set(gt)
    if not common: return 0.0
    p = len(common) / len(pt)
    r = len(common) / len(gt)
    return 2 * p * r / (p + r + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# ① 桥接实体提取（用于 hop≥1 的 query 改写）
# ─────────────────────────────────────────────────────────────────────────────

def extract_bridge_query(question: str, doc: Dict, max_chars: int = 120) -> str:
    """
    从上一跳文档中提取与问题最相关的句子作为桥接信息，
    拼接到下一跳的检索 query 中。

    策略：
      1. 将文档切成句子
      2. 用问题 token 与每句话计算词重叠分数
      3. 取最高分的句子（截断到 max_chars）
      4. 若无匹配则退化为文档标题（原有行为）
    """
    title = doc.get("title", "")
    text = doc.get("text", "")
    if not text:
        return f"{question} {title}"

    q_tokens = set(normalize_answer(question).split())
    # 过滤停用词，保留实质性关键词
    stopwords = {"what", "who", "where", "when", "which", "how", "why",
                 "is", "was", "are", "were", "did", "do", "does",
                 "the", "a", "an", "of", "in", "on", "at", "to", "for"}
    q_keywords = q_tokens - stopwords

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return f"{question} {title}"

    best_sent, best_score = sentences[0], -1.0
    for sent in sentences:
        sent_tokens = set(normalize_answer(sent).split())
        if not sent_tokens:
            continue
        overlap = len(q_keywords & sent_tokens)
        # 偏好包含命名实体特征（首字母大写词）的句子
        named_entity_bonus = sum(
            1 for w in sent.split()
            if w and w[0].isupper() and w.lower() not in stopwords
        ) * 0.1
        score = overlap / (len(sent_tokens) + 1e-9) + named_entity_bonus
        if score > best_score:
            best_score, best_sent = score, sent

    bridge = best_sent[:max_chars].strip()
    return f"{question} {bridge}"


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalState:
    question: str
    question_id: str
    selected_docs: List[Dict] = field(default_factory=list)
    selected_titles: List[str] = field(default_factory=list)
    hop: int = 0
    done: bool = False
    gold_answer: str = ""
    gold_supporting_titles: List[str] = field(default_factory=list)
    candidates: List[Dict] = field(default_factory=list)
    context_pool: List[Dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class MultiHopRetrievalEnv:

    def __init__(
        self,
        retriever,
        reader,
        top_k: int = 10,
        max_hops: int = 2,
        retrieval_mode: str = "context",
        reward_config: Optional[Dict] = None,
    ):
        self.retriever = retriever
        self.reader = reader
        self.top_k = top_k
        self.max_hops = max_hops
        self.retrieval_mode = retrieval_mode

        # ② 奖励解耦：检索奖励与答案奖励分开配置
        self.reward_config = reward_config or {
            # 检索奖励（主）：完全基于 SF，不依赖 reader
            "sf_weight":           1.0,   # sf_f1 权重
            "joint_bonus":         0.5,   # 两篇 SF 全部选中的额外奖励
            "step_weight":         0.2,   # 每步命中 SF 的即时奖励
            "use_step_reward":     True,
            # 答案奖励（辅）：仅在检索有命中时才有贡献，权重低
            "em_weight":           0.3,
            "f1_weight":           0.2,
            # 旧字段兼容（不再使用，保留以免 ablation.py 报错）
            "early_stop_penalty":  0.0,
            "min_hops_before_stop": self.max_hops,
        }

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self, item: Dict) -> RetrievalState:
        state = RetrievalState(
            question=item["question"],
            question_id=item["_id"],
            gold_answer=item["answer"],
            gold_supporting_titles=list({sf[0] for sf in item.get("supporting_facts", [])}),
        )
        if self.retrieval_mode == "context":
            pool = [
                {"title": t, "text": " ".join(s), "score": 1.0}
                for t, s in item.get("context", [])
            ]
            state.context_pool = pool
            state.candidates = self._context_candidates(state, pool)
        else:
            state.candidates = self._retrieve(state)
        return state

    # ── Candidate generation ─────────────────────────────────────────────

    def _context_candidates(
        self, state: RetrievalState, pool: List[Dict]
    ) -> List[Dict]:
        excluded = set(state.selected_titles)
        remaining = [d for d in pool if d["title"] not in excluded]
        random.shuffle(remaining)
        return remaining[: self.top_k]

    def _retrieve(self, state: RetrievalState) -> List[Dict]:
        """
        ① 改进的 query 改写：
           - hop 0：直接用 question
           - hop ≥ 1：用 extract_bridge_query 从上一跳文档中提取桥接句
        """
        if not state.selected_docs:
            query = state.question
        else:
            # 使用最后一跳文档的桥接信息
            query = extract_bridge_query(state.question, state.selected_docs[-1])

        return self.retriever.retrieve(
            query=query,
            top_k=self.top_k,
            exclude_titles=state.selected_titles,
        )

    # ── Step ─────────────────────────────────────────────────────────────

    def step(
        self, state: RetrievalState, action: int
    ) -> Tuple[RetrievalState, float, bool, Dict]:
        """
        ⑥ 去掉 STOP：action 只在 [0, len(candidates)-1] 范围内。
           外部若传入 STOP_ACTION(-1)，强制转为 action=0（最优候选），
           保持兼容性但不再按 STOP 逻辑处理。
        """
        assert not state.done

        # 兼容旧代码传入 STOP_ACTION 的情况
        if action == STOP_ACTION:
            action = 0

        # 越界保护
        if action >= len(state.candidates):
            action = 0

        selected = state.candidates[action]
        state.selected_docs.append(selected)
        state.selected_titles.append(selected["title"])
        state.hop += 1

        # 即时步奖励（检索命中 SF）
        step_reward = self._step_reward(state, selected)
        done = (state.hop >= self.max_hops)

        if done:
            final_reward, info = self._final_reward(state)
            info["step_reward"] = step_reward
            state.done = True
            return state, final_reward + step_reward, True, info
        else:
            if self.retrieval_mode == "context":
                state.candidates = self._context_candidates(state, state.context_pool)
            else:
                state.candidates = self._retrieve(state)
            return state, step_reward, False, {"step_reward": step_reward}

    # ── Reward ───────────────────────────────────────────────────────────

    def _step_reward(self, state: RetrievalState, doc: Dict) -> float:
        """每步命中 SF 的即时奖励，hop 越晚权重越高（桥接文档更难找）。"""
        if not self.reward_config.get("use_step_reward", True):
            return 0.0
        if doc["title"] not in state.gold_supporting_titles:
            return 0.0
        w = self.reward_config.get("step_weight", 0.2)
        # hop1=1.0x, hop2=1.5x：第二跳的桥接文档更难，给更多奖励
        hop_mul = 1.0 + (state.hop - 1) * 0.5
        return w * hop_mul

    def _final_reward(self, state: RetrievalState) -> Tuple[float, Dict]:
        """
        ② 奖励解耦：
           主奖励 = 检索质量（sf_f1 + joint_bonus），完全不依赖 reader
           辅奖励 = 答案质量（em + f1），权重低，且仅在有检索命中时才有意义

        设计意图：
           - policy 优化目标是找到正确的 SF 文档，不应被 reader 误差污染
           - EM/F1 作为小 bonus 保留，给 policy 一个"最终有没有答对"的弱信号
           - joint_bonus：两篇 SF 全部选中时的额外奖励，鼓励完整的两跳推理
        """
        cfg = self.reward_config
        sel = set(state.selected_titles)
        gold = set(state.gold_supporting_titles)

        # ── 检索质量（主，不依赖 reader）────────────────────────────────
        if gold:
            sf_p = len(sel & gold) / max(len(sel), 1)
            sf_r = len(sel & gold) / len(gold)
            sf_f1 = 2 * sf_p * sf_r / (sf_p + sf_r + 1e-9)
            sf_rec = sf_r
            joint_recall = float(gold.issubset(sel))
        else:
            sf_f1 = sf_rec = sf_p = sf_r = 0.0
            joint_recall = 0.0

        retrieval_reward = (
            cfg.get("sf_weight", 1.0) * sf_f1
            + cfg.get("joint_bonus", 0.5) * joint_recall
        )

        # ── 答案质量（辅，依赖 reader）───────────────────────────────────
        docs = [{"title": d["title"], "text": d["text"]} for d in state.selected_docs]
        if docs:
            predicted = self.reader.predict(state.question, docs)
            em = exact_match_score(predicted, state.gold_answer)
            f1 = f1_score(predicted, state.gold_answer)
        else:
            predicted, em, f1 = "", 0.0, 0.0

        # 答案 bonus 仅在有 SF 命中时才叠加，避免 reader 误差污染检索学习
        answer_bonus = 0.0
        if joint_recall > 0 or sf_rec > 0.5:
            answer_bonus = (
                cfg.get("em_weight", 0.3) * em
                + cfg.get("f1_weight", 0.2) * f1
            )

        total_reward = retrieval_reward + answer_bonus

        # ── 诊断信息 ──────────────────────────────────────────────────────
        sf_hit_hop1 = (
            float(state.selected_titles[0] in gold)
            if len(state.selected_titles) >= 1 else 0.0
        )
        sf_hit_hop2 = (
            float(state.selected_titles[1] in gold)
            if len(state.selected_titles) >= 2 else 0.0
        )

        return total_reward, {
            "em":               em,
            "f1":               f1,
            "sf_f1":            sf_f1,
            "sf_recall":        sf_rec,
            "sf_precision":     sf_p,
            "joint_recall":     joint_recall,
            "retrieval_reward": retrieval_reward,
            "answer_bonus":     answer_bonus,
            "sf_hit_hop1":      sf_hit_hop1,
            "sf_hit_hop2":      sf_hit_hop2,
            "predicted_answer": predicted,
            "gold_answer":      state.gold_answer,
        }