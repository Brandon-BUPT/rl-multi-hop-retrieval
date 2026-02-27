"""
Multi-hop Retrieval MDP Environment
retrieval_mode: "context" (oracle distractor pool, 추천), "bm25", "dpr"
"""
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re, string

logger = logging.getLogger(__name__)
STOP_ACTION = -1


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


class MultiHopRetrievalEnv:

    def __init__(self, retriever, reader, top_k=10, max_hops=2,
                 retrieval_mode="context", reward_config=None):
        self.retriever = retriever
        self.reader = reader
        self.top_k = top_k
        self.max_hops = max_hops
        self.retrieval_mode = retrieval_mode
        self.reward_config = reward_config or {
            "em_weight": 1.0, "f1_weight": 0.5, "sf_weight": 0.5,
            "step_weight": 0.1, "use_step_reward": True,
            "early_stop_penalty": 0.3, "min_hops_before_stop": 2,
        }

    def reset(self, item: Dict) -> RetrievalState:
        state = RetrievalState(
            question=item["question"],
            question_id=item["_id"],
            gold_answer=item["answer"],
            gold_supporting_titles=list({sf[0] for sf in item.get("supporting_facts", [])}),
        )
        if self.retrieval_mode == "context":
            pool = [{"title": t, "text": " ".join(s), "score": 1.0}
                    for t, s in item.get("context", [])]
            state.context_pool = pool
            state.candidates = self._context_candidates(state, pool)
        else:
            state.candidates = self._retrieve(state)
        return state

    def _context_candidates(self, state: RetrievalState, pool: List[Dict]) -> List[Dict]:
        excluded = set(state.selected_titles)
        remaining = [d for d in pool if d["title"] not in excluded]
        random.shuffle(remaining)
        return remaining[:self.top_k]

    def _retrieve(self, state: RetrievalState) -> List[Dict]:
        query = state.question
        if state.selected_titles:
            query = f"{state.question} {state.selected_titles[-1]}"
        return self.retriever.retrieve(query=query, top_k=self.top_k,
                                       exclude_titles=state.selected_titles)

    def step(self, state: RetrievalState, action: int) -> Tuple[RetrievalState, float, bool, Dict]:
        assert not state.done
        min_hops = self.reward_config.get("min_hops_before_stop", 2)

        # 强制走满 min_hops 步
        if action == STOP_ACTION and state.hop < min_hops:
            action = 0

        if action == STOP_ACTION or state.hop >= self.max_hops:
            reward, info = self._final_reward(state)
            state.done = True
            return state, reward, True, info

        selected = state.candidates[action]
        state.selected_docs.append(selected)
        state.selected_titles.append(selected["title"])
        state.hop += 1

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

    def _step_reward(self, state: RetrievalState, doc: Dict) -> float:
        if not self.reward_config.get("use_step_reward", True): return 0.0
        if doc["title"] not in state.gold_supporting_titles: return 0.0
        w = self.reward_config.get("step_weight", 0.1)
        hop_mul = 1.0 + (state.hop - 1) * 0.5  # hop1=1.0x, hop2=1.5x
        return w * hop_mul

    def _final_reward(self, state: RetrievalState) -> Tuple[float, Dict]:
        docs = [{"title": d["title"], "text": d["text"]} for d in state.selected_docs]
        if docs:
            predicted = self.reader.predict(state.question, docs)
            em = exact_match_score(predicted, state.gold_answer)
            f1 = f1_score(predicted, state.gold_answer)
        else:
            predicted, em, f1 = "", 0.0, 0.0

        sel = set(state.selected_titles)
        gold = set(state.gold_supporting_titles)
        if gold:
            sf_p = len(sel & gold) / max(len(sel), 1)
            sf_r = len(sel & gold) / len(gold)
            sf_f1 = 2 * sf_p * sf_r / (sf_p + sf_r + 1e-9)
            sf_rec = sf_r
        else:
            sf_f1 = sf_rec = 0.0

        # 惩罚不足 min_hops 就结束
        min_hops = self.reward_config.get("min_hops_before_stop", 2)
        penalty = self.reward_config.get("early_stop_penalty", 0.3)
        stop_pen = -penalty * max(0, min_hops - state.hop)

        cfg = self.reward_config
        reward = (cfg.get("em_weight", 1.0) * em
                  + cfg.get("f1_weight", 0.5) * f1
                  + cfg.get("sf_weight", 0.5) * sf_f1
                  + stop_pen)

        joint_recall = float(gold.issubset(sel)) if gold else 0.0
        sf_hit_hop1 = float(state.selected_titles[0] in gold) if len(state.selected_titles) >= 1 else 0.0
        sf_hit_hop2 = float(state.selected_titles[1] in gold) if len(state.selected_titles) >= 2 else 0.0

        return reward, {
            "em": em, "f1": f1, "sf_f1": sf_f1, "sf_recall": sf_rec,
            "joint_recall": joint_recall,
            "sf_hit_hop1": sf_hit_hop1, "sf_hit_hop2": sf_hit_hop2,
            "predicted_answer": predicted, "gold_answer": state.gold_answer,
        }