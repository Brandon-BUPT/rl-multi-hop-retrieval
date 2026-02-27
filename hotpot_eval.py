"""
HotpotQA 官方 6 指标评测
  Ans  EM/F1 : 答案字符串匹配
  Sup  EM/F1 : Supporting Facts 句子级匹配（启发式句子选择）
  Joint EM/F1: 两者同时正确
"""
import logging, re, string
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from environment import STOP_ACTION

logger = logging.getLogger(__name__)


def _normalize(s):
    def rm_art(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def rm_punc(t): return "".join(c for c in t if c not in set(string.punctuation))
    def ws(t): return " ".join(t.split())
    return ws(rm_art(rm_punc(s.lower())))

def _tokens(s): return _normalize(s).split() if s else []

def ans_em(pred, gold): return float(_normalize(pred) == _normalize(gold))

def ans_f1(pred, gold):
    pt, gt = _tokens(pred), _tokens(gold)
    cnt = defaultdict(int)
    for t in pt: cnt[t] += 1
    overlap = 0
    for t in gt:
        if cnt[t] > 0: overlap += 1; cnt[t] -= 1
    if not pt or not gt: return float(pt == gt)
    p, r = overlap/len(pt), overlap/len(gt)
    return 2*p*r/(p+r) if p+r else 0.0

def sup_em(pred_facts, gold_facts): return float(set(pred_facts) == set(gold_facts))

def sup_f1(pred_facts, gold_facts):
    ps, gs = set(pred_facts), set(gold_facts)
    tp = len(ps & gs)
    if not ps or not gs: return float(ps == gs)
    p, r = tp/len(ps), tp/len(gs)
    return 2*p*r/(p+r) if p+r else 0.0

def _split_sents(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

def predict_sf_sents(selected_docs, answer):
    ans_toks = set(_tokens(answer))
    pred = []
    for doc in selected_docs:
        sents = _split_sents(doc["text"])
        if not sents: continue
        best_i, best_s = 0, -1.0
        for i, sent in enumerate(sents):
            st = set(_tokens(sent))
            sc = len(ans_toks & st) / (len(st) + 1e-9)
            if sc > best_s: best_s, best_i = sc, i
        pred.append((doc["title"], best_i))
    return pred


class HotpotEvaluator:

    def evaluate(self, policy, env, dataset, mode="rl",
                 max_samples: Optional[int] = None) -> Dict:
        policy.eval()
        data = dataset.data
        if max_samples:
            import random
            data = random.sample(data, min(max_samples, len(data)))
        results = defaultdict(list)
        total = len(data)
        logger.info(f"Starting evaluation: {total} samples, mode={mode}")
        for idx, item in enumerate(data):
            if (idx + 1) % 500 == 0 or idx == 0:
                logger.info(f"Progress: {idx + 1}/{total} ({100*(idx+1)/total:.1f}%)")
            for k, v in self._run(policy, env, item, mode).items():
                results[k].append(v)
        policy.train()
        agg = {k: float(np.mean(v)) for k, v in results.items()}
        logger.info(self._fmt(agg))
        return agg

    def _run(self, policy, env, item, mode):
        state = env.reset(item)
        info = {}
        while not state.done:
            if not state.candidates: break
            if mode == "rl":
                with torch.no_grad():
                    sv = policy.encode_state(state.question, state.selected_docs,
                                             hop=len(state.selected_docs))
                    ce = policy.encode_docs(state.candidates)
                    probs = policy.get_probs(sv.to(policy.device), ce.to(policy.device))
                action = probs.argmax(-1).item()
            else:
                action = 0
            if action >= len(state.candidates):
                state, _, _, info = env.step(state, STOP_ACTION)
            else:
                state, _, _, info = env.step(state, action)

        pred_ans = info.get("predicted_answer", "")
        gold_ans = item["answer"]
        a_em = ans_em(pred_ans, gold_ans)
        a_f1 = ans_f1(pred_ans, gold_ans)

        gold_facts = [(sf[0], sf[1]) for sf in item.get("supporting_facts", [])]
        pred_facts = predict_sf_sents(state.selected_docs, pred_ans)
        s_em = sup_em(pred_facts, gold_facts)
        s_f1 = sup_f1(pred_facts, gold_facts)

        return {
            "ans_em": a_em, "ans_f1": a_f1,
            "sup_em": s_em, "sup_f1": s_f1,
            "joint_em": float(a_em==1.0 and s_em==1.0),
            "joint_f1": a_f1 * s_f1,
            "n_hops": state.hop,
            "joint_recall": float(
                set(state.gold_supporting_titles).issubset(set(state.selected_titles))
            ),
        }

    @staticmethod
    def _fmt(m):
        lines = ["="*52, f"{'HotpotQA Official Metrics':^52}", "="*52,
                 f"  {'':20} {'EM':>8}  {'F1':>8}", "-"*52,
                 f"  {'Answer':20} {m.get('ans_em',0):>8.4f}  {m.get('ans_f1',0):>8.4f}",
                 f"  {'Supporting Facts':20} {m.get('sup_em',0):>8.4f}  {m.get('sup_f1',0):>8.4f}",
                 f"  {'Joint':20} {m.get('joint_em',0):>8.4f}  {m.get('joint_f1',0):>8.4f}",
                 "-"*52,
                 f"  joint_recall (doc): {m.get('joint_recall',0):.4f}",
                 f"  avg n_hops:         {m.get('n_hops',0):.2f}",
                 "="*52]
        return "\n".join(lines)