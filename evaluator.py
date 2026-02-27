"""
Evaluator — 训练时快速评测（200 samples）

指标：
  em / f1              : 答案质量
  sf_f1 / sf_recall    : Supporting Facts（文档级）
  joint_recall         : 两篇SF都选中
  sf_hit_hop1/2        : 第1/2跳命中SF
  first_sf_rank_hop1   : SF在第1跳候选中的排名
  n_hops / stopped_early / avg_stop_prob
"""
import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

from environment import STOP_ACTION

logger = logging.getLogger(__name__)


class Evaluator:

    def __init__(self, recall_k_list: List[int] = [1, 2, 5, 10]):
        self.recall_k_list = recall_k_list

    def evaluate(self, policy, env, dataset, mode="rl",
                 max_samples: Optional[int] = None) -> Dict:
        policy.eval()
        data = dataset.data
        if max_samples:
            import random
            data = random.sample(data, min(max_samples, len(data)))

        results = defaultdict(list)
        for item in data:
            r = self._run_episode(policy, env, item, mode)
            for k, v in r.items(): results[k].append(v)

        policy.train()
        return {k: float(np.mean(v)) for k, v in results.items()}

    def _run_episode(self, policy, env, item: Dict, mode: str) -> Dict:
        state = env.reset(item)
        gold_titles = set(state.gold_supporting_titles)
        hop_choices, stop_probs, stopped_early = [], [], False
        info = {}

        while not state.done:
            if not state.candidates:
                break
            candidates_now = [d["title"] for d in state.candidates]

            if mode == "rl":
                with torch.no_grad():
                    sv = policy.encode_state(state.question, state.selected_docs,
                                             hop=len(state.selected_docs))
                    ce = policy.encode_docs(state.candidates)
                    probs = policy.get_probs(sv.to(policy.device), ce.to(policy.device))
                action_idx = probs.argmax(dim=-1).item()
                stop_probs.append(probs[0, -1].item())
            else:
                action_idx = 0

            if action_idx >= len(state.candidates):
                stopped_early = True
                hop_choices.append((None, candidates_now))
                state, _, _, info = env.step(state, STOP_ACTION)
            else:
                hop_choices.append((state.candidates[action_idx]["title"], candidates_now))
                state, _, _, info = env.step(state, action_idx)

        # SF rank at hop1
        first_sf_rank = -1
        if len(hop_choices) >= 1 and gold_titles:
            cands = hop_choices[0][1]
            for sf in gold_titles:
                if sf in cands:
                    first_sf_rank = cands.index(sf)
                    break

        sel = set(state.selected_titles)
        recall_at_k = {}
        for k in self.recall_k_list:
            top_k = set(state.selected_titles[:k])
            recall_at_k[f"recall@{k}"] = float(gold_titles.issubset(top_k)) if gold_titles else 0.0

        return {
            "em":             info.get("em",           0.0),
            "f1":             info.get("f1",           0.0),
            "sf_f1":          info.get("sf_f1",        0.0),
            "sf_recall":      info.get("sf_recall",    0.0),
            "joint_recall":   info.get("joint_recall", 0.0),
            "sf_hit_hop1":    info.get("sf_hit_hop1",  0.0),
            "sf_hit_hop2":    info.get("sf_hit_hop2",  0.0),
            "n_hops":         state.hop,
            "stopped_early":  float(stopped_early),
            "avg_stop_prob":  float(np.mean(stop_probs)) if stop_probs else 0.0,
            "first_sf_rank_hop1": float(first_sf_rank),
            **recall_at_k,
        }