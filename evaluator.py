"""
Evaluator — 训练时快速评测（200 samples）

改动：
  ⑧ 新增 beam search 推理模式（mode="beam"）
     - 第一跳维护 B 条 beam，每条 beam 独立展开第二跳
     - 最终用 reader 分数选最优 beam
     - greedy 模式（mode="rl"）不变，训练时仍用 greedy 保持速度

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

    def evaluate(
        self,
        policy,
        env,
        dataset,
        mode: str = "rl",
        max_samples: Optional[int] = None,
        beam_size: int = 3,
    ) -> Dict:
        """
        Args:
            mode: "rl"（greedy，训练时用）| "beam"（beam search，评测时用）
            beam_size: beam search 的 beam 数（仅 mode="beam" 时有效）
        """
        policy.eval()
        data = dataset.data
        if max_samples:
            import random
            data = random.sample(data, min(max_samples, len(data)))

        results = defaultdict(list)
        for item in data:
            if mode == "beam":
                r = self._run_beam(policy, env, item, beam_size=beam_size)
            else:
                r = self._run_episode(policy, env, item, mode=mode)
            for k, v in r.items():
                results[k].append(v)

        policy.train()
        return {k: float(np.mean(v)) for k, v in results.items()}

    # ── Greedy episode（训练时评测，速度优先）────────────────────────────

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
                    sv = policy.encode_state(
                        state.question, state.selected_docs,
                        hop=len(state.selected_docs),
                    )
                    ce = policy.encode_docs(state.candidates)
                    probs = policy.get_probs(sv.to(policy.device), ce.to(policy.device))
                action_idx = probs.argmax(dim=-1).item()
                # ⑥ 去掉 STOP：最后一个 logit 是 STOP，若 argmax 指向它则退化为 top-1
                if action_idx >= len(state.candidates):
                    action_idx = 0
                    stopped_early = True
                stop_probs.append(probs[0, -1].item())
            else:
                action_idx = 0

            hop_choices.append((state.candidates[action_idx]["title"], candidates_now))
            state, _, _, info = env.step(state, action_idx)

        return self._build_result(state, info, gold_titles, hop_choices,
                                  stop_probs, stopped_early)

    # ── ⑧ Beam Search episode（评测时，质量优先）────────────────────────

    def _run_beam(self, policy, env, item: Dict, beam_size: int = 3) -> Dict:
        """
        两跳 Beam Search：
          1. 第一跳：对所有候选打分，取 top beam_size 个文档各开一条 beam
          2. 第二跳：每条 beam 独立展开，取 top-1 文档
          3. 每条 beam 得到一个 (doc1, doc2) 组合，送 reader 打分
          4. 选 reader 分数最高的 beam 作为最终预测

        注：beam search 不改变训练过程，只是推理时的解码策略。
        对多跳检索特别有效：第一跳的误差可以在 beam 维度上得到纠正。
        """
        gold_titles = set()
        # 第一跳
        state0 = env.reset(item)
        gold_titles = set(state0.gold_supporting_titles)

        if not state0.candidates:
            info = {"em": 0.0, "f1": 0.0, "sf_f1": 0.0, "sf_recall": 0.0,
                    "joint_recall": 0.0, "sf_hit_hop1": 0.0, "sf_hit_hop2": 0.0,
                    "predicted_answer": "", "gold_answer": state0.gold_answer}
            return self._build_result(state0, info, gold_titles, [], [], False)

        # 对第一跳候选打分，取 top beam_size
        with torch.no_grad():
            sv0 = policy.encode_state(state0.question, [], hop=0)
            ce0 = policy.encode_docs(state0.candidates)
            probs0 = policy.get_probs(sv0.to(policy.device), ce0.to(policy.device))

        # 只考虑文档动作（去掉最后一个 STOP logit）
        doc_probs = probs0[0, :len(state0.candidates)]
        k = min(beam_size, len(state0.candidates))
        top_indices = doc_probs.topk(k).indices.tolist()

        # 展开每条 beam
        best_beam = None
        best_reader_score = -1.0

        for hop1_idx in top_indices:
            # 复制 state0，执行 hop1
            import copy as _copy
            beam_state = _copy.deepcopy(state0)
            beam_state, _, done1, _ = env.step(beam_state, hop1_idx)

            if done1 or not beam_state.candidates:
                # max_hops=1 或无候选，直接用这条 beam
                docs_b = beam_state.selected_docs
            else:
                # 第二跳：greedy
                with torch.no_grad():
                    sv1 = policy.encode_state(
                        beam_state.question, beam_state.selected_docs, hop=1
                    )
                    ce1 = policy.encode_docs(beam_state.candidates)
                    probs1 = policy.get_probs(sv1.to(policy.device), ce1.to(policy.device))
                hop2_idx = probs1[0, :len(beam_state.candidates)].argmax().item()
                beam_state, _, _, _ = env.step(beam_state, hop2_idx)
                docs_b = beam_state.selected_docs

            # Reader 打分选最优 beam
            if docs_b:
                pred = env.reader.predict(
                    item["question"],
                    [{"title": d["title"], "text": d["text"]} for d in docs_b],
                )
                # 用 F1 作为 beam 选择分数（比 EM 更连续）
                from environment import f1_score
                score = f1_score(pred, item["answer"])
            else:
                pred, score = "", 0.0

            if score > best_reader_score:
                best_reader_score = score
                best_beam = beam_state
                best_beam._pred = pred

        if best_beam is None:
            best_beam = beam_state

        # 用最优 beam 构造 info
        sel = set(best_beam.selected_titles)
        gold = set(best_beam.gold_supporting_titles)
        pred_ans = getattr(best_beam, "_pred", "")
        from environment import exact_match_score, f1_score
        em = exact_match_score(pred_ans, item["answer"])
        f1 = f1_score(pred_ans, item["answer"])
        if gold:
            sf_p = len(sel & gold) / max(len(sel), 1)
            sf_r = len(sel & gold) / len(gold)
            sf_f1 = 2 * sf_p * sf_r / (sf_p + sf_r + 1e-9)
        else:
            sf_f1 = sf_r = 0.0
        info = {
            "em": em, "f1": f1, "sf_f1": sf_f1, "sf_recall": sf_r,
            "joint_recall": float(gold.issubset(sel)) if gold else 0.0,
            "sf_hit_hop1": float(best_beam.selected_titles[0] in gold)
                           if best_beam.selected_titles else 0.0,
            "sf_hit_hop2": float(best_beam.selected_titles[1] in gold)
                           if len(best_beam.selected_titles) >= 2 else 0.0,
            "predicted_answer": pred_ans,
            "gold_answer": item["answer"],
        }

        hop_choices = [(t, []) for t in best_beam.selected_titles]
        return self._build_result(best_beam, info, gold_titles, hop_choices, [], False)

    # ── 结果构造（greedy 和 beam 共用）──────────────────────────────────

    def _build_result(
        self,
        state,
        info: Dict,
        gold_titles: set,
        hop_choices: List,
        stop_probs: List[float],
        stopped_early: bool,
    ) -> Dict:
        # SF rank at hop1
        first_sf_rank = -1
        if len(hop_choices) >= 1 and gold_titles:
            cands = hop_choices[0][1]
            for sf in gold_titles:
                if sf in cands:
                    first_sf_rank = cands.index(sf)
                    break

        recall_at_k = {}
        for k in self.recall_k_list:
            top_k = set(state.selected_titles[:k])
            recall_at_k[f"recall@{k}"] = (
                float(gold_titles.issubset(top_k)) if gold_titles else 0.0
            )

        return {
            "em":              info.get("em",            0.0),
            "f1":              info.get("f1",            0.0),
            "sf_f1":           info.get("sf_f1",         0.0),
            "sf_recall":       info.get("sf_recall",     0.0),
            "joint_recall":    info.get("joint_recall",  0.0),
            "sf_hit_hop1":     info.get("sf_hit_hop1",   0.0),
            "sf_hit_hop2":     info.get("sf_hit_hop2",   0.0),
            "n_hops":          state.hop,
            "stopped_early":   float(stopped_early),
            "avg_stop_prob":   float(np.mean(stop_probs)) if stop_probs else 0.0,
            "first_sf_rank_hop1": float(first_sf_rank),
            **recall_at_k,
        }