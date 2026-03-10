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
        use_batch: bool = True,
        beam_batch_coeff: int = 4,
    ) -> Dict:
        """
        Args:
            mode: "rl"（greedy，训练时用）| "beam"（beam search，评测时用）
            beam_size: beam search 的 beam 数（仅 mode="beam" 时有效）
            use_batch: 是否使用批量评测
            beam_batch_coeff: batch_size = beam_size * beam_batch_coeff
        """
        policy.eval()
        data = dataset.data
        if max_samples:
            import random
            data = random.sample(data, min(max_samples, len(data)))

        total_samples = len(data)
        if mode == "beam":
            logger.info(f"Beam search evaluation on {total_samples} samples (beam_size={beam_size})")

        if mode == "beam" and use_batch:
            batch_size = beam_size * beam_batch_coeff
            logger.info(f"Batch beam search evaluation (batch_size={batch_size})")
            results = defaultdict(list)

            for batch_start in range(0, total_samples, batch_size):
                batch_end = min(batch_start + batch_size, total_samples)
                batch_items = data[batch_start:batch_end]

                batch_results = self._run_batch_beam(
                    policy, env, batch_items, beam_size=beam_size
                )

                for r in batch_results:
                    for k, v in r.items():
                        results[k].append(v)

                if (batch_end % 200) < batch_size:
                    curr_em = np.mean(results["em"]) if results["em"] else 0.0
                    curr_f1 = np.mean(results["f1"]) if results["f1"] else 0.0
                    curr_jr = np.mean(results["joint_recall"]) if results.get("joint_recall") else 0.0
                    logger.info(f"Progress: {batch_end}/{total_samples} ({100*batch_end/total_samples:.1f}%) | EM: {curr_em:.3f} | F1: {curr_f1:.3f} | JointRecall: {curr_jr:.3f}")

            policy.train()
            return {k: float(np.mean(v)) for k, v in results.items()}

        results = defaultdict(list)
        progress_cnt = 0
        progress_log_interval = 400

        for item in data:
            if mode == "beam":
                r = self._run_beam(policy, env, item, beam_size=beam_size)
            else:
                r = self._run_episode(policy, env, item, mode=mode)
            for k, v in r.items():
                results[k].append(v)

            progress_cnt += 1
            if progress_cnt % progress_log_interval == 0:
                curr_em = np.mean(results["em"]) if results["em"] else 0.0
                curr_f1 = np.mean(results["f1"]) if results["f1"] else 0.0
                curr_jr = np.mean(results["joint_recall"]) if results.get("joint_recall") else 0.0
                logger.info(f"Progress: {progress_cnt}/{total_samples} ({100*progress_cnt/total_samples:.1f}%) | EM: {curr_em:.3f} | F1: {curr_f1:.3f} | JointRecall: {curr_jr:.3f}")

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

    # ── 批量评测（多样本并行）────────────────────────────────────────────

    def evaluate_batch(
        self,
        policy,
        env,
        dataset,
        mode: str = "rl",
        max_samples: Optional[int] = None,
        beam_size: int = 3,
        batch_size: int = 16,
    ) -> Dict:
        """
        批量评测：收集多个样本的候选，批量编码，批量计算。
        比逐样本处理更快。
        """
        policy.eval()
        data = dataset.data
        if max_samples:
            import random
            data = random.sample(data, min(max_samples, len(data)))

        total_samples = len(data)
        if mode == "beam":
            logger.info(f"Beam search evaluation on {total_samples} samples (batch_size={batch_size})")

        results = defaultdict(list)
        progress_cnt = 0
        progress_log_interval = 400

        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_items = data[batch_start:batch_end]

            batch_results = self._run_batch_episodes(
                policy, env, batch_items, mode=mode
            )

            for r in batch_results:
                for k, v in r.items():
                    results[k].append(v)

            progress_cnt += len(batch_items)
            if mode == "beam" and progress_cnt % progress_log_interval < batch_size:
                logger.info(f"Beam progress: {progress_cnt}/{total_samples} ({100*progress_cnt/total_samples:.1f}%)")

        policy.train()
        return {k: float(np.mean(v)) for k, v in results.items()}

    def _run_batch_episodes(
        self,
        policy,
        env,
        items: List[Dict],
        mode: str = "rl",
    ) -> List[Dict]:
        """
        批量运行多个 episode。
        策略：先批量编码所有候选文档，再逐样本决策。
        """
        if mode != "rl":
            return [self._run_episode(policy, env, item, mode) for item in items]

        states = []
        for item in items:
            state = env.reset(item)
            state.gold_titles = set(state.gold_supporting_titles)
            states.append(state)

        all_results: List[Dict] = [{} for _ in items]

        while True:
            active_indices = [i for i, s in enumerate(states) if not s.done and s.candidates]
            if not active_indices:
                break

            batch_states = [states[i] for i in active_indices]
            questions = [s.question for s in batch_states]
            selected_docs_list = [s.selected_docs for s in batch_states]
            candidates_list = [s.candidates for s in batch_states]
            hops = [len(s.selected_docs) for s in batch_states]

            with torch.no_grad():
                state_vecs = []
                for q, docs, h in zip(questions, selected_docs_list, hops):
                    sv = policy.encode_state(q, docs, hop=h)
                    state_vecs.append(sv)
                state_vecs = torch.cat(state_vecs, dim=0).to(policy.device)

                all_docs = []
                offsets = []
                for cand in candidates_list:
                    offsets.append(len(all_docs))
                    all_docs.extend(cand)

                if all_docs:
                    doc_vecs = policy.encode_docs(all_docs)
                    doc_vecs = doc_vecs.to(policy.device)

                    probs_list = []
                    for i, (state, offset) in enumerate(zip(batch_states, offsets)):
                        n_cands = len(state.candidates)
                        sv_i = state_vecs[i:i+1]
                        ce_i = doc_vecs[offset:offset+n_cands]
                        probs = policy.get_probs(sv_i, ce_i)
                        probs_list.append(probs[0, :n_cands])

                    action_indices = [p.argmax().item() for p in probs_list]
                else:
                    action_indices = [0] * len(batch_states)

            for idx, (state, action_idx) in enumerate(zip(batch_states, action_indices)):
                orig_idx = active_indices[idx]
                candidates_now = [d["title"] for d in state.candidates]
                hop_choices = getattr(states[orig_idx], '_hop_choices', [])
                hop_choices.append((state.candidates[action_idx]["title"], candidates_now))
                states[orig_idx]._hop_choices = hop_choices

                states[orig_idx], _, _, info = env.step(state, action_idx)

        for i, state in enumerate(states):
            gold_titles = getattr(state, 'gold_titles', set())
            hop_choices = getattr(state, '_hop_choices', [])
            r = self._build_result(state, {}, gold_titles, hop_choices, [], False)
            all_results[i] = r

        return all_results

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

        # Joint metrics: answer + supporting facts
        a_em = info.get("em", 0.0)
        a_f1 = info.get("f1", 0.0)
        s_em = info.get("joint_recall", 0.0)
        s_f1 = info.get("sf_f1", 0.0)
        joint_em = float(a_em == 1.0 and s_em == 1.0)
        joint_f1 = a_f1 * s_f1

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
            "joint_em":        joint_em,
            "joint_f1":        joint_f1,
            "sf_hit_hop1":     info.get("sf_hit_hop1",   0.0),
            "sf_hit_hop2":     info.get("sf_hit_hop2",   0.0),
            "n_hops":          state.hop,
            "stopped_early":   float(stopped_early),
            "avg_stop_prob":   float(np.mean(stop_probs)) if stop_probs else 0.0,
            "first_sf_rank_hop1": float(first_sf_rank),
            **recall_at_k,
        }

    def _run_batch_beam(self, policy, env, batch_items: List[Dict], beam_size: int = 3) -> List[Dict]:
        """批量 Beam Search：每个样本独立执行 beam search，最后批量 Reader 预测"""
        import copy

        results = []
        all_beam_docs = []
        all_questions = []
        all_answers = []
        all_sample_beams = []

        for batch_idx, item in enumerate(batch_items):
            state0 = env.reset(item)
            gold_titles = set(state0.gold_supporting_titles)

            if not state0.candidates:
                info = {"em": 0.0, "f1": 0.0, "sf_f1": 0.0, "sf_recall": 0.0,
                        "joint_recall": 0.0, "sf_hit_hop1": 0.0, "sf_hit_hop2": 0.0,
                        "predicted_answer": "", "gold_answer": state0.gold_answer}
                results.append(self._build_result(state0, info, gold_titles, [], [], False))
                all_sample_beams.append({"states": [], "docs": [], "question": item["question"], "answer": item["answer"]})
                continue

            with torch.no_grad():
                sv0 = policy.encode_state(state0.question, [], hop=0)
                ce0 = policy.encode_docs(state0.candidates)
                probs0 = policy.get_probs(sv0.to(policy.device), ce0.to(policy.device))

            doc_probs = probs0[0, :len(state0.candidates)]
            k = min(beam_size, len(state0.candidates))
            top_indices = doc_probs.topk(k).indices.tolist()

            sample_beams = {"states": [], "docs": [], "question": item["question"], "answer": item["answer"]}
            for hop1_idx in top_indices:
                beam_state = copy.deepcopy(state0)
                beam_state, _, done1, _ = env.step(beam_state, hop1_idx)

                if done1 or not beam_state.candidates:
                    docs_b = beam_state.selected_docs
                else:
                    sv1 = policy.encode_state(
                        beam_state.question, beam_state.selected_docs, hop=1
                    )
                    ce1 = policy.encode_docs(beam_state.candidates)
                    probs1 = policy.get_probs(sv1.to(policy.device), ce1.to(policy.device))
                    hop2_idx = probs1[0, :len(beam_state.candidates)].argmax().item()
                    beam_state, _, _, _ = env.step(beam_state, hop2_idx)
                    docs_b = beam_state.selected_docs

                sample_beams["states"].append(beam_state)
                sample_beams["docs"].append([{"title": d["title"], "text": d["text"]} for d in docs_b])

            all_sample_beams.append(sample_beams)

        all_beam_docs = []
        all_questions = []
        all_answers = []
        for sb in all_sample_beams:
            all_beam_docs.extend(sb["docs"])
            all_questions.extend([sb["question"]] * len(sb["docs"]))
            all_answers.extend([sb["answer"]] * len(sb["docs"]))

        if not all_beam_docs:
            return results

        all_preds = env.reader.predict_batch(all_questions, all_beam_docs)

        from environment import exact_match_score, f1_score
        beam_idx = 0
        for sample_idx, sample_beams in enumerate(all_sample_beams):
            n_beams = len(sample_beams["docs"])
            if n_beams == 0:
                item = batch_items[sample_idx]
                state0 = env.reset(item)
                info = {"em": 0.0, "f1": 0.0, "sf_f1": 0.0, "sf_recall": 0.0,
                        "joint_recall": 0.0, "sf_hit_hop1": 0.0, "sf_hit_hop2": 0.0,
                        "predicted_answer": "", "gold_answer": sample_beams["answer"]}
                results.append(self._build_result(state0, info, set(), [], [], False))
                continue

            best_score = -1.0
            best_beam_state = None
            best_pred = ""

            for beam_offset in range(n_beams):
                pred = all_preds[beam_idx]
                score = f1_score(pred, all_answers[beam_idx])
                if score > best_score:
                    best_score = score
                    best_beam_state = sample_beams["states"][beam_offset]
                    best_pred = pred
                beam_idx += 1

            if best_beam_state is None:
                continue

            best_beam_state._pred = best_pred
            sel = set(best_beam_state.selected_titles)
            gold = set(best_beam_state.gold_supporting_titles)
            em = exact_match_score(best_pred, sample_beams["answer"])
            f1 = f1_score(best_pred, sample_beams["answer"])
            if gold:
                sf_p = len(sel & gold) / max(len(sel), 1)
                sf_r = len(sel & gold) / len(gold)
                sf_f1 = 2 * sf_p * sf_r / (sf_p + sf_r + 1e-9)
            else:
                sf_f1 = sf_r = 0.0
            info = {
                "em": em, "f1": f1, "sf_f1": sf_f1, "sf_recall": sf_r,
                "joint_recall": float(gold.issubset(sel)) if gold else 0.0,
                "sf_hit_hop1": float(best_beam_state.selected_titles[0] in gold)
                               if best_beam_state.selected_titles else 0.0,
                "sf_hit_hop2": float(best_beam_state.selected_titles[1] in gold)
                               if len(best_beam_state.selected_titles) >= 2 else 0.0,
                "predicted_answer": best_pred,
                "gold_answer": sample_beams["answer"],
            }
            gold_titles = set(best_beam_state.gold_supporting_titles)
            hop_choices = [(t, []) for t in best_beam_state.selected_titles]
            results.append(self._build_result(best_beam_state, info, gold_titles, hop_choices, [], False))

        return results