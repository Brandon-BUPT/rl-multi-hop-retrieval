"""
PPO Trainer — 单塔架构，批量并行 Rollout + 梯度累积
"""
import copy, logging, random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from environment import MultiHopRetrievalEnv, RetrievalState, STOP_ACTION
from policy import build_state_text

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advs, rets = [], []
    gae, nv = 0.0, 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * nv * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advs.insert(0, gae)
        rets.insert(0, gae + values[t])
        nv = values[t]
    return advs, rets


class RolloutBuffer:
    def __init__(self): self.clear()

    def add(self, state_text, hop, state_vec, cand_embs,
            action, log_prob, reward, value, done, info):
        self.state_texts.append(state_text)
        self.hops.append(int(hop))
        self.states.append(state_vec.detach().cpu())
        self.cand_embs.append(cand_embs.detach().cpu())
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))
        self.infos.append(info)

    def clear(self):
        self.state_texts, self.hops, self.states, self.cand_embs = [], [], [], []
        self.actions, self.log_probs, self.rewards, self.values = [], [], [], []
        self.dones, self.infos = [], []

    def __len__(self): return len(self.actions)


class PPOTrainer:

    def __init__(
        self, policy, env: MultiHopRetrievalEnv,
        lr=1e-5, ppo_epochs=4, clip_epsilon=0.2, kl_coef=0.1,
        gamma=0.99, gae_lambda=0.95,
        batch_size=16, rollout_batch_size=64, encode_batch_size=64,
        grad_accum_steps=4, value_loss_coef=0.5, entropy_coef=0.05,
        max_grad_norm=1.0, output_dir="outputs", device="cpu",
        eval_every=512, save_every=1024,
        ref_update_every=200,  # ③ 每隔多少 global_step 更新一次参考策略
    ):
        self.policy = policy
        self.env = env
        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.rollout_batch_size = rollout_batch_size
        self.encode_batch_size = encode_batch_size
        self.grad_accum_steps = grad_accum_steps
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_every = eval_every
        self.save_every = save_every
        self.ref_update_every = ref_update_every  # ③

        # 差异化学习率：encoder 慢，其余正常
        enc_ids = set(id(p) for p in policy.state_encoder.parameters())
        enc_trainable = [p for p in policy.state_encoder.parameters() if p.requires_grad]
        other_params  = [p for p in policy.parameters()
                         if p.requires_grad and id(p) not in enc_ids]
        self.optimizer = AdamW([
            {"params": enc_trainable, "lr": lr * 0.1},
            {"params": other_params,  "lr": lr},
        ], weight_decay=1e-4)
        logger.info(f"Optimizer: encoder {len(enc_trainable)} (lr={lr*0.1:.1e}), "
                    f"other {len(other_params)} (lr={lr:.1e})")

        if kl_coef > 0:
            self.ref_policy = copy.deepcopy(policy)
            for p in self.ref_policy.parameters(): p.requires_grad = False
            self.ref_policy.eval()
            logger.info(f"KL ref_policy enabled, will update every {ref_update_every} steps")
        else:
            self.ref_policy = None

        self.buffer = RolloutBuffer()
        self.global_step = 0
        self.metrics_history = defaultdict(list)

    # ── Encoding ──────────────────────────────────────────────────────────

    def _encode_states(self, texts: List[str], hops: List[int],
                       grad: bool = False) -> torch.Tensor:
        all_vecs = []
        for i in range(0, len(texts), self.encode_batch_size):
            bt = texts[i:i+self.encode_batch_size]
            bh = hops[i:i+self.encode_batch_size]
            enc = self.policy.tokenizer(
                bt, max_length=512, truncation=True, padding=True, return_tensors="pt"
            ).to(self.device)
            if grad:
                out = self.policy.state_encoder(**enc)
            else:
                with torch.no_grad():
                    out = self.policy.state_encoder(**enc)
            h = (out.pooler_output if (hasattr(out, "pooler_output") and out.pooler_output is not None)
                 else out.last_hidden_state[:, 0, :])
            vecs = self.policy.state_proj(h)
            hop_idx = torch.tensor([min(x, 3) for x in bh], device=self.device)
            vecs = vecs + self.policy.hop_embedding(hop_idx)
            all_vecs.append(vecs)
        return torch.cat(all_vecs, dim=0)

    @torch.no_grad()
    def _encode_docs(self, docs: List[Dict]) -> torch.Tensor:
        """单塔：文档和 state 用同一 encoder + state_proj"""
        all_embs = []
        bs = self.encode_batch_size * 2
        for i in range(0, len(docs), bs):
            batch = docs[i:i+bs]
            texts = [f"{d['title']}. {d['text'][:400]}" for d in batch]
            enc = self.policy.tokenizer(
                texts, max_length=256, truncation=True, padding=True, return_tensors="pt"
            ).to(self.device)
            out = self.policy.state_encoder(**enc)
            h = (out.pooler_output if (hasattr(out, "pooler_output") and out.pooler_output is not None)
                 else out.last_hidden_state[:, 0, :])
            all_embs.append(self.policy.state_proj(h))
        return torch.cat(all_embs, dim=0)

    # ── Rollout ───────────────────────────────────────────────────────────

    def collect_rollout_batch(self, items: List[Dict]) -> List[Dict]:
        N = len(items)
        states = [self.env.reset(item) for item in items]
        local_trans: List[List] = [[] for _ in range(N)]
        ep_infos: List[Dict] = [{} for _ in range(N)]
        active = list(range(N))

        while active:
            texts = [build_state_text(states[i].question, states[i].selected_docs) for i in active]
            hops  = [len(states[i].selected_docs) for i in active]
            has_cands = [len(states[i].candidates) > 0 for i in active]

            state_vecs = self._encode_states(texts, hops)

            # 批量编码候选文档
            all_docs, sizes = [], []
            for j, i in enumerate(active):
                if has_cands[j]:
                    all_docs.extend(states[i].candidates)
                    sizes.append(len(states[i].candidates))
                else:
                    sizes.append(0)

            cand_embs_list = [None] * len(active)
            if all_docs:
                all_embs = self._encode_docs(all_docs)
                offset = 0
                for j, sz in enumerate(sizes):
                    if sz > 0:
                        cand_embs_list[j] = all_embs[offset:offset+sz]
                        offset += sz

            # 批量 forward
            has_idx = [j for j, h in enumerate(has_cands) if h]
            acts, lps, vals = None, None, None
            if has_idx:
                max_k = max(cand_embs_list[j].shape[0] for j in has_idx)
                padded = []
                for j in has_idx:
                    ce = cand_embs_list[j]
                    if ce.shape[0] < max_k:
                        ce = torch.cat([ce, torch.zeros(max_k-ce.shape[0], ce.shape[1], device=self.device)])
                    padded.append(ce.unsqueeze(0))
                batch_cands = torch.cat(padded, dim=0)
                batch_svecs = state_vecs[has_idx]
                with torch.no_grad():
                    logits, vals = self.policy.forward(batch_svecs, batch_cands)
                    dist = torch.distributions.Categorical(logits=logits)
                    acts = dist.sample()
                    lps  = dist.log_prob(acts)

            next_active = []
            ptr = 0
            for j, i in enumerate(active):
                s = states[i]
                sv = state_vecs[j].unsqueeze(0)
                if has_cands[j]:
                    av, lv, vv = acts[ptr].item(), lps[ptr].item(), vals[ptr].item()
                    ce = cand_embs_list[j]; ptr += 1
                else:
                    av = lv = vv = 0
                    ce = torch.zeros(1, self.policy.hidden_dim, device=self.device)

                env_action = STOP_ACTION if av >= len(s.candidates) else av
                s, reward, done, info = self.env.step(s, env_action)
                states[i] = s
                local_trans[i].append((texts[j], hops[j], sv.cpu(), ce.cpu(),
                                       av, lv, reward, vv, done))
                if done: ep_infos[i] = info
                else: next_active.append(i)
            active = next_active

        ep_results = []
        for i in range(N):
            trans = local_trans[i]
            if not trans: continue
            rews  = [t[6] for t in trans]
            vals_ = [t[7] for t in trans]
            dones = [t[8] for t in trans]
            advs, rets = compute_gae(rews, vals_, dones, self.gamma, self.gae_lambda)
            for k, t in enumerate(trans):
                st, hop, sv, ce, ac, lp, rw, vl, dn = t
                self.buffer.add(st, hop, sv, ce, ac, lp, rews[k], vl, dn,
                                {"advantage": advs[k], "return": rets[k]})
            info = ep_infos[i]
            ep_results.append({
                "episode_reward": sum(rews),
                "em":           info.get("em",           0.0),
                "f1":           info.get("f1",           0.0),
                "sf_f1":        info.get("sf_f1",        0.0),
                "joint_recall": info.get("joint_recall", 0.0),
                "sf_hit_hop1":  info.get("sf_hit_hop1",  0.0),
                "sf_hit_hop2":  info.get("sf_hit_hop2",  0.0),
                "n_hops": len(trans),
            })
        return ep_results

    # ── PPO Update ────────────────────────────────────────────────────────

    def update(self) -> Dict:
        if not self.buffer: return {}
        N = len(self.buffer)
        actions    = torch.tensor(self.buffer.actions,   device=self.device)
        old_lps    = torch.tensor(self.buffer.log_probs, device=self.device)
        advantages = torch.tensor([i["advantage"] for i in self.buffer.infos], device=self.device)
        returns    = torch.tensor([i["return"]    for i in self.buffer.infos], device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        max_k = max(ce.shape[0] for ce in self.buffer.cand_embs)
        padded = []
        for ce in self.buffer.cand_embs:
            if ce.shape[0] < max_k:
                ce = torch.cat([ce, torch.zeros(max_k-ce.shape[0], ce.shape[1])], dim=0)
            padded.append(ce.unsqueeze(0))
        all_cands_cpu = torch.cat(padded, dim=0)

        texts = self.buffer.state_texts
        hops  = self.buffer.hops
        metrics = defaultdict(float)
        n_upd = 0

        for _ in range(self.ppo_epochs):
            idx = torch.randperm(N).tolist()
            self.optimizer.zero_grad()
            accum = 0

            for start in range(0, N, self.batch_size):
                bidx = idx[start:start+self.batch_size]
                if not bidx: continue

                b_states = self._encode_states([texts[i] for i in bidx],
                                               [hops[i]  for i in bidx], grad=True)
                b_cands  = all_cands_cpu[bidx].to(self.device)
                b_acts   = actions[bidx]
                b_olps   = old_lps[bidx]
                b_adv    = advantages[bidx]
                b_ret    = returns[bidx]

                _, new_lps, entropy, values = self.policy.get_action_and_value(
                    b_states, b_cands, b_acts)
                ratio    = (new_lps - b_olps).exp()
                clip_adv = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * b_adv
                pol_loss = -torch.min(ratio * b_adv, clip_adv).mean()
                val_loss = F.mse_loss(values.squeeze(-1), b_ret)
                ent_loss = -entropy.mean()

                kl_loss = torch.tensor(0.0, device=self.device)
                if self.ref_policy is not None:
                    with torch.no_grad():
                        ref_logits, _ = self.ref_policy.forward(b_states.detach(), b_cands)
                    curr_logits, _ = self.policy.forward(b_states, b_cands)
                    kl_loss = F.kl_div(F.log_softmax(curr_logits, dim=-1),
                                       F.softmax(ref_logits, dim=-1), reduction="batchmean")

                loss = (pol_loss
                        + self.value_loss_coef * val_loss
                        + self.entropy_coef * ent_loss
                        + self.kl_coef * kl_loss) / self.grad_accum_steps
                loss.backward()
                accum += 1

                metrics["policy_loss"] += pol_loss.item()
                metrics["value_loss"]  += val_loss.item()
                metrics["entropy"]     += entropy.mean().item()
                metrics["kl_loss"]     += kl_loss.item()
                metrics["total_loss"]  += loss.item() * self.grad_accum_steps
                n_upd += 1

                if accum % self.grad_accum_steps == 0:
                    nn.utils.clip_grad_norm_(self.policy.trainable_parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if accum % self.grad_accum_steps != 0:
                nn.utils.clip_grad_norm_(self.policy.trainable_parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        return {k: v / max(n_upd, 1) for k, v in metrics.items()}

    # ── Train Loop ────────────────────────────────────────────────────────

    def train(self, train_data, dev_data, max_episodes: int,
              evaluator=None, hotpot_evaluator=None):
        logger.info(f"Training: {max_episodes} eps, rollout={self.rollout_batch_size}, "
                    f"batch={self.batch_size}, accum={self.grad_accum_steps}")
        ep_m = defaultdict(list)
        total, best_joint = 0, -1.0

        while total < max_episodes:
            items = random.choices(train_data.data,
                                   k=min(self.rollout_batch_size, max_episodes - total))
            results = self.collect_rollout_batch(items)
            total += len(items)
            for r in results:
                for k, v in r.items(): ep_m[k].append(v)

            if len(self.buffer) >= self.batch_size * self.grad_accum_steps:
                upd = self.update()
                self.buffer.clear()
                self.global_step += 1

                # ③ 周期性更新 KL 参考策略
                # 固定初始参考策略会随训练推进导致 KL 越来越大从而限制探索；
                # 每隔 ref_update_every 步将当前策略快照为新的参考策略。
                if (self.ref_policy is not None
                        and self.ref_update_every > 0
                        and self.global_step % self.ref_update_every == 0):
                    self.ref_policy.load_state_dict(
                        copy.deepcopy(self.policy.state_dict())
                    )
                    logger.info(f"[Step {self.global_step}] ref_policy updated")

                if self.global_step % 10 == 0:
                    W = self.rollout_batch_size * 10
                    logger.info(
                        f"[Step {self.global_step} | Ep {total}] "
                        f"reward={np.mean(ep_m['episode_reward'][-W:]):.4f} "
                        f"EM={np.mean(ep_m['em'][-W:]):.4f} "
                        f"F1={np.mean(ep_m['f1'][-W:]):.4f} "
                        f"joint={np.mean(ep_m['joint_recall'][-W:]):.3f} "
                        f"h1={np.mean(ep_m['sf_hit_hop1'][-W:]):.3f} "
                        f"h2={np.mean(ep_m['sf_hit_hop2'][-W:]):.3f} | "
                        + " ".join(f"{k}={v:.4f}" for k, v in upd.items())
                    )

            if evaluator is not None and total % self.eval_every < self.rollout_batch_size:
                m = evaluator.evaluate(self.policy, self.env, dev_data, mode="rl", max_samples=200)
                logger.info(f"[Ep {total}] Dev: {m}")
                jt = m.get("joint_recall", 0.0)
                if jt > best_joint:
                    best_joint = jt
                    torch.save(self.policy.state_dict(), self.output_dir / "policy_best.pt")
                    logger.info(f"New best joint_recall={jt:.4f}")
                if hotpot_evaluator is not None and self.global_step % 50 == 0:
                    hm = hotpot_evaluator.evaluate(self.policy, self.env, dev_data,
                                                    mode="rl", max_samples=500)
                    logger.info(f"[Ep {total}] Official: {hm}")

            if total % self.save_every < self.rollout_batch_size:
                ckpt = self.output_dir / f"policy_ep{total}.pt"
                torch.save(self.policy.state_dict(), ckpt)
                logger.info(f"Saved: {ckpt}")

        torch.save(self.policy.state_dict(), self.output_dir / "policy_final.pt")
        logger.info("Training complete.")