# RL Multi-hop Retrieval on HotpotQA

使用强化学习 (PPO) 优化 Multi-hop 检索路径选择，以 HotpotQA 为数据集。

## 项目结构

```
.
├── main.py           # 主入口
├── data_utils.py     # HotpotQA 数据加载
├── retriever.py      # DPR (dense) + BM25 (sparse) 检索器
├── environment.py    # MDP 环境定义（状态、动作、奖励）
├── policy.py         # 策略网络（文档打分 + STOP token）
├── ppo_trainer.py    # PPO 训练器（GAE + KL 约束）
├── reader.py         # 抽取式阅读器（RoBERTa-SQuAD2）
├── evaluator.py      # 评估指标（EM/F1/SF-F1/Recall@k）
├── baseline.py       # BM25/DPR 贪心基线
├── ablation.py       # 消融实验
└── requirements.txt
```

## 安装

```bash
pip install -r requirements.txt
```

## MDP 定义

| 组件 | 定义 |
|------|------|
| **State** | `(question, [selected docs so far])` |
| **Action** | 从 top-K 候选中选择一篇文档，或 `STOP` |
| **Transition** | 选中文档加入已选列表，重新检索新候选 |
| **Reward** | `w1·EM + w2·F1 + w3·SF-F1 + w4·step_reward` |

### Reward 组成

```python
# 最终奖励（episode 结束时）
R_final = em_weight * EM + f1_weight * F1 + sf_weight * SF_F1

# 阶段性奖励（每步，缓解稀疏问题）
R_step = step_weight if selected_doc in gold_supporting_facts else 0.0

# 总奖励
R = R_final + R_step
```

## 训练

```bash
# 训练 RL 策略（使用 DPR 检索器）
python main.py \
    --mode train \
    --data_dir data/hotpotqa \
    --index_dir data/index \
    --top_k 10 \
    --max_hops 3 \
    --max_episodes 50000 \
    --batch_size 16 \
    --ppo_epochs 4 \
    --clip_epsilon 0.2 \
    --kl_coef 0.1 \
    --use_kl \
    --use_step_reward \
    --output_dir outputs/rl_model

# 如需在 GPU 上运行
python main.py --mode train --device cuda ...
```

## 评估（含基线对比）

```bash
python main.py \
    --mode eval \
    --checkpoint outputs/rl_model/policy_final.pt \
    --data_dir data/hotpotqa \
    --index_dir data/index \
    --output_dir outputs/eval
```

输出 `eval_results.json`，包含：
- `rl_policy`: RL 策略结果
- `bm25_baseline`: BM25 贪心基线
- `dense_baseline`: DPR 贪心基线

## 消融实验

```bash
python main.py \
    --mode ablation \
    --data_dir data/hotpotqa \
    --index_dir data/index \
    --output_dir outputs/ablation
```

消融变体：

| 变体 | 说明 |
|------|------|
| `full_model` | 完整模型（KL + step reward） |
| `no_kl` | 去除 KL 散度约束 |
| `no_step_reward` | 去除阶段性奖励（稀疏化） |
| `sparse_reward` | 仅用 EM 作为最终奖励（最稀疏） |
| `dense_reward` | 加强阶段性奖励（最稠密） |

## 评估指标

- **Answer EM**: 精确匹配率
- **Answer F1**: Token 级 F1
- **Supporting Fact F1**: 支持事实文档覆盖 F1
- **Supporting Fact Recall**: 支持事实文档召回率
- **Recall@k**: 前 k 步中是否命中所有支持事实

## PPO 实现细节

```python
# Clipped PPO objective
L_CLIP = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]

# Value function loss
L_VF = MSE(V(s), G_t)

# KL penalty (ablation)
L_KL = KL(π_θ || π_ref)

# Entropy bonus
L_ENT = -H(π_θ)

# Total loss
L = -L_CLIP + 0.5*L_VF - 0.01*L_ENT + β*L_KL
```

GAE (Generalized Advantage Estimation):
```
δ_t = r_t + γV(s_{t+1}) - V(s_t)
A_t = Σ (γλ)^k δ_{t+k}
```

## 数据准备

数据集会自动下载（HotpotQA 官方链接），或手动放置：
```
data/hotpotqa/
  hotpot_train_v1.1.json
  hotpot_dev_distractor_v1.json
```

索引首次运行时自动构建（约需 30-60 分钟，之后缓存）。
