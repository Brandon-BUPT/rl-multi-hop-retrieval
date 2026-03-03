"""
RL Multi-hop Retrieval — Main Entry Point（重构版）

新增参数：
  --encoder_backend   : "auto" | "sentence_transformer" | "huggingface"
  --encoder_model     : 任意 sentence-transformers 或 HuggingFace 模型名称
                        推荐：
                          BAAI/bge-base-en-v1.5        (高质量，768d)
                          sentence-transformers/all-MiniLM-L6-v2  (极快，384d)
                          intfloat/e5-base-v2           (均衡，768d)
  --dense_index_type  : "flat" | "ivf"（大语料建议 ivf）
  --freeze_ratio      : 按比例冻结 transformer 层（0.0~1.0），
                        与 --freeze_layers 二选一

删除参数：
  --ctx_encoder_model（旧 DPR ctx encoder，已不需要）

其余参数与旧版完全兼容。
"""
import argparse
import json
import logging
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*overflowing tokens.*", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 屏蔽第三方库的噪声日志
for _noisy in ["httpx", "httpcore", "sentence_transformers", "huggingface_hub",
               "transformers.modeling_utils", "filelock"]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="RL Multi-hop Retrieval (refactored)")

    # ── 基础 ──────────────────────────────────────────────────────────────
    p.add_argument("--mode",      choices=["train", "eval"], default="train")
    p.add_argument("--data_dir",  default="data/hotpotqa")
    p.add_argument("--index_dir", default="data/index")
    p.add_argument("--cache_dir", default="cache")

    # ── 编码器（重构核心）─────────────────────────────────────────────────
    p.add_argument(
        "--encoder_model",
        default="BAAI/bge-base-en-v1.5",
        help=(
            "编码器模型名称。推荐：\n"
            "  BAAI/bge-base-en-v1.5  (高质量，768d)\n"
            "  sentence-transformers/all-MiniLM-L6-v2  (极快，384d)\n"
            "  intfloat/e5-base-v2  (均衡，768d)\n"
            "  BAAI/bge-m3  (多语言)"
        ),
    )
    p.add_argument(
        "--encoder_backend",
        default="auto",
        choices=["auto", "sentence_transformer", "huggingface"],
        help="编码后端类型。auto=优先 sentence-transformers，fallback HuggingFace",
    )
    p.add_argument(
        "--query_prefix",
        default="",
        help="查询前缀（e5 系列需要 'query: '）",
    )
    p.add_argument(
        "--doc_prefix",
        default="",
        help="文档前缀（e5 系列需要 'passage: '）",
    )

    # ── 检索 ──────────────────────────────────────────────────────────────
    p.add_argument("--reader_model",     default="deepset/roberta-base-squad2")
    p.add_argument("--freeze_layers",    type=int,   default=6,
                   help="冻结底部 N 层（-1=不冻结；与 freeze_ratio 二选一）")
    p.add_argument("--freeze_ratio",     type=float, default=0.5,
                   help="按比例冻结（0.0~1.0，freeze_layers>=0 时优先）")
    p.add_argument("--policy_hidden",    type=int,   default=None,
                   help="投影后隐藏维度，None=与编码器维度一致")
    p.add_argument("--retrieval_mode",   default="context",
                   choices=["context", "bm25", "dpr", "dense", "hybrid"])
    p.add_argument("--top_k",            type=int,   default=10)
    p.add_argument("--max_hops",         type=int,   default=2)
    p.add_argument("--index_batch_size", type=int,   default=512)
    p.add_argument("--dense_index_type", default="flat", choices=["flat", "ivf"],
                   help="FAISS 索引类型（大语料用 ivf）")

    # ── PPO 超参 ──────────────────────────────────────────────────────────
    p.add_argument("--ppo_epochs",         type=int,   default=4)
    p.add_argument("--clip_epsilon",       type=float, default=0.2)
    p.add_argument("--kl_coef",            type=float, default=0.1)
    p.add_argument("--use_kl",             action="store_true", default=True)
    p.add_argument("--gamma",              type=float, default=0.99)
    p.add_argument("--gae_lambda",         type=float, default=0.95)
    p.add_argument("--lr",                 type=float, default=1e-5)
    p.add_argument("--batch_size",         type=int,   default=16)
    p.add_argument("--rollout_batch_size", type=int,   default=64)
    p.add_argument("--encode_batch_size",  type=int,   default=64)
    p.add_argument("--grad_accum_steps",   type=int,   default=4)
    p.add_argument("--max_episodes",       type=int,   default=100000)
    p.add_argument("--eval_every",         type=int,   default=512)
    p.add_argument("--save_every",         type=int,   default=1024)
    p.add_argument("--ref_update_every",   type=int,   default=200)
    p.add_argument("--beam_size",          type=int,   default=3)

    # ── 奖励 ──────────────────────────────────────────────────────────────
    p.add_argument("--reward_em_weight",          type=float, default=0.3)
    p.add_argument("--reward_f1_weight",          type=float, default=0.2)
    p.add_argument("--reward_sf_weight",          type=float, default=1.0)
    p.add_argument("--reward_joint_bonus",        type=float, default=0.5)
    p.add_argument("--reward_step_weight",        type=float, default=0.2)
    p.add_argument("--reward_early_stop_penalty", type=float, default=0.0)
    p.add_argument("--reward_min_hops",           type=int,   default=2)
    p.add_argument("--use_step_reward",           action="store_true", default=True)

    # ── 其他 ──────────────────────────────────────────────────────────────
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--recall_k",   type=int, nargs="+", default=[1, 2, 5, 10])
    p.add_argument("--max_samples", type=int, default=None,
                   help="eval 时采样数量，None=全量 dev set")

    return p.parse_args()


def build_retriever(args, corpus=None):
    """根据 retrieval_mode 构建检索器"""
    from retriever import (
        BM25Retriever, DenseRetriever, HybridRetriever,
        create_encoder_backend,
    )

    bm25 = BM25Retriever(index_dir=args.index_dir)

    # dense / dpr / hybrid 模式都需要构建向量索引
    if args.retrieval_mode in ("dense", "dpr", "hybrid"):
        backend = create_encoder_backend(
            model_name=args.encoder_model,
            backend_type=args.encoder_backend,
            device=args.device,
            cache_dir=args.cache_dir,
            query_prefix=args.query_prefix,
            doc_prefix=args.doc_prefix,
        )
        dense = DenseRetriever(
            encoder=backend,
            index_dir=args.index_dir,
            batch_size=args.index_batch_size,
            index_type=args.dense_index_type,
        )
        if corpus is not None:
            if args.retrieval_mode in ("dense", "dpr"):
                dense.build_index(corpus)
                return dense
            else:  # hybrid
                bm25.build_index(corpus)
                dense.build_index(corpus)
                return HybridRetriever(bm25, dense)
        return dense if args.retrieval_mode in ("dense", "dpr") else HybridRetriever(bm25, dense)

    elif args.retrieval_mode == "bm25":
        if corpus is not None:
            bm25.build_index(corpus)
        return bm25

    else:  # context oracle
        return bm25  # 不会被调用，placeholder


def _run_eval(args, policy, env, dev_data, encoder_backend):
    """
    eval 模式统一入口：
      - 无 checkpoint：只跑 Greedy baseline（BM25 + Dense）
      - 有 checkpoint：跑 RL beam search，同时跑 Greedy baseline，对齐输出
    """
    from evaluator import Evaluator
    from eval_greedy_baseline import (
        evaluate_bm25_greedy, evaluate_dense_greedy, DenseGreedy
    )
    from reader import ExtractiveReader

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载原始 dev json（greedy baseline 直接读 json，不走 Dataset）
    dev_path = os.path.join(args.data_dir, "hotpot_dev_distractor.json")
    with open(dev_path) as f:
        dev_raw = json.load(f)
    if args.max_samples:
        dev_raw = random.sample(dev_raw, min(args.max_samples, len(dev_raw)))
    logger.info(f"Eval on {len(dev_raw)} samples")

    reader = env.reader
    all_results = {}

    # ── 1. BM25 Greedy ────────────────────────────────────────────────────
    # logger.info("=== Running BM25 Greedy baseline ===")
    # bm25_m = evaluate_bm25_greedy(dev_raw, reader, max_hops=args.max_hops)
    # all_results["BM25 Greedy"] = bm25_m

    # ── 2. Dense Greedy（用同一个 encoder_backend）────────────────────────
    logger.info(f"=== Running Dense Greedy baseline ({args.encoder_model}) ===")
    dense_greedy = DenseGreedy(
        encoder_model=args.encoder_model,
        backend_type=args.encoder_backend,
        device=args.device,
        cache_dir=args.cache_dir,
        query_prefix=args.query_prefix,
        doc_prefix=args.doc_prefix,
        batch_size=args.encode_batch_size,
    )
    dense_m = evaluate_dense_greedy(dev_raw, reader, dense_greedy, max_hops=args.max_hops)
    dense_label = f"Dense Greedy ({args.encoder_model.split('/')[-1]})"
    all_results[dense_label] = dense_m

    # ── 3. RL Policy（仅有 checkpoint 时）────────────────────────────────
    if args.checkpoint:
        logger.info("=== Running RL Policy (beam search) ===")
        evaluator = Evaluator(recall_k_list=args.recall_k)
        rl_m = evaluator.evaluate(
            policy, env, dev_data,
            mode="beam",
            beam_size=args.beam_size,
        )
        ckpt_name = Path(args.checkpoint).stem   # e.g. "policy_best"
        all_results[f"RL ({ckpt_name})"] = rl_m
    else:
        logger.info("No checkpoint provided — skipping RL eval")

    # 保存 JSON
    save_path = os.path.join(args.output_dir, "eval_results.json")
    json.dump(all_results, open(save_path, "w"), indent=2)
    logger.info(f"Results saved to {save_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    logger.info(f"Mode: {args.mode} | Device: {args.device}")
    logger.info(f"Encoder: {args.encoder_model} (backend={args.encoder_backend})")

    from data_utils import HotpotQADataset
    from retriever import create_encoder_backend
    from environment import MultiHopRetrievalEnv
    from policy import PolicyNetwork
    from ppo_trainer import PPOTrainer
    from reader import ExtractiveReader
    from evaluator import Evaluator
    from hotpot_eval import HotpotEvaluator

    # ── 数据 ──────────────────────────────────────────────────────────────
    train_data = HotpotQADataset(args.data_dir, split="train",          cache_dir=args.cache_dir)
    dev_data   = HotpotQADataset(args.data_dir, split="dev_distractor", cache_dir=args.cache_dir)
    logger.info(f"Train: {len(train_data)}, Dev: {len(dev_data)}")

    # ── 编码器 Backend（policy 和 retriever 共享同一 backend）────────────
    encoder_backend = create_encoder_backend(
        model_name=args.encoder_model,
        backend_type=args.encoder_backend,
        device=args.device,
        cache_dir=args.cache_dir,
        query_prefix=args.query_prefix,
        doc_prefix=args.doc_prefix,
    )
    logger.info(f"Encoder backend ready: dim={encoder_backend.dim}")

    # ── 检索器 ────────────────────────────────────────────────────────────
    if args.retrieval_mode == "context":
        retriever = None  # oracle 模式不需要外部检索器
        logger.info("Context oracle mode — skipping index build")
    else:
        corpus = {**train_data.corpus, **dev_data.corpus}
        logger.info(f"Corpus: {len(corpus)} passages")
        retriever = build_retriever(args, corpus)

    # ── Reader ────────────────────────────────────────────────────────────
    reader = ExtractiveReader(model_name=args.reader_model, device=args.device)

    # ── 环境 ──────────────────────────────────────────────────────────────
    reward_config = {
        "sf_weight":            args.reward_sf_weight,
        "joint_bonus":          args.reward_joint_bonus,
        "step_weight":          args.reward_step_weight,
        "use_step_reward":      args.use_step_reward,
        "em_weight":            args.reward_em_weight,
        "f1_weight":            args.reward_f1_weight,
        "early_stop_penalty":   args.reward_early_stop_penalty,
        "min_hops_before_stop": args.reward_min_hops,
    }
    env = MultiHopRetrievalEnv(
        retriever=retriever,
        reader=reader,
        top_k=args.top_k,
        max_hops=args.max_hops,
        retrieval_mode=args.retrieval_mode,
        reward_config=reward_config,
    )

    # ── Policy（共享 encoder_backend）────────────────────────────────────
    policy = PolicyNetwork(
        encoder_backend=encoder_backend,
        hidden_dim=args.policy_hidden,
        freeze_layers=args.freeze_layers,
        freeze_ratio=args.freeze_ratio,
        device=args.device,
        cache_dir=args.cache_dir,
    ).to(args.device)

    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=args.device)
        policy.load_state_dict(state, strict=False)
        logger.info("Checkpoint loaded successfully")

    # ── Train / Eval ──────────────────────────────────────────────────────
    if args.mode == "eval":
        _run_eval(args, policy, env, dev_data, encoder_backend)
        return

    # Train
    evaluator      = Evaluator(recall_k_list=args.recall_k)
    hotpot_eval    = HotpotEvaluator()
    trainer = PPOTrainer(
        policy=policy,
        env=env,
        lr=args.lr,
        ppo_epochs=args.ppo_epochs,
        clip_epsilon=args.clip_epsilon,
        kl_coef=args.kl_coef if args.use_kl else 0.0,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        batch_size=args.batch_size,
        rollout_batch_size=args.rollout_batch_size,
        encode_batch_size=args.encode_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        output_dir=args.output_dir,
        device=args.device,
        eval_every=args.eval_every,
        save_every=args.save_every,
        ref_update_every=args.ref_update_every,
    )
    trainer.train(
        train_data=train_data,
        dev_data=dev_data,
        max_episodes=args.max_episodes,
        evaluator=evaluator,
        hotpot_evaluator=hotpot_eval,
    )


if __name__ == "__main__":
    main()