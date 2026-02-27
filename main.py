"""
RL Multi-hop Retrieval — Main Entry Point
"""
import argparse, json, logging, os, random, warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*overflowing tokens.*", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       choices=["train","eval"], default="train")
    p.add_argument("--data_dir",   default="data/hotpotqa")
    p.add_argument("--index_dir",  default="data/index")
    p.add_argument("--cache_dir",  default="cache")
    p.add_argument("--encoder_model", default="facebook/dpr-question_encoder-single-nq-base")
    p.add_argument("--reader_model",  default="deepset/roberta-base-squad2")
    p.add_argument("--freeze_layers", type=int, default=6)
    p.add_argument("--policy_hidden", type=int, default=768)
    p.add_argument("--retrieval_mode", default="context", choices=["context","bm25","dpr"])
    p.add_argument("--top_k",    type=int, default=10)
    p.add_argument("--max_hops", type=int, default=2)
    p.add_argument("--index_batch_size", type=int, default=256)
    p.add_argument("--ppo_epochs",     type=int,   default=4)
    p.add_argument("--clip_epsilon",   type=float, default=0.2)
    p.add_argument("--kl_coef",        type=float, default=0.1)
    p.add_argument("--use_kl",         action="store_true", default=True)
    p.add_argument("--gamma",          type=float, default=0.99)
    p.add_argument("--gae_lambda",     type=float, default=0.95)
    p.add_argument("--lr",             type=float, default=1e-5)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--rollout_batch_size", type=int, default=64)
    p.add_argument("--encode_batch_size",  type=int, default=64)
    p.add_argument("--grad_accum_steps",   type=int, default=4)
    p.add_argument("--max_episodes",       type=int, default=100000)
    p.add_argument("--eval_every",         type=int, default=512)
    p.add_argument("--save_every",         type=int, default=1024)
    p.add_argument("--reward_em_weight",          type=float, default=0.3)
    p.add_argument("--reward_f1_weight",          type=float, default=0.2)
    p.add_argument("--reward_sf_weight",          type=float, default=1.0)
    p.add_argument("--reward_joint_bonus",        type=float, default=0.5)
    p.add_argument("--reward_step_weight",        type=float, default=0.2)
    p.add_argument("--reward_early_stop_penalty", type=float, default=0.0)  # ⑥ 已无效，保留兼容
    p.add_argument("--reward_min_hops",           type=int,   default=2)    # ⑥ 已无效，保留兼容
    p.add_argument("--use_step_reward", action="store_true", default=True)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--recall_k",   type=int, nargs="+", default=[1,2,5,10])
    p.add_argument("--ref_update_every", type=int, default=200,
                   help="每隔多少 global_step 更新一次 KL 参考策略（0=不更新）")  # ③
    p.add_argument("--beam_size",  type=int, default=3,
                   help="评测时 beam search 的 beam 数（1=greedy）")              # ⑧
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    logger.info(f"Mode: {args.mode} | Device: {args.device}")

    from data_utils import HotpotQADataset
    from retriever import DenseRetriever, BM25Retriever
    from environment import MultiHopRetrievalEnv
    from policy import PolicyNetwork
    from ppo_trainer import PPOTrainer
    from reader import ExtractiveReader
    from evaluator import Evaluator
    from hotpot_eval import HotpotEvaluator

    train_data = HotpotQADataset(args.data_dir, split="train",          cache_dir=args.cache_dir)
    dev_data   = HotpotQADataset(args.data_dir, split="dev_distractor", cache_dir=args.cache_dir)
    logger.info(f"Train: {len(train_data)}, Dev: {len(dev_data)}")

    bm25_retriever  = BM25Retriever(index_dir=args.index_dir)
    dense_retriever = DenseRetriever(model_name=args.encoder_model, index_dir=args.index_dir,
                                      device=args.device, cache_dir=args.cache_dir,
                                      batch_size=args.index_batch_size)

    if args.retrieval_mode in ("bm25", "dpr"):
        corpus = {**train_data.corpus, **dev_data.corpus}
        logger.info(f"Corpus: {len(corpus)} passages")
        bm25_retriever.build_index(corpus)
        if args.retrieval_mode == "dpr":
            dense_retriever.build_index(corpus)
    else:
        logger.info("Context oracle mode — skipping index build")

    retriever = (bm25_retriever if args.retrieval_mode == "bm25"
                 else dense_retriever if args.retrieval_mode == "dpr"
                 else bm25_retriever)

    reader = ExtractiveReader(model_name=args.reader_model, device=args.device)

    env = MultiHopRetrievalEnv(
        retriever=retriever, reader=reader,
        top_k=args.top_k, max_hops=args.max_hops,
        retrieval_mode=args.retrieval_mode,
        reward_config={
            "sf_weight":            args.reward_sf_weight,
            "joint_bonus":          args.reward_joint_bonus,
            "em_weight":            args.reward_em_weight,
            "f1_weight":            args.reward_f1_weight,
            "step_weight":          args.reward_step_weight,
            "use_step_reward":      args.use_step_reward,
            # 兼容字段（⑥ 已无效）
            "early_stop_penalty":   args.reward_early_stop_penalty,
            "min_hops_before_stop": args.reward_min_hops,
        }
    )

    policy = PolicyNetwork(
        encoder_model=args.encoder_model,
        hidden_dim=args.policy_hidden,
        freeze_layers=args.freeze_layers,
        device=args.device,
    ).to(args.device)

    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        missing, unexpected = policy.load_state_dict(ckpt, strict=False)
        if [k for k in missing]: logger.info(f"Missing keys: {missing}")
        if unexpected: logger.warning(f"Unexpected keys: {unexpected}")
        logger.info("Checkpoint loaded.")

    evaluator = Evaluator(recall_k_list=args.recall_k)

    if args.mode == "train":
        trainer = PPOTrainer(
            policy=policy, env=env,
            lr=args.lr, ppo_epochs=args.ppo_epochs,
            clip_epsilon=args.clip_epsilon,
            kl_coef=args.kl_coef if args.use_kl else 0.0,
            gamma=args.gamma, gae_lambda=args.gae_lambda,
            batch_size=args.batch_size,
            rollout_batch_size=args.rollout_batch_size,
            encode_batch_size=args.encode_batch_size,
            grad_accum_steps=args.grad_accum_steps,
            output_dir=args.output_dir, device=args.device,
            eval_every=args.eval_every, save_every=args.save_every,
            ref_update_every=args.ref_update_every,  # ③
        )
        trainer.train(
            train_data=train_data, dev_data=dev_data,
            max_episodes=args.max_episodes,
            evaluator=evaluator,
            hotpot_evaluator=HotpotEvaluator(),
        )

    elif args.mode == "eval":
        from eval_greedy_baseline import evaluate_bm25_greedy, print_table
        hotpot_evaluator = HotpotEvaluator()
        logger.info("="*60)
        logger.info("Starting evaluation on dev set")
        logger.info("="*60)

        logger.info("\n[1/3] Evaluating RL policy (greedy)...")
        rl_m = hotpot_evaluator.evaluate(policy=policy, env=env, dataset=dev_data, mode="rl")

        logger.info("\n[2/3] Evaluating RL policy (beam search)...")
        beam_m = evaluator.evaluate(policy, env, dev_data, mode="beam",
                                    beam_size=args.beam_size, max_samples=500)
        logger.info(f"RL beam (size={args.beam_size}): {json.dumps(beam_m, indent=2)}")

        logger.info("\n[3/3] Evaluating BM25 greedy baseline...")
        bm_m = evaluate_bm25_greedy(dev_data.data, reader, max_hops=args.max_hops)
        print_table(bm_m, "BM25 Greedy (Context Oracle)")

        logger.info("\n" + "="*60)
        logger.info("Evaluation complete. Saving results...")
        logger.info("="*60)
        os.makedirs(args.output_dir, exist_ok=True)
        results = {"rl_greedy": rl_m, "rl_beam": beam_m, "bm25": bm_m}
        json.dump(results,
                  open(os.path.join(args.output_dir, "eval_results.json"), "w"), indent=2)
        logger.info(f"Results saved to {args.output_dir}/eval_results.json")


if __name__ == "__main__":
    main()