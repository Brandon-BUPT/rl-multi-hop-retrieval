"""
Ablation Study Runner

Ablation variants:
1. no_kl         : PPO without KL divergence constraint
2. no_step_reward: PPO without intermediate (step) reward
3. sparse_reward : only EM at end (maximally sparse)
4. dense_reward  : step reward at every hop + final reward (maximally dense)

Runs each variant and saves comparison results.
"""

import copy
import json
import logging
import os
from pathlib import Path
from typing import Dict

import torch

logger = logging.getLogger(__name__)


ABLATION_CONFIGS = {
    "full_model": {
        "use_kl": True,
        "kl_coef": 0.1,
        "use_step_reward": True,
        "reward_config": {
            "em_weight": 1.0, "f1_weight": 0.5,
            "sf_weight": 0.3, "step_weight": 0.1,
            "use_step_reward": True,
        }
    },
    "no_kl": {
        "use_kl": False,
        "kl_coef": 0.0,
        "use_step_reward": True,
        "reward_config": {
            "em_weight": 1.0, "f1_weight": 0.5,
            "sf_weight": 0.3, "step_weight": 0.1,
            "use_step_reward": True,
        }
    },
    "no_step_reward": {
        "use_kl": True,
        "kl_coef": 0.1,
        "use_step_reward": False,
        "reward_config": {
            "em_weight": 1.0, "f1_weight": 0.5,
            "sf_weight": 0.3, "step_weight": 0.0,
            "use_step_reward": False,
        }
    },
    "sparse_reward": {
        "use_kl": True,
        "kl_coef": 0.1,
        "use_step_reward": False,
        "reward_config": {
            "em_weight": 1.0, "f1_weight": 0.0,
            "sf_weight": 0.0, "step_weight": 0.0,
            "use_step_reward": False,
        }
    },
    "dense_reward": {
        "use_kl": True,
        "kl_coef": 0.1,
        "use_step_reward": True,
        "reward_config": {
            "em_weight": 1.0, "f1_weight": 0.5,
            "sf_weight": 0.5, "step_weight": 0.3,
            "use_step_reward": True,
        }
    },
}


def run_ablation(args, base_policy, base_env, retriever, reader, dev_data, evaluator):
    """
    Run all ablation experiments.
    Each ablation trains for a smaller number of episodes (args.max_episodes // 5)
    to keep runtime tractable, then evaluates on dev.
    """
    from environment import MultiHopRetrievalEnv
    from ppo_trainer import PPOTrainer

    results = {}
    ablation_episodes = max(args.max_episodes // 5, 1000)

    for ablation_name, config in ABLATION_CONFIGS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running ablation: {ablation_name}")
        logger.info(f"Config: {config}")
        logger.info(f"{'='*60}")

        # Fresh policy
        policy = copy.deepcopy(base_policy)
        policy = policy.to(args.device)

        # Build env with ablation reward config
        env = MultiHopRetrievalEnv(
            retriever=retriever,
            reader=reader,
            top_k=args.top_k,
            max_hops=args.max_hops,
            reward_config=config["reward_config"]
        )

        # PPO Trainer
        ablation_out = os.path.join(args.output_dir, f"ablation_{ablation_name}")
        os.makedirs(ablation_out, exist_ok=True)

        trainer = PPOTrainer(
            policy=policy,
            env=env,
            lr=args.lr,
            ppo_epochs=args.ppo_epochs,
            clip_epsilon=args.clip_epsilon,
            kl_coef=config["kl_coef"],
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            batch_size=args.batch_size,
            output_dir=ablation_out,
            device=args.device,
            eval_every=ablation_episodes,  # eval only at end
        )

        from data_utils import HotpotQADataset
        train_data = HotpotQADataset(args.data_dir, split="train", cache_dir=args.cache_dir)

        trainer.train(
            train_data=train_data,
            dev_data=dev_data,
            max_episodes=ablation_episodes,
            evaluator=None  # no mid-training eval
        )

        # Evaluate
        metrics = evaluator.evaluate(
            policy=policy,
            env=env,
            dataset=dev_data,
            mode="rl",
            max_samples=500
        )
        results[ablation_name] = {**config, "metrics": metrics}
        logger.info(f"[{ablation_name}] Metrics: {metrics}")

    # Save
    out_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nAblation results saved to {out_path}")

    # Print comparison table
    print_ablation_table(results)
    return results


def print_ablation_table(results: Dict):
    metrics_of_interest = ["em", "f1", "sf_f1", "sf_recall", "recall@2"]
    header = f"{'Variant':<25}" + "".join(f"{m:>12}" for m in metrics_of_interest)
    print("\n" + "=" * 80)
    print("ABLATION RESULTS")
    print("=" * 80)
    print(header)
    print("-" * 80)
    for name, data in results.items():
        m = data.get("metrics", {})
        row = f"{name:<25}" + "".join(f"{m.get(k, 0):>12.4f}" for k in metrics_of_interest)
        print(row)
    print("=" * 80)
