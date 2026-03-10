# AGENTS.md

This file contains guidelines for agentic coding assistants working in this repository.

## Build/Lint/Test Commands

**No automated tests or linting configured.** This is a research project without standard CI/CD.

```bash
# Install dependencies
pip install -r requirements.txt

# ── Training ─────────────────────────────────────────────────────────────────
# Run main training (context oracle mode recommended)
python main.py --mode train \
    --data_dir data/hotpotqa \
    --index_dir data/index \
    --top_k 10 \
    --max_hops 3

# ── Evaluation ──────────────────────────────────────────────────────────────
# Run evaluation with baselines comparison
python main.py --mode eval \
    --data_dir data/hotpotqa \
    --index_dir data/index \
    --checkpoint outputs/rl_model/policy_final.pt \
    --encoder_model BAAI/bge-base-en-v1.5 \
    --device cuda \
    --retrieval_mode context

# Run only Dense Greedy baseline (no RL policy checkpoint needed)
python main.py --mode eval \
    --data_dir data/hotpotqa \
    --index_dir data/index

# Quick evaluation with sampling (for debugging)
python main.py --mode eval \
    --max_samples 100 \
    --checkpoint outputs/rl_model/policy_final.pt

# ── Ablation Study ──────────────────────────────────────────────────────────
python main.py --mode ablation \
    --data_dir data/hotpotqa \
    --index_dir data/index

# ── Data Analysis ────────────────────────────────────────────────────────────
python test.py  # Analyze gold distribution in HotpotQA
```

**Note:** No pytest, unittest, or linting (ruff/flake8/black) is configured. Use `--max_samples N` for quick testing.

## Code Style Guidelines

### Imports
- Order: standard library, third-party, local (separated by blank lines)
- Group imports alphabetically within each section
```python
import json
import os
from pathlib import Path

import numpy as np
import torch

from environment import MultiHopRetrievalEnv
```
- Import heavy dependencies (transformers, faiss) inside functions when needed

### Type Hints & Naming
- Use `typing` module for all function signatures: `from typing import Dict, List, Optional, Tuple`
- Classes: PascalCase (`PolicyNetwork`, `MultiHopRetrievalEnv`)
- Functions/variables: snake_case (`compute_gae`, `encode_state`)
- Constants: UPPER_CASE at module level (`STOP_ACTION = -1`)
- Private methods: single underscore prefix (`_load_data`, `_init_weights`)

### Docstrings
- Module-level docstring with purpose
- Function docstrings with Args/Returns format:
```python
"""
Multi-hop Retrieval MDP Environment

State: (question, list of selected documents so far)
Action: select one of top-K candidates, or STOP
"""

def compute_gae(...) -> Tuple[List[float], List[float]]:
    """Compute Generalized Advantage Estimation. Returns (advantages, returns)."""
```

### Logging & Error Handling
- Use `logging` module, not `print()`. Initialize: `logger = logging.getLogger(__name__)`
- Main.py configures basic logging format globally
- Try/except for external dependencies and file I/O
- Log warnings/errors, provide graceful fallbacks
- Raise RuntimeError with descriptive messages for invalid states

### PyTorch Patterns
- Move tensors/devices with `.to(device)`
- Use `torch.no_grad()` context for inference
- Detach and move to CPU before storing in buffers
- Initialize weights with `xavier_uniform_` and zero biases
- Use `nn.Sequential` for simple feedforward stacks

### File Structure & Formatting
- Each module has single responsibility (data_utils.py, retriever.py, environment.py)
- Main.py orchestrates imports and argument parsing
- Constants at module level or in dataclasses
- Use `pathlib.Path` for file paths
- Use `dataclass` for state/structured data (`RetrievalState`)
- 4-space indentation, line length ~88-100 characters (black-compatible)
- Blank lines between functions and logical sections, one statement per line

### Configuration & Argument Parsing
- Use `argparse` for CLI arguments (see main.py:parse_args)
- Define config dicts for subsystems (reward_config, retrieval params)
- Store hyperparameters at class/module level for easy reference
- Use `default="cuda" if torch.cuda.is_available() else "cpu"` for device detection

## Key Module Overview

| Module | Purpose |
|--------|---------|
| `main.py` | Entry point, argument parsing, train/eval orchestration |
| `policy.py` | PolicyNetwork with cross-attention scorer |
| `environment.py` | MultiHopRetrievalEnv MDP, state management, rewards |
| `retriever.py` | BM25, Dense (FAISS), Hybrid retrievers |
| `reader.py` | Extractive QA reader (RoBERTa/DPR) |
| `evaluator.py` | Evaluation metrics (EM, F1, recall@k), beam search |
| `ppo_trainer.py` | PPO training loop with GAE |
| `data_utils.py` | HotpotQA dataset loading |

## Evaluation Modes

### Dense Greedy Baseline
- Always runs when `--mode eval`
- Uses encoder (BGE/MiniLM) for retrieval
- Batch processing enabled via `--encode_batch_size`

### RL Policy (Beam Search)
- Runs when `--checkpoint` is provided
- Uses trained policy for multi-hop retrieval
- Beam search decoding (`--beam_size`, default=3)

### Retrieval Modes
- `--retrieval_mode context`: Oracle distractor pool (recommended)
- `--retrieval_mode bm25`: BM25 sparse retrieval
- `--retrieval_mode dense`: Dense DPR/FAISS retrieval
- `--retrieval_mode hybrid`: BM25 + Dense fusion

## Training Patterns
- Set random seeds before training: `set_seed()` in main.py
- Wrap data loading in try/except for external dependencies
- Build indices/caches once, reuse across training/eval
- Save checkpoints with descriptive names: `policy_ep{episode}.pt`
- Track metrics in `defaultdict(list)` for history
- Log key metrics periodically (every N episodes/steps)

## Evaluation Patterns
- Switch to eval mode: `policy.eval()` before inference
- Use `torch.no_grad()` context manager for all inference
- Return to train mode after eval: `policy.train()`
- Support `max_samples` parameter for quick debugging
- Return dict of aggregate metrics (mean across dataset)

## GPU Optimization
- Batch encode states/docs to maximize GPU utilization
- Use `torch.no_grad()` for all inference during rollout
- Implement gradient accumulation for larger effective batch sizes
- `--encode_batch_size` controls batch size for encoder (default: 512)

## Warnings & Filtering
- Filter transformer overflow warnings:
```python
warnings.filterwarnings("ignore", message=".*overflowing tokens are not returned.*", category=UserWarning)
```

## Documentation Style
- Module docstrings at top explaining purpose and modes
- Comments in some modules (environment.py, policy.py, ppo_trainer.py, retriever.py, evaluator.py) are in Chinese - preserve this pattern when editing those files
- Use section dividers (─) to logically separate major code blocks
