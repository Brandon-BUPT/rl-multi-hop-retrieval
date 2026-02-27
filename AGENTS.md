# AGENTS.md

This file contains guidelines for agentic coding assistants working in this repository.

## Build/Lint/Test Commands

**No automated tests or linting configured.** This is a research project without standard CI/CD.

```bash
# Install dependencies
pip install -r requirements.txt

# Run main training (context oracle mode recommended)
python main.py --mode train --data_dir data/hotpotqa --index_dir data/index --top_k 10 --max_hops 3

# Run evaluation with baselines comparison
python main.py --mode eval --checkpoint outputs/rl_model/policy_final.pt

# Run ablation study
python main.py --mode ablation --data_dir data/hotpotqa --index_dir data/index
```

**Note:** No pytest, unittest, or linting (ruff/flake8/black) is configured. When adding functionality, manually test by running the relevant mode.

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
- Import inside functions for heavy dependencies (transformers, faiss) when needed

### Type Hints
- Use `typing` module for all function signatures
```python
from typing import Dict, List, Optional, Tuple

def encode_state(self, question: str, selected_docs: List[Dict]) -> torch.Tensor:
```

### Naming Conventions
- Classes: PascalCase (`PolicyNetwork`, `MultiHopRetrievalEnv`)
- Functions/variables: snake_case (`compute_gae`, `encode_state`)
- Constants: UPPER_CASE at module level (`STOP_ACTION = -1`)
- Private methods: single underscore prefix (`_load_data`, `_init_weights`)

### Docstrings
- Module-level docstring with purpose
- Function docstrings with Args/Returns format
```python
"""
Multi-hop Retrieval MDP Environment

State: (question, list of selected documents so far)
Action: select one of top-K candidates, or STOP
"""

def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """Compute Generalized Advantage Estimation. Returns (advantages, returns)."""
```

### Logging
- Use `logging` module, not `print()`
- Initialize logger at module level: `logger = logging.getLogger(__name__)`
- Main.py configures basic logging format globally

### Error Handling
- Try/except for external dependencies and file I/O
- Log warnings/errors, provide graceful fallbacks
- Raise RuntimeError with descriptive messages for invalid states

### PyTorch Patterns
- Move tensors/devices with `.to(device)`
- Use `torch.no_grad()` context for inference
- Detach and move to CPU before storing in buffers
- Initialize weights with `xavier_uniform_` and zero biases
- Use `nn.Sequential` for simple feedforward stacks

### File Structure
- Each module has single responsibility (data_utils.py, retriever.py, environment.py)
- Main.py orchestrates imports and argument parsing
- Constants defined at module level or in dataclasses
- Use `pathlib.Path` for file paths

### Data Structures
- Use `dataclass` for state/structured data (`RetrievalState`)
- Dict returns for metrics/results
- Lists for sequential data
- Tuples for fixed-size returns

### Formatting
- 4-space indentation
- Line length ~88-100 characters (compatible with black)
- Blank lines between functions and logical sections
- One statement per line

### Configuration & Argument Parsing
- Use `argparse` for CLI arguments (see main.py:parse_args)
- Define config dicts for subsystems (reward_config, retrieval params)
- Store hyperparameters at class/module level for easy reference
- Use `default="cuda" if torch.cuda.is_available() else "cpu"` for device detection

### Training Patterns
- Set random seeds before training: `set_seed()` in main.py
- Wrap data loading in try/except for external dependencies
- Build indices/caches once, reuse across training/eval
- Save checkpoints with descriptive names: `policy_ep{episode}.pt`
- Track metrics in `defaultdict(list)` for history
- Log key metrics periodically (every N episodes/steps)

### Evaluation Patterns
- Switch to eval mode: `policy.eval()` before inference
- Use `torch.no_grad()` context manager for all inference
- Return to train mode after eval: `policy.train()`
- Support `max_samples` parameter for quick debugging
- Return dict of aggregate metrics (mean across dataset)

### State Management
- Use `@dataclass` for structured state (RetrievalState)
- Store metadata (ids, titles) alongside tensors
- Use field(default_factory=list) for mutable defaults
- Include done flag for episode termination

### Checkpointing
- Use `torch.save(model.state_dict(), path)` for saving
- Use `model.load_state_dict(torch.load(path, map_location=device))` for loading
- Check `checkpoint` argument to conditionally load models
- Save both intermediate and final checkpoints

### Warnings & Filtering
- Filter transformer overflow warnings to reduce noise:
```python
warnings.filterwarnings(
    "ignore",
    message=".*overflowing tokens are not returned.*",
    category=UserWarning,
)
```

### GPU Optimization Patterns
- Batch encode states/docs to maximize GPU utilization (see ppo_trainer.py:172)
- Use `torch.no_grad()` context for all inference during rollout
- Implement gradient accumulation for larger effective batch sizes
- Pad variable-length sequences within batches for parallel processing
- Store tensors on CPU in buffers after detaching gradients

### Indexing & Caching
- Build FAISS/DPR/BM25 indices once, cache to disk with pickle
- Use `.pkl` or `.faiss` extensions for cached indices
- Check cache existence before building (index_dir/*.pkl, index_dir/*.faiss)
- Include metadata (titles, texts) with cached indices for reconstruction

### Documentation Style
- Module docstrings at top explaining purpose and modes
- Comments in some modules (environment.py, policy.py, ppo_trainer.py, retriever.py, evaluator.py) are in Chinese - preserve this pattern when editing those files
- Use section dividers (─) to logically separate major code blocks
