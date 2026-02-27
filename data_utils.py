"""
HotpotQA dataset utilities
Download: https://hotpotqa.github.io/
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from torch.utils.data import Dataset

HOTPOTQA_URLS = {
    "train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
    "dev_distractor": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    "dev_fullwiki": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
}


class HotpotQADataset(Dataset):
    """
    Each item:
    {
        "_id": str,
        "question": str,
        "answer": str,
        "supporting_facts": [(title, sent_idx), ...],
        "context": [(title, [sent1, sent2, ...]), ...],
        "type": "bridge" | "comparison",
        "level": "easy" | "medium" | "hard"
    }
    """

    def __init__(self, data_dir: str, split: str = "train", cache_dir: str = "cache"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.data = self._load_data()
        # Build a corpus of all passages for retrieval
        self.corpus = self._build_corpus()

    def _load_data(self) -> List[Dict]:
        cache_path = self.cache_dir / f"hotpotqa_{self.split}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        filepath = self.data_dir / f"hotpot_{self.split}_v1.1.json"
        if not filepath.exists():
            filepath = self.data_dir / f"hotpot_{self.split}.json"
        if not filepath.exists():
            self._download(self.split)
            filepath = self.data_dir / f"hotpot_{self.split}.json"

        with open(filepath, "r") as f:
            raw = json.load(f)

        data = []
        for item in raw:
            data.append({
                "_id": item["_id"],
                "question": item["question"],
                "answer": item["answer"],
                "supporting_facts": item.get("supporting_facts", []),
                "context": item.get("context", []),
                "type": item.get("type", "bridge"),
                "level": item.get("level", "medium")
            })

        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

        return data

    def _download(self, split: str):
        url = HOTPOTQA_URLS[split]
        self.data_dir.mkdir(parents=True, exist_ok=True)
        dest = self.data_dir / f"hotpot_{split}.json"
        print(f"Downloading {url} -> {dest}")
        resp = requests.get(url, stream=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    def _build_corpus(self) -> Dict[str, str]:
        """title -> full text (all sentences joined)"""
        corpus = {}
        for item in self.data:
            for title, sents in item["context"]:
                if title not in corpus:
                    corpus[title] = " ".join(sents)
        return corpus

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]

    def get_supporting_titles(self, item: Dict) -> List[str]:
        return list({sf[0] for sf in item["supporting_facts"]})

    def get_all_titles(self, item: Dict) -> List[str]:
        return [title for title, _ in item["context"]]


def collate_fn(batch: List[Dict]) -> List[Dict]:
    return batch
