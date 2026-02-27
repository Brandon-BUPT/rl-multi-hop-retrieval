"""
Extractive Reader: Given a question and context documents, predict the answer span.
Uses a standard QA model (e.g. RoBERTa fine-tuned on SQuAD2).
"""

import logging
from typing import Dict, List

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)


class ExtractiveReader:

    def __init__(
        self,
        model_name: str = "deepset/roberta-base-squad2",
        device: str = "cpu",
        max_length: int = 512,
        stride: int = 128,
        top_k_answers: int = 1,
    ):
        device_id = 0 if device == "cuda" else -1
        logger.info(f"Loading reader: {model_name}")
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=device_id,
        )
        self.max_length = max_length
        self.stride = stride
        self.top_k_answers = top_k_answers

    def predict(self, question: str, context_docs: List[Dict]) -> str:
        """
        Predict answer given question and list of {title, text} context docs.
        Concatenates all docs and runs QA pipeline.
        Returns the best answer string.
        """
        if not context_docs:
            return ""

        # Concatenate all context docs
        context = " ".join(
            f"{d['title']}: {d['text']}" for d in context_docs
        )

        try:
            result = self.qa_pipeline(
                question=question,
                context=context,
                max_answer_len=50,
                handle_impossible_answer=True,
            )
            return result.get("answer", "")
        except Exception as e:
            logger.warning(f"Reader error: {e}")
            return ""

    def predict_batch(
        self,
        questions: List[str],
        context_docs_list: List[List[Dict]]
    ) -> List[str]:
        """Batch prediction."""
        results = []
        for q, docs in zip(questions, context_docs_list):
            results.append(self.predict(q, docs))
        return results
