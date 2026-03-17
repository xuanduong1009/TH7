from typing import Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_NAME,
    MAX_DOC_CHARS,
    MAX_INPUT_LENGTH,
)
from utils import normalize_whitespace


class LLMQLMScorer:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_doc_chars: int = MAX_DOC_CHARS,
        max_input_length: int = MAX_INPUT_LENGTH,
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.max_doc_chars = max_doc_chars
        self.max_input_length = max_input_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}
        if self.device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

        self.parameter_count = sum(param.numel() for param in self.model.parameters())

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device in (None, "", "auto"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return device

    def describe(self):
        return {
            "model_name": self.model_name,
            "parameters": int(self.parameter_count),
            "framework": "transformers",
            "device": self.device,
            "max_doc_chars": self.max_doc_chars,
            "max_input_length": self.max_input_length,
            "batch_size": self.batch_size,
        }

    def build_prompt(self, document_text: str) -> str:
        clean_document = normalize_whitespace(document_text)[: self.max_doc_chars]
        return f"Document: {clean_document}\nQuery:"

    @torch.no_grad()
    def _score_batch(self, query: str, documents: Sequence[str]) -> list[float]:
        prompts = [self.build_prompt(document_text) for document_text in documents]
        full_texts = [f"{prompt} {query}".strip() for prompt in prompts]

        enc_full = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        )
        enc_prompt = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        )

        input_ids = enc_full["input_ids"].to(self.device)
        attention_mask = enc_full["attention_mask"].to(self.device)
        prompt_attention = enc_prompt["attention_mask"].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        label_mask = attention_mask[:, 1:].bool()
        prompt_lengths = prompt_attention.sum(dim=1)
        positions = torch.arange(token_log_probs.shape[1], device=self.device).unsqueeze(0)
        query_mask = label_mask & (positions >= (prompt_lengths - 1).unsqueeze(1))

        scores = token_log_probs.masked_fill(~query_mask, 0.0).sum(dim=1)
        has_query_tokens = query_mask.any(dim=1)
        fallback = torch.full_like(scores, -1e9)
        scores = torch.where(has_query_tokens, scores, fallback)

        return [float(score) for score in scores.cpu()]

    def score_documents(
        self,
        query: str,
        documents: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> list[float]:
        effective_batch_size = batch_size or self.batch_size
        all_scores = []
        for start in range(0, len(documents), effective_batch_size):
            batch = documents[start : start + effective_batch_size]
            all_scores.extend(self._score_batch(query, batch))
        return all_scores

    def score(self, query: str, document_text: str) -> float:
        return self.score_documents(query=query, documents=[document_text])[0]
