from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass
class TokenizerBundle:
    tokenizer: object
    pad_id: int
    vocab_size: int


def build_tokenizer(model_name: str = "gpt2") -> TokenizerBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return TokenizerBundle(tokenizer=tokenizer, pad_id=int(tokenizer.pad_token_id), vocab_size=int(tokenizer.vocab_size))


def load_wikitext103(tokenizer_name: str = "gpt2", block_size: int = 512):
    tokenizer_bundle = build_tokenizer(tokenizer_name)
    tokenizer = tokenizer_bundle.tokenizer
    raw = load_dataset("wikitext", "wikitext-103-raw-v1")

    def tokenize(batch):
        return tokenizer(batch["text"])

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_ds = tokenized.map(group_texts, batched=True)
    lm_ds.set_format(type="torch", columns=["input_ids", "labels"])
    return lm_ds, tokenizer_bundle


def collate_lm_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {"input_ids": input_ids, "labels": labels}
