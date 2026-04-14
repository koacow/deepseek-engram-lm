from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ExperimentConfig:
    vocab_size: int
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    d_ff: int = 3072
    dropout: float = 0.1
    engram_layers: Tuple[int, ...] = (2, 6)
    engram_orders: Tuple[int, ...] = (2, 3)
    engram_heads_per_order: int = 8
    engram_head_dim: int = 256
    engram_gate_frozen: bool = False
    engram_table_target_size: int = 5_000_000
    control_adapter_width: int = 1536


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.scale


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.proj(attn))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ControlAdapter(nn.Module):
    def __init__(self, d_model: int, width: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up = nn.Linear(d_model, width)
        self.down = nn.Linear(width, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(F.gelu(self.up(self.norm(x))))
        return self.dropout(y)


def _next_prime(start: int) -> int:
    candidate = max(2, start)
    while True:
        if candidate < 4:
            is_prime = candidate in (2, 3)
        else:
            is_prime = candidate % 2 and all(candidate % i for i in range(3, int(candidate**0.5) + 1, 2))
        if is_prime:
            return candidate
        candidate += 1


class NGramHasher:
    def __init__(
        self,
        vocab_size: int,
        layer_ids: Sequence[int],
        orders: Sequence[int],
        heads_per_order: int,
        seed: int = 0,
        pad_id: int = 0,
        table_target_size: int = 5_000_000,
    ):
        self.vocab_size = int(vocab_size)
        self.layer_ids = tuple(int(x) for x in layer_ids)
        self.orders = tuple(int(x) for x in orders)
        self.heads_per_order = int(heads_per_order)
        self.seed = int(seed)
        self.pad_id = int(pad_id)

        self.base_table_size = max(257, table_target_size // max(1, len(self.layer_ids) * len(self.orders) * self.heads_per_order))
        self.table_sizes = self._build_prime_tables()
        g = torch.Generator().manual_seed(self.seed)
        self.multipliers = {}
        for layer_id in self.layer_ids:
            vals = torch.randint(1, 2**31 - 1, (max(self.orders),), generator=g, dtype=torch.long)
            self.multipliers[layer_id] = (vals * 2 + 1).tolist()

    def _build_prime_tables(self) -> dict[int, dict[int, List[int]]]:
        table_sizes: dict[int, dict[int, List[int]]] = {}
        used = set()
        start = self.base_table_size
        for layer_id in self.layer_ids:
            table_sizes[layer_id] = {}
            for order in self.orders:
                sizes: List[int] = []
                current = start
                for _ in range(self.heads_per_order):
                    p = _next_prime(current)
                    while p in used:
                        p = _next_prime(p + 1)
                    used.add(p)
                    sizes.append(p)
                    current = p + 1
                table_sizes[layer_id][order] = sizes
        return table_sizes

    def hash(self, input_ids: torch.Tensor, layer_id: int) -> torch.Tensor:
        x = input_ids.long()
        b, t = x.shape
        device = x.device
        m = self.multipliers[layer_id]

        all_hashes: List[torch.Tensor] = []
        for order in self.orders:
            shifted = []
            for k in range(order):
                if k == 0:
                    shifted.append(x)
                else:
                    pad = torch.full((b, k), self.pad_id, device=device, dtype=torch.long)
                    shifted.append(torch.cat([pad, x[:, :-k]], dim=1))
            mix = shifted[0] * m[0]
            for k in range(1, order):
                mix = torch.bitwise_xor(mix, shifted[k] * m[k])
            for j, mod in enumerate(self.table_sizes[layer_id][order]):
                all_hashes.append((mix % mod).unsqueeze(-1))

        return torch.cat(all_hashes, dim=-1)


class EngramAdapter(nn.Module):
    def __init__(self, cfg: ExperimentConfig, layer_id: int, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.hasher = NGramHasher(
            vocab_size=cfg.vocab_size,
            layer_ids=cfg.engram_layers,
            orders=cfg.engram_orders,
            heads_per_order=cfg.engram_heads_per_order,
            seed=0,
            pad_id=pad_id,
            table_target_size=cfg.engram_table_target_size,
        )

        self.tables = nn.ModuleDict()
        for order in cfg.engram_orders:
            for head_idx, table_size in enumerate(self.hasher.table_sizes[layer_id][order]):
                key = f"o{order}_h{head_idx}"
                self.tables[key] = nn.Embedding(table_size, cfg.engram_head_dim)

        mem_dim = len(cfg.engram_orders) * cfg.engram_heads_per_order * cfg.engram_head_dim
        self.value_proj = nn.Linear(mem_dim, cfg.d_model)
        self.key_proj = nn.Linear(mem_dim, cfg.d_model)
        self.query_norm = RMSNorm(cfg.d_model)
        self.key_norm = RMSNorm(cfg.d_model)
        # Do not pad conv and manually pad left in forward
        self.short_conv = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=4,
                            groups=cfg.d_model, padding=0)
        self.short_conv_norm = RMSNorm(cfg.d_model)
        nn.init.zeros_(self.short_conv.bias)
        self.gate_frozen = cfg.engram_gate_frozen

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        hashed = self.hasher.hash(input_ids, self.layer_id)  # [B,T,orders*heads]
        pieces = []
        idx = 0
        for order in self.cfg.engram_orders:
            for head_idx in range(self.cfg.engram_heads_per_order):
                key = f"o{order}_h{head_idx}"
                pieces.append(self.tables[key](hashed[:, :, idx]))
                idx += 1
        embeddings = torch.cat(pieces, dim=-1)

        value = self.value_proj(embeddings)
        key = self.key_norm(self.key_proj(embeddings))
        query = self.query_norm(hidden_states)

        if self.gate_frozen:
            gate = torch.full_like(value[..., :1], 0.5)
        else:
            score = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(hidden_states.size(-1))
            gate = torch.sigmoid(score)

        delta = gate * value
        # Normalize delta, pad left for causal conv, apply conv, and add back to delta
        conv_in = F.pad(self.short_conv_norm(delta).transpose(1, 2), (3, 0))
        conv_out = F.silu(self.short_conv(conv_in)).transpose(1, 2)
        return delta + conv_out


def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss with teacher-forcing causal shift.

    logits[t] predicts position t+1, so we align logits[:, :-1] with
    labels[:, 1:]. This is the standard next-token-prediction objective.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


class _BaseLM(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        b, t = input_ids.shape
        if t > self.cfg.block_size:
            raise ValueError(f"Sequence length {t} exceeds block size {self.cfg.block_size}")
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = _causal_lm_loss(logits, labels) if labels is not None else None
        return logits, loss


class BaselineLM(_BaseLM):
    pass


class ParamsControlLM(_BaseLM):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__(cfg)
        self.control_layers = nn.ModuleDict({
            str(layer_id): ControlAdapter(cfg.d_model, cfg.control_adapter_width, cfg.dropout)
            for layer_id in cfg.engram_layers
        })

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        b, t = input_ids.shape
        if t > self.cfg.block_size:
            raise ValueError(f"Sequence length {t} exceeds block size {self.cfg.block_size}")
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.cfg.engram_layers:
                x = x + self.control_layers[str(idx)](x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = _causal_lm_loss(logits, labels) if labels is not None else None
        return logits, loss


class EngramLM(_BaseLM):
    def __init__(self, cfg: ExperimentConfig, pad_id: int = 0):
        super().__init__(cfg)
        self.adapters = nn.ModuleDict({
            str(layer_id): EngramAdapter(cfg, layer_id=layer_id, pad_id=pad_id)
            for layer_id in cfg.engram_layers
        })

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        b, t = input_ids.shape
        if t > self.cfg.block_size:
            raise ValueError(f"Sequence length {t} exceeds block size {self.cfg.block_size}")
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for idx, block in enumerate(self.blocks):
            x = block(x)
            # Adapter inserted POST-block so the gating query sees the full
            # block output (attn + FFN), matching the ControlAdapter insertion.
            if idx in self.cfg.engram_layers:
                x = x + self.adapters[str(idx)](x, input_ids)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = _causal_lm_loss(logits, labels) if labels is not None else None
        return logits, loss
