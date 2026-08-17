"""
Autoregressive Transformer Decoders for Hierarchical Classification.

This module implements modern transformer decoder architectures designed to ingest
visual backbone embeddings and autoregressively predict hierarchical class tokens.

Architectural References:
    - Transformer/Cross-Attention: "Attention Is All You Need" (Vaswani et al., 2017)
      https://arxiv.org/abs/1706.03762
    - RMSNorm: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
      https://arxiv.org/abs/1910.07467
    - SwiGLU FeedForward: "GLU Variants Improve Transformer" (Shazeer, 2020)
      https://arxiv.org/abs/2002.05202
    - FlashAttention (via PyTorch SDPA): "Fast and Memory-Efficient Exact Attention
      with IO-Awareness" (Dao et al., 2022)
      https://arxiv.org/abs/2205.14135

The primary implementation (`XADecoder`) utilizes a LLaMA/Mistral-style backbone
(RMSNorm + SwiGLU) extended with Cross-Attention to process external memory contexts.

See reference implementation(s):
    - https://github.com/meta-pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/torchtune/modules/transformer.py
    - https://github.com/huggingface/transformers/blob/f0e41a3ef4daf287c694a4731d50eefe9d57d48c/src/transformers/models/mistral/modular_mistral.py#L44
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDecoder(nn.Module, ABC):
    """Abstract API for the autoregressive classification head decoder."""

    @property
    @abstractmethod
    def d_model(self) -> int:
        """Return the expected embedding dimension."""
        pass

    @abstractmethod
    def forward(
        self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor | None = None, tgt_is_causal: bool = True
    ) -> torch.Tensor:
        """
        Standard forward pass for training or stateless generation.
        Expected shapes (assuming batch_first=False for compatibility):
        tgt: (Seq, Batch, Dim)
        memory: (Mem_Seq, Batch, Dim)
        """
        pass

    # Note: A .step() method here for KV-caching could be added in the future.


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (normed * self.weight.to(torch.float32)).to(in_dtype)


class XADecoderLayer(nn.Module):
    """
    A modern decoder layer utilizing RMSNorm, SwiGLU (Gated MLP),
    and native Scaled Dot Product Attention.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model

        self.norm_self_attn = RMSNorm(d_model)
        self.norm_cross_attn = RMSNorm(d_model)
        self.norm_ff = RMSNorm(d_model)

        self.self_attn_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.self_attn_out = nn.Linear(d_model, d_model, bias=False)

        self.cross_attn_q = nn.Linear(d_model, d_model, bias=False)
        self.cross_attn_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.cross_attn_out = nn.Linear(d_model, d_model, bias=False)

        # SwiGLU FeedForward
        hidden_dim = int(8 * d_model / 3)
        self.ff_w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.ff_w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.ff_w3 = nn.Linear(hidden_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_is_causal: bool = True):
        seq_len, bsz, _ = tgt.shape

        # 1. Self Attention (Causal)
        x = self.norm_self_attn(tgt)
        qkv = self.self_attn_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.contiguous().transpose(0, 1).view(bsz, seq_len, self.nhead, -1).transpose(1, 2) for t in qkv]

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=tgt_is_causal, dropout_p=self.dropout.p if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model).transpose(0, 1)
        tgt = tgt + self.dropout(self.self_attn_out(attn_out))

        # 2. Cross Attention (Context from Backbone)
        x = self.norm_cross_attn(tgt)
        mem_seq, _, _ = memory.shape
        q_c = self.cross_attn_q(x).contiguous().transpose(0, 1).view(bsz, seq_len, self.nhead, -1).transpose(1, 2)
        kv_c = self.cross_attn_kv(memory).chunk(2, dim=-1)
        k_c, v_c = [t.contiguous().transpose(0, 1).view(bsz, mem_seq, self.nhead, -1).transpose(1, 2) for t in kv_c]

        cross_out = F.scaled_dot_product_attention(q_c, k_c, v_c, is_causal=False, dropout_p=self.dropout.p if self.training else 0.0)
        cross_out = cross_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model).transpose(0, 1)
        tgt = tgt + self.dropout(self.cross_attn_out(cross_out))

        # 3. SwiGLU FeedForward
        x = self.norm_ff(tgt)
        ff_out = F.silu(self.ff_w1(x)) * self.ff_w2(x)
        tgt = tgt + self.dropout(self.ff_w3(ff_out))

        return tgt


class XADecoder(BaseDecoder):
    """
    Standard sequence-to-sequence decoder stack utilizing gated MLPs and Cross-Attention.
    """

    def __init__(self, d_model: int, num_layers: int = 4, nhead: int = 1, dropout: float = 0.1):
        super().__init__()
        self._d_model = d_model
        self.layers = nn.ModuleList([XADecoderLayer(d_model, nhead, dropout) for _ in range(num_layers)])
        self.final_norm = RMSNorm(d_model)

    @property
    def d_model(self) -> int:
        return self._d_model

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor | None = None, tgt_is_causal: bool = True):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_is_causal=tgt_is_causal)

        return self.final_norm(tgt)


class DecoderLayer(nn.Module):
    """A standard LLM-style self-attention layer (No Cross-Attention)."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model

        self.norm_attn = RMSNorm(d_model)
        self.norm_ff = RMSNorm(d_model)

        self.attn_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)

        hidden_dim = int(8 * d_model / 3)
        self.ff_w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.ff_w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.ff_w3 = nn.Linear(hidden_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        seq_len, bsz, _ = x.shape

        # 1. Self Attention
        normed = self.norm_attn(x)
        qkv = self.attn_qkv(normed).chunk(3, dim=-1)
        q, k, v = [t.contiguous().transpose(0, 1).view(bsz, seq_len, self.nhead, -1).transpose(1, 2) for t in qkv]

        # Note: We pass our custom boolean mask, so we set is_causal=False
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, dropout_p=self.dropout.p if self.training else 0.0
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model).transpose(0, 1)
        x = x + self.dropout(self.attn_out(attn_out))

        # 2. SwiGLU FeedForward
        normed = self.norm_ff(x)
        ff_out = F.silu(self.ff_w1(normed)) * self.ff_w2(normed)
        x = x + self.dropout(self.ff_w3(ff_out))

        return x


# Note: This doesn't work well at all, but I'll leave it here for reference
class PrefixDecoder(BaseDecoder):
    """
    Decoder-Only architecture. Prepends 'memory' to 'tgt', processes them
    together, and slices the memory off before returning.
    """

    def __init__(self, d_model: int, num_layers: int = 2, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self._d_model = d_model
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, dropout) for _ in range(num_layers)])
        self.final_norm = RMSNorm(d_model)

    @property
    def d_model(self):
        return self._d_model

    def _generate_prefix_mask(self, mem_len: int, tgt_len: int, device: torch.device):
        """
        Creates a mask where:
        - Memory can see all Memory (bi-directional)
        - Target can see all Memory
        - Target can see past Target (causal)
        - Target CANNOT see future Target
        - Memory CANNOT see Target
        """
        tot_len = mem_len + tgt_len

        # False means "do not attend" (mask out) in SDPA when using a boolean mask
        mask = torch.zeros(tot_len, tot_len, dtype=torch.bool, device=device)

        # 1. Memory attends to memory
        mask[:mem_len, :mem_len] = True

        # 2. Tgt attends to memory
        mask[mem_len:, :mem_len] = True

        # 3. Tgt attends to tgt (causal - upper triangular)
        tgt_causal = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device), diagonal=1)
        mask[mem_len:, mem_len:] = tgt_causal

        # PyTorch SDPA expects the mask shape to broadcast with (bsz, nhead, seq, seq)
        # So we reshape to (1, 1, seq, seq)
        return mask.view(1, 1, tot_len, tot_len)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor | None = None, tgt_is_causal: bool = True):
        mem_len = memory.size(0)
        tgt_len = tgt.size(0)

        # 1. Concat memory and target sequence
        # Shape: (mem_len + tgt_len, batch_size, d_model)
        full_seq = torch.cat([memory, tgt], dim=0)

        # 2. Create the specialized prefix mask
        attn_mask = self._generate_prefix_mask(mem_len, tgt_len, device=tgt.device)

        # 3. Pass through self-attention layers
        for layer in self.layers:
            full_seq = layer(full_seq, attn_mask=attn_mask)

        full_seq = self.final_norm(full_seq)

        # 4. Slice off the memory to satisfy the API contract
        # The head only wants the sequence corresponding to the hierarchical token decisions
        return full_seq[mem_len:]
