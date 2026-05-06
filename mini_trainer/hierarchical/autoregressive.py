from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import torch
import torch.nn as nn

from mini_trainer.classifier import EmbeddingContext, SupervisionContext
from mini_trainer.utils.imports import import_class

P = ParamSpec("P")
R = TypeVar("R")

def register_generator(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        setattr(func, "__register_name__", name)
        return func

    return decorator

class AutoregressiveMixin(nn.Module, ABC):
    """
    Intermediary abstract mixin unifying transformer-based generation, 
    setup, and prediction logic across different classifier hierarchies.
    """
    sequence_length : int
    preclassification_size : int
    
    # --- Structural Abstractions ---
    @abstractmethod
    def preclassification(self, x: torch.Tensor) -> torch.Tensor: pass
    
    @abstractmethod
    def classify(self, sequence: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]: pass

    # --- Generation Abstractions ---
    @abstractmethod
    def _get_step_logits(self, sequence: torch.Tensor, step: int) -> torch.Tensor:
        """Extract unnormalized logits from the sequence for the current step."""
        pass

    @abstractmethod
    def _get_step_decision(self, sequence: torch.Tensor, logits: torch.Tensor, step: int, mode: str) -> torch.Tensor:
        """Convert logits into the next token embedding based on generation mode."""
        pass

    @abstractmethod
    def _get_token_embedding(self, token_indices: torch.Tensor, step: int) -> torch.Tensor:
        """Fetch the embedding for explicit token indices (used by beam search)."""
        pass

    @abstractmethod
    def _get_supervised_embeddings(self, y: torch.Tensor, batch_size: int, device: torch.device) -> list[torch.Tensor]:
        """Map ground truth labels to the target embedding sequence (excluding BOS)."""
        pass

    # --- Shared Boilerplate ---
    def _init_autoregressive_components(self, decoder_cls: type | str, decoder_kwargs: dict[str, Any] | None = None):
        if isinstance(decoder_cls, str):
            decoder_cls = import_class(decoder_cls)
            
        self.positional = nn.Embedding(num_embeddings=self.sequence_length, embedding_dim=self.preclassification_size)
        self.BOS = nn.Embedding(num_embeddings=1, embedding_dim=self.preclassification_size)
        
        decoder_kwargs = decoder_kwargs or {}
        decoder_kwargs["d_model"] = self.preclassification_size
        self.decoder = decoder_cls(**decoder_kwargs)
        
        self._generators: dict[str, Callable[..., torch.Tensor]] = {}
        for method_name in dir(self):
            method = getattr(self, method_name)
            name = getattr(method, "__register_name__", None)
            if name is not None:
                self._generators[name] = method

    def _prepare_generate(self, x: torch.Tensor):
        context = self.preclassification(x)
        if EmbeddingContext.active():
            EmbeddingContext.set(context)

        batch_size = context.shape[0]
        device = context.device
        BOS: torch.Tensor = self.BOS(torch.zeros((batch_size,), dtype=torch.long, device=device))
        POS = self.positional.weight.unsqueeze(1)

        return context, BOS, POS

    # --- Unified Generators ---
    def _standard_generate(self, x: torch.Tensor, mode: str):
        context, BOS, POS = self._prepare_generate(x)
        decision = BOS.unsqueeze(0).repeat(self.sequence_length, 1, 1)
        
        for step in range(self.sequence_length - 1):
            sequence = self.decoder(tgt=decision + POS, memory=context.unsqueeze(0), tgt_is_causal=True)
            logits = self._get_step_logits(sequence, step)
            decision[step + 1] = self._get_step_decision(sequence, logits, step, mode=mode)
            
        return sequence

    @register_generator("geometric")
    def _geometric_generate(self, x: torch.Tensor): return self._standard_generate(x, "geometric")

    @register_generator("soft")
    def _soft_generate(self, x: torch.Tensor): return self._standard_generate(x, "soft")

    @register_generator("greedy")
    def _greedy_generate(self, x: torch.Tensor): return self._standard_generate(x, "greedy")

    @register_generator("supervised")
    def _supervised_generate(self, x: torch.Tensor, y: Any):
        context, BOS, POS = self._prepare_generate(x)
        batch_size, device = context.shape[0], context.device
        
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long, device=device, requires_grad=False)
            
        sequence_embs = self._get_supervised_embeddings(y, batch_size, device)
        sequence_embs.append(BOS)
        sequence_tgt = torch.stack(sequence_embs[::-1], dim=0)
        
        return self.decoder(tgt=sequence_tgt + POS, memory=context.unsqueeze(0), tgt_is_causal=True)

    @register_generator("beam search")
    def _beam_generate(self, x: torch.Tensor, beam_width: int = 32) -> torch.Tensor:
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}")

        context, BOS, POS = self._prepare_generate(x)
        batch_size = x.shape[0]
        d_model = BOS.shape[1]

        decision = BOS.unsqueeze(0).repeat(self.sequence_length, 1, 1).unsqueeze(2)  # [seq, batch, beam=1, dim]
        beam_scores = torch.zeros(batch_size, 1, device=x.device)

        for step in range(self.sequence_length - 1):
            current_beam_width = decision.shape[2]
            decision_flat = decision.reshape(self.sequence_length, batch_size * current_beam_width, d_model)
            context_flat = context.unsqueeze(1).expand(batch_size, current_beam_width, context.shape[-1]).reshape(batch_size * current_beam_width, context.shape[-1])

            sequence = self.decoder(tgt=decision_flat + POS, memory=context_flat.unsqueeze(0), tgt_is_causal=True)

            log_probs = self._get_step_logits(sequence, step).log_softmax(dim=1)
            vocab_size = log_probs.shape[1]

            log_probs = log_probs.reshape(batch_size, current_beam_width, vocab_size)
            candidate_scores = (beam_scores.unsqueeze(-1) + log_probs).reshape(batch_size, current_beam_width * vocab_size)

            next_beam_width = min(beam_width, current_beam_width * vocab_size)
            top_scores, top_indices = candidate_scores.topk(next_beam_width, dim=1)

            parent_beam = top_indices // vocab_size
            token_index = top_indices % vocab_size

            gather_decision = parent_beam.view(1, batch_size, next_beam_width, 1).expand(self.sequence_length, batch_size, next_beam_width, d_model)
            decision = decision.gather(dim=2, index=gather_decision)

            decision[step + 1] = self._get_token_embedding(token_index, step)
            beam_scores = top_scores

        best_beam = beam_scores.argmax(dim=1)
        best_index = best_beam.view(1, batch_size, 1, 1).expand(self.sequence_length, batch_size, 1, d_model)

        return self.decoder(tgt=decision.gather(dim=2, index=best_index).squeeze(2) + POS, memory=context.unsqueeze(0), tgt_is_causal=True)

    # --- Standard Execution Flow ---
    def generator(self, method: str) -> Callable[..., torch.Tensor]:
        if len(method.strip()) == 0:
            raise ValueError("Generator method must contain at least 1 non-space character.")

        query = method.casefold()
        matches: list[str] = []

        for name, generator in self._generators.items():
            folded = name.casefold()
            if folded == query:
                return generator
            if folded.startswith(query):
                matches.append(name)

        match len(matches):
            case 0:
                raise ValueError(
                    f'Generator "{method}" is not a registered method of {type(self).__name__}. Valid options are {list(self._generators)}'
                )
            case 1:
                return self._generators[matches[0]]
            case _:
                raise ValueError(
                    f'Generator "{method}" is ambiguous for {type(self).__name__}; '
                    f"it matches {matches}. Valid options are {list(self._generators)}"
                )

    def generate(self, x: torch.Tensor, method: str, **kwargs):
        return self.generator(method)(x=x, **kwargs)

    def forward(self, x: torch.Tensor, y: Any = None, method : str="beam"):
        if y is None:
            y = SupervisionContext.get()
        if self.training:
            if y is None or torch.rand(1).item() > 0.5:
                sequence = self.generate(x=x, method="soft")
            else:
                sequence = self.generate(x=x, y=y, method="supervised")
        else:
            sequence = self.generate(x=x, method=method)
        return self.classify(sequence)