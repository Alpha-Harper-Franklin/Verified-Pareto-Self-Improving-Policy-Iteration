from __future__ import annotations

import re

from transformers import LogitsProcessor


class CharClassLogitsProcessor(LogitsProcessor):
    """
    Conservative character-class based logits processor.

    Penalizes tokens that decode to characters outside a safe DSL subset.
    This is intentionally simple and language-agnostic: it is NOT a template
    fallback and does not inject content; it only suppresses clearly-invalid
    characters during decoding.
    """

    def __init__(self, tokenizer, penalty: float = 30.0):
        self.tokenizer = tokenizer
        self.penalty = float(penalty)

        # Allow a conservative ASCII subset that covers: INC DSL, node names, numbers,
        # SI suffixes, punctuation, and whitespace/newlines.
        self._allowed = re.compile(r"^[A-Za-z0-9\n\r\t \-_=:/\\.,+*()\[\]{}%]*$")

        penalties = []
        for tok_id in range(len(tokenizer)):
            s = tokenizer.decode([tok_id], skip_special_tokens=True)
            if not s or (not self._allowed.match(s)):
                penalties.append(self.penalty)
            else:
                penalties.append(0.0)

        import torch

        self._penalty_tensor = torch.tensor(penalties)

    def __call__(self, input_ids, scores):
        import torch

        vocab = int(scores.shape[-1])
        if self._penalty_tensor.numel() != vocab:
            cur = int(self._penalty_tensor.numel())
            if cur < vocab:
                pad = torch.full(
                    (vocab - cur,),
                    float(self.penalty),
                    dtype=self._penalty_tensor.dtype,
                    device=self._penalty_tensor.device,
                )
                self._penalty_tensor = torch.cat([self._penalty_tensor, pad], dim=0)
            else:
                self._penalty_tensor = self._penalty_tensor[:vocab]
        if self._penalty_tensor.device != scores.device:
            self._penalty_tensor = self._penalty_tensor.to(scores.device)
        scores = scores.clone()
        scores[..., :] -= self._penalty_tensor
        return scores
