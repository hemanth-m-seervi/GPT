import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(
        self,
        model,
        new_chars: int,
        context: TensorType[int],
        context_length: int,
        int_to_char: dict,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float | None = 0.9,
    ) -> str:
        """Autoregressive generation with temperature + top-k/top-p sampling."""
        model.eval()

        # Keep RNG behavior normal; do NOT reset RNG every step.
        result: list[str] = []

        for _ in range(new_chars):
            if context.shape[1] > context_length:
                context = context[:, -context_length:]

            logits = model(context)  # (1, T, vocab_size)
            last_logits = logits[:, -1, :]  # (1, vocab_size)

            # Temperature
            last_logits = last_logits / max(temperature, 1e-8)

            # Convert to probabilities
            probs = nn.functional.softmax(last_logits, dim=-1)

            # Top-k filtering
            if top_k is not None and top_k > 0:
                topk_vals, topk_idx = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
                filtered = torch.zeros_like(probs)
                filtered.scatter_(dim=-1, index=topk_idx, src=topk_vals)
                probs = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            # Top-p (nucleus) filtering
            if top_p is not None and 0 < top_p < 1:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumulative > top_p
                cutoff[..., 0] = False  # always keep at least 1

                sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                # Sample from filtered sorted distribution, then map back
                next_sorted = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_idx.gather(dim=-1, index=next_sorted)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            context = torch.cat((context, next_token), dim=-1)
            result.append(int_to_char[next_token.item()])

        return ''.join(result)

