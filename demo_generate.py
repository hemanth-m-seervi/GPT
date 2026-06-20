import sys
import argparse
import torch

from main import load_checkpoint
from generate import Solution as GenerateSolution


def prompt_to_tensor(prompt: str, itos, device: str):
    stoi = {ch: i for i, ch in enumerate(itos)}
    tokens = [stoi.get(ch, 0) for ch in prompt]
    if len(tokens) == 0:
        # fallback to a single random token
        return torch.randint(0, len(itos), (1, 1), device=device)
    return torch.tensor([tokens], dtype=torch.long, device=device)


def greedy_generate(model, context, itos, new_chars: int, context_length: int, device: str):
    model.eval()
    result = []
    ctx = context
    for _ in range(new_chars):
        if ctx.shape[1] > context_length:
            ctx = ctx[:, -context_length:]
        logits = model(ctx)  # (1, T, vocab)
        last_logits = logits[:, -1, :]
        next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
        ctx = torch.cat((ctx, next_token), dim=-1)
        result.append(itos[next_token.item()])
    return ''.join(result)


def main():
    parser = argparse.ArgumentParser(description='Demo generation')
    parser.add_argument('prompt', nargs='*', help='Optional prompt to start generation')
    parser.add_argument('--chars', type=int, default=500, help='Number of characters to generate')
    parser.add_argument('--greedy', action='store_true', help='Use deterministic greedy autocomplete (argmax)')
    parser.add_argument('--temp', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=30, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    parser.add_argument('--no-repeat-ngram', type=int, default=0, help='Prevent repeating n-grams of this length (0 disables)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = 'gpt_model.pth'

    print(f'Loading checkpoint from {ckpt_path} on {device}...')
    model, itos = load_checkpoint(ckpt_path, device=device)

    prompt = ' '.join(args.prompt) if len(args.prompt) > 0 else ''
    context = prompt_to_tensor(prompt, itos, device)

    if args.greedy:
        generated = greedy_generate(model, context, itos, args.chars, context_length=64, device=device)
    else:
        # Custom sampling that supports no-repeat-ngram
        def sampling_generate(model, context, itos, new_chars, context_length, device, temp, top_k, top_p, no_repeat_ngram):
            model.eval()
            ctx = context
            generated_tokens = []
            vocab_size = len(itos)

            for _ in range(new_chars):
                if ctx.shape[1] > context_length:
                    ctx = ctx[:, -context_length:]

                logits = model(ctx)  # (1, T, V)
                last_logits = logits[:, -1, :].float()  # (1, V)

                # Temperature
                last_logits = last_logits / max(temp, 1e-8)

                probs = torch.softmax(last_logits, dim=-1)

                # Top-k
                if top_k is not None and top_k > 0:
                    topk_vals, topk_idx = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
                    filtered = torch.zeros_like(probs)
                    filtered.scatter_(dim=-1, index=topk_idx, src=topk_vals)
                    probs = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                # Top-p (nucleus)
                if top_p is not None and 0 < top_p < 1:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = cumulative > top_p
                    cutoff[..., 0] = False
                    sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    next_sorted = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = sorted_idx.gather(dim=-1, index=next_sorted)
                    candidate = next_token.item()
                else:
                    # apply no-repeat-ngram before sampling if requested
                    if no_repeat_ngram and no_repeat_ngram > 0:
                        probs = probs.squeeze(0)
                        banned = torch.zeros_like(probs, dtype=torch.bool)
                        prev = ctx.squeeze(0).tolist()
                        N = no_repeat_ngram
                        if N <= 0:
                            pass
                        else:
                            # prefix is last N-1 tokens
                            prefix = prev[-(N - 1):] if N > 1 and len(prev) >= (N - 1) else []
                            # collect existing N-grams
                            existing_ngrams = set()
                            if len(prev) >= N:
                                for i in range(len(prev) - N + 1):
                                    existing_ngrams.add(tuple(prev[i:i+N]))

                            for token_idx in range(vocab_size):
                                if N == 1:
                                    cand = (token_idx,)
                                else:
                                    cand = tuple(prefix + [token_idx]) if len(prefix) == (N - 1) else None
                                if cand is not None and cand in existing_ngrams:
                                    banned[token_idx] = True

                        probs = probs.masked_fill(banned, 0.0)
                        if probs.sum() <= 0:
                            # fallback to argmax if all banned
                            candidate = int(torch.argmax(last_logits, dim=-1).item())
                        else:
                            probs = probs / probs.sum()
                            candidate = int(torch.multinomial(probs, num_samples=1).item())
                        next_token = torch.tensor([[candidate]], device=device)
                    else:
                        # standard multinomial sampling
                        probs = probs.squeeze(0)
                        candidate = int(torch.multinomial(probs, num_samples=1).item())
                        next_token = torch.tensor([[candidate]], device=device)

                ctx = torch.cat((ctx, next_token.to(device)), dim=-1)
                generated_tokens.append(next_token.item())

            return ''.join([itos[t] for t in generated_tokens])

        generated = sampling_generate(
            model, context, itos, args.chars, context_length=64, device=device,
            temp=args.temp, top_k=args.top_k, top_p=args.top_p, no_repeat_ngram=args.no_repeat_ngram
        )

    print('\n=== Generated sample ===\n')
    print(generated)

    out_path = 'sample_output.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(generated)

    print(f'\nSaved generated text to {out_path}')


if __name__ == '__main__':
    main()
