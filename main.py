"""
Complete GPT training and generation pipeline.
Trains on Tiny Shakespeare and generates text.
"""

import os
import torch
import torch.nn as nn
from model.gpt import GPT
from data.vocab import Solution as VocabSolution
from data.loader import Solution as LoaderSolution
from train import Solution as TrainSolution
from generate import Solution as GenerateSolution


def load_data(filepath="data/input.txt"):
    """Load raw text data from file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please download Tiny Shakespeare.")
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def build_vocab_and_encode(text):
    """Build character-level vocabulary and encode text to token IDs."""
    vocab_solution = VocabSolution()
    stoi, itos = vocab_solution.build_vocab(text)
    encoded = torch.tensor(vocab_solution.encode(text, stoi), dtype=torch.long)
    return stoi, itos, encoded


def create_model(vocab_size, context_length=64, model_dim=128, num_heads=4, num_blocks=2, device='cpu'):
    """Instantiate GPT model."""
    model = GPT(
        vocab_size=vocab_size,
        context_length=context_length,
        model_dim=model_dim,
        num_blocks=num_blocks,
        num_heads=num_heads
    )
    model = model.to(device)
    return model


def train_model(model, data, epochs=1000, context_length=64, batch_size=16, lr=3e-4, device='cpu'):
    """Train the GPT model and return final loss."""
    trainer = TrainSolution()
    data = data.to(device)
    
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")
    print(f"Epochs: {epochs} | Context: {context_length} | Batch Size: {batch_size} | LR: {lr}")
    print(f"Dataset size: {len(data)} tokens")
    print(f"{'='*60}\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        torch.manual_seed(epoch)
        ix = torch.randint(len(data) - context_length, (batch_size,))
        x = torch.stack([data[i:i + context_length] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + context_length] for i in ix]).to(device)
        
        logits = model(x)
        B, T, C = logits.shape
        loss = nn.functional.cross_entropy(logits.view(B * T, C), y.view(B * T))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1:4d} / {epochs} | Loss: {loss.item():.4f}")
    
    print(f"\nTraining complete. Final loss: {loss.item():.4f}\n")
    return loss.item()


def save_checkpoint(model, itos, checkpoint_path="gpt_model.pth"):
    """Save model and vocabulary checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'itos': itos,
        'model_config': {
            'vocab_size': model.vocab_projection.out_features,
            'context_length': model.position_embeddings.num_embeddings,
            'model_dim': model.word_embeddings.embedding_dim,
            'num_blocks': len(model.transformer_blocks),
            'num_heads': 4,  # Default assumption
        }
    }, checkpoint_path)
    print(f"✓ Model checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path="gpt_model.pth", device='cpu'):
    """Load model and vocabulary from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['model_config']
    model = GPT(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    itos = checkpoint['itos']
    return model, itos


def generate_text(model, itos, context_length=64, new_chars=500, device='cpu'):
    """Generate text from the trained model."""
    generator = GenerateSolution()
    
    # Start with a few random characters as context
    initial_tokens = torch.randint(0, len(itos), (1, 1), device=device)
    
    print(f"\n{'='*60}")
    print("GENERATING TEXT")
    print(f"{'='*60}\n")
    
    generated_text = generator.generate(
        model=model,
        new_chars=new_chars,
        context=initial_tokens,
        context_length=context_length,
        int_to_char=itos,
        temperature=0.8,
        top_k=30,
        top_p=0.9,
    )
    
    print(generated_text)
    print(f"\n{'='*60}\n")
    return generated_text


def main():
    """Main training and generation pipeline."""
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # 1. Load data
    print("1. Loading data...")
    text = load_data()
    print(f"   Loaded {len(text)} characters\n")
    
    # 2. Build vocabulary and encode
    print("2. Building vocabulary...")
    stoi, itos, encoded_data = build_vocab_and_encode(text)
    vocab_size = len(stoi)
    print(f"   Vocabulary size: {vocab_size} characters\n")
    
    # 3. Create model
    print("3. Creating model...")
    context_length = 64
    model_dim = 256
    num_heads = 4
    num_blocks = 4
    model = create_model(vocab_size, context_length, model_dim, num_heads, num_blocks, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   GPT model created with {total_params:,} parameters\n")
    
    # 4. Train model
    print("4. Training model...")
    batch_size = 32
    lr = 3e-4
    epochs = 3000
    train_model(model, encoded_data, epochs, context_length, batch_size, lr, device)
    
    # 5. Save checkpoint
    print("5. Saving checkpoint...")
    save_checkpoint(model, itos)
    
    # 6. Generate text
    print("6. Generating text...")
    generated = generate_text(model, itos, context_length, new_chars=500, device=device)
    
    print("✓ Pipeline complete!")


if __name__ == "__main__":
    main()
