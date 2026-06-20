# My GPT — Built from Scratch

> Assembled from the NeetCode ML course on [NeetCode.io](https://neetcode.io)
> Built by **Hemanth M Sirvi** on June 19, 2026

Every file in this project is code I wrote and submitted while completing the NeetCode ML course.
The problems progressively build from gradient descent fundamentals all the way to a working GPT.

## What This Project Includes

This repository contains a full from-scratch GPT pipeline:

- `foundations/` for the math and neural network building blocks
- `model/` for embeddings, attention, transformer blocks, normalization, and GPT
- `data/` for tokenization, preprocessing, vocabulary building, and batching
- `train.py` for the reusable training loop
- `generate.py` for autoregressive sampling with temperature, top-k, and top-p filtering
- `main.py` for the end-to-end train-and-generate workflow

## Dataset

The current pipeline trains on `data/input.txt`, which contains the Tiny Shakespeare dataset.

Current run details:

- Raw text size: 1,115,393 characters
- Character-level vocabulary size: 65 unique characters
- Tokenization: character-level

## Training Details

The active training configuration in `main.py` is:

- Context length: 64
- Batch size: 32
- Learning rate: 0.0003
- Epochs: 3000
- Optimizer: AdamW
- Model size: 3,205,185 parameters

Latest observed training results:

- Epoch 100: loss 2.4891
- Epoch 500: loss 1.9626
- Epoch 1000: loss 1.8332
- Epoch 1500: loss 1.6462
- Epoch 2000: loss 1.6118
- Epoch 2500: loss 1.4865
- Epoch 3000: loss 1.3789

During training, the model prints loss every 100 epochs, saves a checkpoint to `gpt_model.pth`, and then generates sample text from the trained model.

## Stronger Training Setup

If you want better text quality, use this stronger configuration in [main.py](main.py):

- Model dimension: 256
- Number of transformer blocks: 4
- Batch size: 32
- Epochs: 3000

Expected result from the latest run:

- Final loss reached 1.3789
- Generated text should be noticeably more coherent than the smaller baseline run

## Project Structure

```
model/          Attention, Transformer, GPT architecture
  attention.py                Self-attention head
  multi_head_attention.py     Multi-headed attention
  transformer.py              Transformer block
  gpt.py                      GPT model
  normalization.py            Layer normalization
  batch_normalization.py      Batch normalization
  rms_normalization.py        RMS normalization
  embeddings.py               Word embeddings
  positional_encoding.py      Positional encoding
  kv_cache.py                 KV-cache for fast inference
  grouped_query_attention.py  Grouped query attention

data/           Data pipeline
  tokenizer.py                BPE tokenizer
  vocab.py                    Character-level vocabulary
  loader.py                   Batched training data loader
  dataset.py                  GPT dataset preparation
  nlp_preprocessing.py        NLP preprocessing
  tokenizer_utils.py          Tokenization edge cases

train.py        GPT training loop
generate.py     Text generation
main.py         End-to-end train/save/generate pipeline

foundations/    Neural network primitives built from scratch
  neuron.py, backprop.py, mlp.py, activations.py, loss.py,
  training_loop.py, dead_relu_detector.py, ...
```

## Quick Start

1. Activate your virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the end-to-end pipeline:

```bash
python main.py
```

If you want to run the helper scripts separately, you can import the `Solution` classes from `train.py` and `generate.py` in your own driver code.

## Output Files

- `gpt_model.pth`: saved checkpoint containing the trained model weights, vocabulary, and model config

## Demo & Sharing

To quickly show the model output to someone, use the included demo script or share the generated text file and checkpoint.

- Generate a sample (loads `gpt_model.pth`, generates 500 chars, saves to `sample_output.txt`):

```bash
python demo_generate.py
```

- Generate with a short prompt:

```bash
python demo_generate.py "To be, or not to be"
```

- The script prints the sample and writes it to `sample_output.txt` in the repo root. Share `sample_output.txt` to show outputs without requiring others to run the model.

- Deterministic autocomplete (greedy argmax):

```bash
python demo_generate.py --greedy "To be, or not to be"
```

- Control the number of generated characters:

```bash
python demo_generate.py --chars 200 "Hello"
python demo_generate.py --chars 200 --greedy "Hello"
```

- To let others reproduce your demo, share these files:
  - `gpt_model.pth` (the trained checkpoint)
  - `sample_output.txt` (example output)
  - `README.md` (this file) and `data/input.txt` (dataset reference)



## Generation Settings

The generation step uses the trained checkpoint and samples text autoregressively with:

- temperature: 0.8
- top-k: 30
- top-p: 0.9

```bash
pip install -r requirements.txt
python main.py
```

## Course

This project was built by completing the [NeetCode ML Course](https://neetcode.io/practice?tab=coreSkills&topic=Machine+Learning):
- Math Foundations (gradient descent, activations, loss functions)
- Neural Networks from scratch (neuron, backprop, MLP)
- PyTorch fundamentals
- NLP pipeline (embeddings, tokenization, attention)
- Transformer architecture
- GPT model + text generation
