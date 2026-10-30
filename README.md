# 🤖 Transformer Next-Token Predictor

A minimal, from-scratch **character-level language model** built with PyTorch. It implements a decoder-only Transformer (GPT-style) that learns to predict the next character in a text corpus and can generate new text after training.

---

## 📁 Project Structure

```
transformer_nexttoken/
├── model.py          # Transformer architecture (attention, blocks, generation)
├── train.py          # Training loop, tokenizer, evaluation, and sampling
├── data.txt          # Plain-text training corpus
├── checkpoint.pt     # Saved model weights + vocab (created after training)
└── requirements.txt  # Python dependencies
```

---

## 🏗️ Architecture

The model is a **decoder-only Transformer** with causal (masked) self-attention.

| Component | Details |
|---|---|
| Tokenizer | Character-level (no external library needed) |
| Embedding | Token embedding + learned positional embedding |
| Attention | Single-head causal self-attention (scaled dot-product) |
| Blocks | Pre-LayerNorm Transformer blocks with residual connections |
| Feed-Forward | 2-layer MLP with GELU activation (4× hidden dim) |
| Output | Linear projection → vocabulary logits |

### Default Hyperparameters (set in `train.py`)

| Parameter | Value | Description |
|---|---|---|
| `D_MODEL` | 128 | Embedding / hidden dimension |
| `N_LAYERS` | 4 | Number of Transformer blocks |
| `BLOCK_SIZE` | 64 | Context window (max tokens seen) |
| `BATCH_SIZE` | 32 | Training batch size |
| `LR` | 3e-4 | AdamW learning rate |
| `MAX_STEPS` | 3000 | Total training steps |
| `EVAL_EVERY` | 300 | Steps between loss evaluations |

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Requirements:** Python 3.8+, PyTorch. A CUDA-capable GPU is optional — the script auto-detects `cuda` and falls back to `cpu`.

### 2. Prepare your training data

Place any plain-text file in the project folder and name it `data.txt`. The file must be large enough for the 90/10 train/val split (at least ~150 characters; more text = better results).

A popular starting corpus is [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

---

## 🚀 Training

```bash
python train.py
```

Training will print a progress line every 300 steps:

```
vocab=65  params=0.40M  device=cuda
step     0 | train 4.174 | val 4.175
step   300 | train 2.143 | val 2.198
step   600 | train 1.897 | val 1.963
...
--- generated sample ---
ROMEO: What light through yonder window breaks?
```

After training, a **`checkpoint.pt`** file is saved containing the model weights and the character vocabulary.

---

## 🧠 How It Works

```
Input text  →  character tokenizer  →  token + position embeddings
            →  N × TransformerBlock (LayerNorm → CausalAttention → LayerNorm → FFN)
            →  final LayerNorm  →  linear head  →  logits over vocab
            →  cross-entropy loss  (training) / multinomial sampling  (inference)
```

### Generation

The `generate()` method supports:
- **`max_new_tokens`** — number of characters to produce
- **`temperature`** — controls randomness (< 1 = sharper, > 1 = more random)
- **`top_k`** — restricts sampling to the top-*k* most likely next tokens

---

## 🔧 Customization

You can tune any of these constants at the top of [`train.py`](train.py):

```python
DATA_FILE  = "data.txt"   # path to your corpus
BLOCK_SIZE = 64           # context length
BATCH_SIZE = 32
D_MODEL    = 128          # increase for a larger model
N_LAYERS   = 4
LR         = 3e-4
MAX_STEPS  = 3000
```

---

## 📄 License

This project is released for educational purposes. Feel free to use and adapt it.
