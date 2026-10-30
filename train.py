import os
import torch
from model import NextTokenTransformer

# ---------------- config ----------------
DATA_FILE   = "data.txt"      # any plain-text file (put it in this folder)
BLOCK_SIZE  = 64
BATCH_SIZE  = 32
D_MODEL     = 128
N_LAYERS    = 4
LR          = 3e-4
MAX_STEPS   = 3000
EVAL_EVERY  = 300
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# ---------------- data (character-level tokenizer) ----------------
if not os.path.exists(DATA_FILE):
    raise SystemExit(f"Put a text file named '{DATA_FILE}' in this folder (e.g. Tiny Shakespeare).")
text = open(DATA_FILE, encoding="utf-8").read()
if not text:
    raise SystemExit(
        f"'{DATA_FILE}' is empty. Add a plain-text training corpus before running training."
    )

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
min_split_size = BLOCK_SIZE + 2
if len(train_data) < min_split_size or len(val_data) < min_split_size:
    raise SystemExit(
        f"'{DATA_FILE}' is too short for BLOCK_SIZE={BLOCK_SIZE} with a 90/10 split. "
        f"Each split needs at least {min_split_size} characters; add more text or reduce BLOCK_SIZE."
    )

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])   # targets = inputs shifted by 1
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, iters=50):
    model.eval()
    out = {}
    for split in ("train", "val"):
        out[split] = sum(model(*get_batch(split))[1].item() for _ in range(iters)) / iters
    model.train()
    return out

# ---------------- train ----------------
model = NextTokenTransformer(vocab_size, D_MODEL, N_LAYERS, BLOCK_SIZE).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"vocab={vocab_size}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M  device={device}")

for step in range(MAX_STEPS + 1):
    if step % EVAL_EVERY == 0:
        l = estimate_loss(model)
        print(f"step {step:5d} | train {l['train']:.3f} | val {l['val']:.3f}")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

torch.save({"model": model.state_dict(), "stoi": stoi, "itos": itos}, "checkpoint.pt")

# ---------------- sample ----------------
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n--- generated sample ---")
print(decode(model.generate(start, 300, temperature=0.8, top_k=40)[0].tolist()))
