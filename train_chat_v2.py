import json, random
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import tiktoken
from model import TinyModel, block_size

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

# load the story model as the starting point
model = TinyModel()
model.load_weights("cricket_base.safetensors")

# load conversations, keep 5% aside to check it's really learning
with open("chat_v2.json") as f:
    data = json.load(f)
random.seed(0)
random.shuffle(data)
n_val = len(data) // 20
val_data, train_data = data[:n_val], data[n_val:]

def make_batch(exs):
    T = max(len(e["prompt"]) + len(e["reply"]) for e in exs) - 1
    X, Y, M = [], [], []
    for e in exs:
        full = e["prompt"] + e["reply"]
        p = len(e["prompt"])
        x, y = full[:-1], full[1:]                           # predict the next token
        m = [1 if j >= p - 1 else 0 for j in range(len(y))]  # 1 only where the target is part of the reply
        pad = T - len(x)
        X.append(x + [EOT] * pad)
        Y.append(y + [EOT] * pad)
        M.append(m + [0] * pad)                              # padding is never graded
    return mx.array(X), mx.array(Y), mx.array(M, dtype=mx.float32)

def loss_fn(model, x, y, m):
    ce = nn.losses.cross_entropy(model(x), y, reduction="none")
    return (ce * m).sum() / m.sum()                          # average over reply tokens only

def reply(question, max_new=80, temp=0.7):
    idx = mx.array([enc.encode(f"User: {question}\nBot:")])
    out = []
    for _ in range(max_new):
        logits = model(idx[:, -block_size:])[:, -1, :]
        nxt = mx.random.categorical(logits / temp, num_samples=1)
        if nxt.item() == EOT:
            break
        out.append(nxt.item())
        idx = mx.concatenate([idx, nxt], axis=1)
    return enc.decode(out)

tests = ["Hi!", "Who are you?", "What is LBW?", "Who is Sachin Tendulkar?", "How many players are in a cricket team?"]

print("=== BEFORE fine-tuning ===")
for q in tests:
    print("User:", q, "\nBot:", reply(q), "\n")

optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.05)
loss_and_grad = nn.value_and_grad(model, loss_fn)
batch_size = 32

epochs = int(input("How many epochs would you like to train for? "))
best = float("inf")
for epoch in range(epochs):
    x, y, m = make_batch(random.sample(train_data, batch_size))
    loss, grads = loss_and_grad(model, x, y, m)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    if epoch % 50 == 0:
        # check val on ALL the val examples, so the number is reliable
        vals = [loss_fn(model, *make_batch(val_data[i:i + 64])).item()
                for i in range(0, len(val_data), 64)]
        val = sum(vals) / len(vals)
        print(f"step {epoch}: train {loss.item():.3f}, val {val:.3f}", end="")
        if val < best:
            best = val
            model.save_weights("chatterbox_v2.safetensors")
            print("  ← best so far, saved", end="")
        print()

model.load_weights("chatterbox_v2.safetensors")      # use the best version for the tests
print(f"\nbest val: {best:.3f}")

print("=== AFTER fine-tuning ===")
for q in tests:
    print("User:", q, "\nBot:", reply(q), "\n")