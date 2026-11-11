import os, time
from functools import partial
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import tiktoken
from mlx.utils import tree_flatten
from model import TinyModel, block_size

enc = tiktoken.get_encoding("gpt2")
save_file = "cricket_base.safetensors"

# data: stories (for English) + cricket (for knowledge)
stories = mx.load("tokens_gpt2.npy")
with open("cricket.txt", encoding="utf-8") as f:
    cricket = mx.array(enc.encode_ordinary(f.read()))
print(f"stories: {len(stories):,} tokens | cricket: {len(cricket):,} tokens")

def split(d):
    n = int(0.95 * len(d))
    return d[:n], d[n:]
st_tr, st_va = split(stories)
cr_tr, cr_va = split(cricket)

batch_size = 32

def sample(d, n):
    ix = mx.random.randint(0, len(d) - block_size - 1, (n,))
    chunk = d[ix[:, None] + mx.arange(block_size + 1)[None, :]]   # speed-up 1: grab all chunks at once
    return chunk[:, :-1], chunk[:, 1:]

def get_batch(stories_d, cricket_d):
    xs, ys = sample(stories_d, batch_size // 2)    # half stories
    xc, yc = sample(cricket_d, batch_size // 2)    # half cricket
    return mx.concatenate([xs, xc]), mx.concatenate([ys, yc])

def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y, reduction="mean")

model = TinyModel()
mx.eval(model.parameters())
print("parameters:", sum(v.size for _, v in tree_flatten(model.parameters())))
if os.path.exists(save_file):
    model.load_weights(save_file)
    print("Resuming from", save_file)

epochs = int(input("How many steps? "))
warmup = min(200, epochs // 10)
lr = optim.join_schedules(                                         # speed-up 2: warm up, then slowly lower the learning rate
    [optim.linear_schedule(1e-5, 1e-3, warmup), optim.cosine_decay(1e-3, steps - warmup, 1e-4)],
    [warmup])
optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.05)
loss_and_grad = nn.value_and_grad(model, loss_fn)

state = [model.state, optimizer.state]
@partial(mx.compile, inputs=state, outputs=state)                  # speed-up 3: compile the training step
def train_step(x, y):
    loss, grads = loss_and_grad(model, x, y)
    grads, _ = optim.clip_grad_norm(grads, 1.0)                    # stops sudden huge updates
    optimizer.update(model, grads)
    return loss

t0 = time.time()
for epoch in range(epochs):
    x, y = get_batch(st_tr, cr_tr)
    loss = train_step(x, y)
    mx.eval(state)

    if epoch % 250 == 0:
        xs, ys = sample(st_va, 32)
        xc, yc = sample(cr_va, 32)
        secs = (time.time() - t0) / (epoch + 1)
        print(f"step {epoch}: train {loss.item():.3f} | val stories {loss_fn(model, xs, ys).item():.3f} "
              f"| val cricket {loss_fn(model, xc, yc).item():.3f} | {secs:.2f}s/step")
    if epoch % 1000 == 0 and epoch > 0:
        model.save_weights(save_file)                              # save every 1000 steps, just in case

model.save_weights(save_file)
print("saved", save_file)

def generate(prompt, n=80):
    idx = mx.array([enc.encode(prompt)])
    for _ in range(n):
        logits = model(idx[:, -block_size:])[:, -1, :]
        idx = mx.concatenate([idx, mx.random.categorical(logits / 0.8, num_samples=1)], axis=1)
    return enc.decode(idx[0].tolist())

for p in ["Cricket is", "Sachin Tendulkar", "Once upon a time,"]:
    print("\n---", generate(p))
