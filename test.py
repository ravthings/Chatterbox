import mlx.core as mx
import tiktoken
from model import TinyModel, block_size

enc = tiktoken.get_encoding("gpt2")
model = TinyModel()
model.load_weights("bedtime_llama.safetensors")

idx = mx.array([enc.encode("Once upon a time,")])
for _ in range(60):
    logits = model(idx[:, -block_size:])[:, -1, :]
    idx = mx.concatenate([idx, mx.random.categorical(logits, num_samples=1)], axis=1)
print(enc.decode(idx[0].tolist()))