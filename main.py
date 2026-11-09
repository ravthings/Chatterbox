import mlx.core as mx
import mlx.nn as nn

vocab_size = 50257
block_size = 128
n_embd = 128

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.rope  = nn.RoPE(head_size)
        self.head_size = head_size

    def __call__(self, x):
        B, T, C = x.shape
        k = self.rope(self.key(x))
        q = self.rope(self.query(x))
        v = self.value(x)
        scores = q @ k.transpose(0, 2, 1) * self.head_size ** -0.5
        mask = mx.tril(mx.ones((T, T))) == 1
        scores = mx.where(mask, scores, float("-inf"))
        weights = mx.softmax(scores, axis=-1)
        return weights @ v