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

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = nn.Linear(num_heads * head_size, n_embd, bias=False)

    def __call__(self, x):
        return self.proj(mx.concatenate([h(x) for h in self.heads], axis=-1))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 4 * n_embd * 2 // 3
        self.gate_proj = nn.Linear(n_embd, hidden, bias=False)
        self.up_proj   = nn.Linear(n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, n_embd, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

class Block(nn.Module):
    def __init__(self, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward()
        self.ln1 = nn.RMSNorm(n_embd)
        self.ln2 = nn.RMSNorm(n_embd)

    def __call__(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x