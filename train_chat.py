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
model.load_weights("bedtime_llama.safetensors")

# load conversations, keep 5% aside to check it's really learning
with open("chat_data.json") as f:
    data = json.load(f)
random.seed(0)
random.shuffle(data)
n_val = len(data) // 20
val_data, train_data = data[:n_val], data[n_val:]

