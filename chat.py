import mlx.core as mx
import tiktoken
from model import TinyModel, block_size

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

model = TinyModel()
model.load_weights("chatterbox.safetensors")

def stream_reply(question, max_new=100, temp=0.7):
    idx = mx.array([enc.encode(f"User: {question}\nBot:")])
    print("Bot:", end="", flush=True)
    for _ in range(max_new):
        logits = model(idx[:, -block_size:])[:, -1, :]
        nxt = mx.random.categorical(logits / temp, num_samples=1)
        if nxt.item() == EOT:
            break
        idx = mx.concatenate([idx, nxt], axis=1)
        print(enc.decode([nxt.item()]), end="", flush=True)
    print("\n")

print("Chatterbox 🗣️  (type 'quit' to stop)\n")
while True:
    q = input("You: ")
    if q.strip().lower() == "quit":
        break
    stream_reply(q)