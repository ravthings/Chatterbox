from datasets import load_dataset
import tiktoken, json

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token
block_size = 128

ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train_sft")

examples, seen, too_long = [], set(), 0
for row in ds:
    msgs = row["messages"]
    for i in range(len(msgs) - 1):                       # find every user -> assistant pair
        user, bot = msgs[i], msgs[i + 1]
        if user["role"] != "user" or bot["role"] != "assistant":
            continue
        key = (user["content"], bot["content"])
        if key in seen:                                   # skip repeats
            continue
        seen.add(key)

        prompt = enc.encode(f"User: {user['content']}\nBot:")
        reply = enc.encode(" " + bot["content"]) + [EOT]
        if len(prompt) + len(reply) > block_size:         # must fit in 128 tokens
            too_long += 1
            continue
        examples.append({"prompt": prompt, "reply": reply})

print("examples:", len(examples), "| skipped (too long):", too_long)
print("---")
ex = examples[10]
print(enc.decode(ex["prompt"] + ex["reply"]))

with open("chat_data.json", "w") as f:
    json.dump(examples, f)
print("saved chat_data.json")