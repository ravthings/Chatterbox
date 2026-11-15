import json, random, re
from collections import Counter
from datasets import load_dataset
import tiktoken

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token
block_size = 256

# ---------- 1. clean the cricket Q&A ----------
VAGUE = ["the text", "passage", "mentioned", "this match", "this tournament", "this series",
         "this player", "in this", "the match?", "the final?", "the series?", "the tournament?"]

def sentences(a):
    return [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", a) if s.strip()]
STOP = set("the a an of in on at to for is was were are who what which when where how did does do "
        "by with and or from that this as be has have had it its his her their name".split())
def keywords(t):
    return {w for w in re.findall(r"[a-z0-9']+", t.lower()) if w not in STOP and len(w) > 2}

cricket, seen, why = [], set(), Counter()
with open("cricket_qa.jsonl") as f:
    for line in f:
        r = json.loads(line)
        q, a = r["q"].strip(), r["a"].strip()
        ql, al = q.lower(), a.lower()
        if ql in seen:
            why["duplicate question"] += 1; continue
        if any(v in ql or v in al for v in VAGUE):
            why["vague"] += 1; continue
        s = sentences(a)
        if len(set(s)) < len(s):
            why["repeated sentence"] += 1; continue
        if not 5 <= len(a.split()) <= 60:
            why["too short/long"] += 1; continue
        if len(keywords(q) & keywords(a)) < 2:
            why["off-topic answer"] += 1; continue
        seen.add(ql)
        cricket.append((q, a))

print(f"cricket: kept {len(cricket)} | removed: {dict(why)}")
random.seed(1)
print("\n--- 10 random kept pairs ---")
for q, a in random.sample(cricket, 10):
    print(f"Q: {q}\nA: {a}\n")

# ---------- 2. who-am-I examples (write your own!) ----------
me = [
    ("Who are you?", "I'm Chatterbox, a small chatbot that knows about cricket."),
    ("What can you do?", "I can answer simple questions about cricket, like the rules, famous players and big tournaments."),
    ("Who made you?", "I was built from scratch by ravthings, using MLX on a Mac."),
]

# ---------- 3. some everyday chat, so "Hi!" still works ----------
ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train_sft")
everyday, eseen = [], set()
for row in ds:
    m = row["messages"]
    for i in range(len(m) - 1):
        if m[i]["role"] == "user" and m[i + 1]["role"] == "assistant":
            key = m[i]["content"]
            if key not in eseen:
                eseen.add(key)
                everyday.append((m[i]["content"], m[i + 1]["content"]))
random.shuffle(everyday)
everyday = everyday[:1500]                       # keep cricket the majority

# ---------- 4. turn into training examples ----------
def to_example(q, a):
    prompt = enc.encode(f"User: {q}\nBot:")
    reply = enc.encode(" " + a) + [EOT]
    if len(prompt) + len(reply) > block_size:
        return None
    return {"prompt": prompt, "reply": reply}

pairs = cricket + me * 20 + everyday             # repeat the who-am-I ones so they stick
examples = [e for e in (to_example(q, a) for q, a in pairs) if e]
random.shuffle(examples)

with open("chat_v2.json", "w") as f:
    json.dump(examples, f)
print(f"saved chat_v2.json: {len(examples)} examples "
      f"({len(cricket)} cricket, {len(me)}x20 about me, {len(everyday)} everyday)")