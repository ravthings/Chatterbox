import json, random, os, urllib.request

MODEL = "llama3.2:3b"
OUT = "cricket_qa.jsonl"

# split cricket.txt into ~1200-character chunks
with open("cricket.txt", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]
chunks, cur = [], ""
for l in lines:
    cur += l + "\n"
    if len(cur) > 1200:
        chunks.append(cur)
        cur = ""
random.seed(0)
random.shuffle(chunks)                     # so we cover lots of different pages

