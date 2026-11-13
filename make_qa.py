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

PROMPT = """Here is some text about cricket:

{text}

Write 3 questions a curious kid might ask about cricket, and answer each one using ONLY facts from the text above.
Rules:
- Each answer is 1 to 3 short, simple sentences.
- Questions must make sense on their own (never say "the text" or "the passage").
- If the text has no useful facts, return an empty list.
Reply in JSON: {{"pairs": [{{"q": "...", "a": "..."}}]}}"""

def ask(text):
    body = json.dumps({
        "model": MODEL, "stream": False, "format": "json",
        "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
        "options": {"temperature": 0.7},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(json.loads(r.read())["message"]["content"])

# resume where we stopped last time
start = 0
if os.path.exists(OUT):
    with open(OUT) as f:
        done = [json.loads(l)["chunk"] for l in f if l.strip()]
    start = max(done) + 1 if done else 0

n = int(input(f"{len(chunks)} chunks available, starting at {start}. Process up to chunk: "))
total = 0
with open(OUT, "a") as out:
    for i in range(start, min(n, len(chunks))):
        try:
            pairs = ask(chunks[i]).get("pairs", [])
        except Exception as e:
            print("skip", i, e)
            continue
        for p in pairs:
            q, a = str(p.get("q", "")).strip(), str(p.get("a", "")).strip()
            if q and a:
                out.write(json.dumps({"chunk": i, "q": q, "a": a}) + "\n")
                total += 1
        out.flush()
        if i % 10 == 0 and pairs:
            print(f"chunk {i}: {total} pairs so far | e.g. Q: {pairs[0].get('q')} A: {pairs[0].get('a')}")
print("done,", total, "new pairs")