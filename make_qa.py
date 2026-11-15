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

PROMPT = """Here is some text about cricket:

{text}

Write up to 3 question-and-answer pairs that a cricket fan might ask a chatbot.
Rules:
- Only use facts stated in the text above.
- Each question must make sense with no other context: name the player, team, match or year it is about. Never say "the text", "the match" or "he".
- Prefer important, general facts: rules, famous players, records, how the game works. Skip ball-by-ball scores and minor tournament details.
- Each answer is 1 to 3 complete sentences that mention the subject again. Never answer with just a number.
- If there are no good facts, return an empty list.
- Answers contain ONLY facts from the text: no opinions, praise or extra commentary.
- Keep each answer under 40 words.
Reply in JSON: {{"pairs": [{{"q": "...", "a": "..."}}]}}"""

def ask(text):
    body = json.dumps({
        "model": MODEL, "stream": False, "format": "json",
        "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
        "options": {"temperature": 0.7, "num_predict": 600},
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
        if not isinstance(pairs, list):
            pairs = []
        for p in pairs:
            if not isinstance(p, dict):
                continue                      # skip weird replies instead of crashing
            q, a = str(p.get("q", "")).strip(), str(p.get("a", "")).strip()
            if len(q.split()) >= 4 and 6 <= len(a.split()) <= 50 and a.lower() not in q.lower():
                out.write(json.dumps({"chunk": i, "q": q, "a": a}) + "\n")
                total += 1
        out.flush()
        if i % 10 == 0 and pairs:
            print(f"chunk {i}: {total} pairs so far | e.g. Q: {pairs[0].get('q')} A: {pairs[0].get('a')}")
print("done,", total, "new pairs")