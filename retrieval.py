import math, re
from collections import Counter, defaultdict

STOP = set("the a an of in on at to for is was were are who what which when where how did does do by with and or from that this as be has have had it its his her their can you me i by with and or from that this as be has have had it its his her their can you me i many much tell about".split())
           
           

def words(t):
    return [w for w in re.findall(r"[a-z0-9']+", t.lower()) if w not in STOP and len(w) > 1]

class Retriever:
    def __init__(self, path="cricket.txt", max_chars=450):
        # 1. cut cricket.txt into short passages (~450 characters, about 100 tokens)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 20]
        self.passages, cur = [], ""
        for s in sents:
            if len(cur) + len(s) > max_chars and cur:
                self.passages.append(cur.strip())
                cur = ""
            cur += s + " "
        if cur:
            self.passages.append(cur.strip())
            JUNK = ["isbn", "retrieved", "archived", "doi:", "cricinfo", "on facebook",
                    "on youtube", "external links", "further reading", "wisden india"]
            self.passages = [p for p in self.passages if not any(j in p.lower() for j in JUNK)]           

        # 2. build an index: word -> which passages contain it, and how often
        self.index = defaultdict(list)
        self.lengths = []
        for i, p in enumerate(self.passages):
            counts = Counter(words(p))
            self.lengths.append(sum(counts.values()))
            for w, c in counts.items():
                self.index[w].append((i, c))
        self.N = len(self.passages)
        self.avg = sum(self.lengths) / self.N

    def search(self, query, k=3, k1=1.5, b=0.75):
        scores = Counter()
        for w in set(words(query)):
            postings = self.index.get(w)
            if not postings:
                continue
            idf = math.log(1 + (self.N - len(postings) + 0.5) / (len(postings) + 0.5))   # rare words count more
            for i, tf in postings:
                L = self.lengths[i]
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * L / self.avg))
                # bonus: passages that START with the question's keywords are usually ABOUT it
        qw = set(words(query))
        for i in list(scores):
            head = set(words(self.passages[i][:80]))
            scores[i] += 2.0 * sum(1 for w in qw if w in head)
        return [(self.passages[i], s) for i, s in scores.most_common(k)]

if __name__ == "__main__":
    r = Retriever()
    print(f"{r.N:,} passages\n")
    for q in ["What is LBW?", "Who is Sachin Tendulkar?", "How many players are in a cricket team?",
              "Who won the 2019 Cricket World Cup?", "What is a googly?"]:
        best, score = r.search(q, k=1)[0]
        print(f"Q: {q}\n→ ({score:.1f}) {best[:300]}...\n")