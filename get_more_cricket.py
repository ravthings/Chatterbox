import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    user_agent="chatterbox-cricket/1.0 (github.com/ravthings)", language="en")

seeds = [
    "Cricket", "Test cricket", "One Day International", "Twenty20",
    "Cricket World Cup", "ICC Men's T20 World Cup", "The Ashes", "Indian Premier League",
    "Big Bash League", "Australia national cricket team", "India national cricket team",
    "England cricket team", "Women's cricket", "Glossary of cricket terms",
]

# 1. collect titles: the seeds + every linked page with "cricket" in the name
titles = set(seeds)
for s in seeds:
    for t in wiki.page(s).links:
        if ":" not in t and "cricket" in t.lower():
            titles.add(t)
titles = sorted(titles)[:800]
print(f"found {len(titles)} pages, downloading...")

# 2. download them (skip tiny stubs)
text, kept = "", 0
for i, t in enumerate(titles):
    try:
        p = wiki.page(t)
        if p.exists() and len(p.text) > 2000:
            text += p.text + "\n\n"
            kept += 1
    except Exception as e:
        print("skip", t, e)
    if i % 50 == 0:
        print(f"{i}/{len(titles)}  kept {kept}  ({len(text):,} chars)")

with open("cricket.txt", "w", encoding="utf-8") as f:
    f.write(text)
print(f"\nDONE: {kept} pages, {len(text):,} characters")
