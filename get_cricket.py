# pip install wikipedia-api
import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    user_agent="chatterbox-cricket/1.0 (github.com/ravthings)",
    language="en",
)

pages = [
    # the game
    "Cricket", "History of cricket", "Laws of cricket", "Test cricket",
    "One Day International", "Twenty20", "Cricket World Cup",
    "ICC Men's T20 World Cup", "The Ashes", "Indian Premier League", "Big Bash League",
    # how it's played
    "Batting (cricket)", "Bowling (cricket)", "Fielding (cricket)", "Wicket-keeper",
    "Dismissal (cricket)", "Leg before wicket", "Run (cricket)", "Over (cricket)",
    "Duckworth–Lewis–Stern method", "Cricket pitch", "Cricket bat", "Cricket ball",
    "Fast bowling", "Spin bowling", "Swing bowling", "Leg spin", "Off spin",
    "Yorker", "Googly", "Bouncer (cricket)",
    # teams
    "Australia national cricket team", "India national cricket team", "England cricket team",
    "Pakistan national cricket team", "New Zealand national cricket team",
    "South Africa national cricket team", "Sri Lanka national cricket team",
    "West Indies cricket team", "Australia women's national cricket team",
    "Women's cricket", "Women's Cricket World Cup",
    # players
    "Don Bradman", "Sachin Tendulkar", "Virat Kohli", "MS Dhoni", "Shane Warne",
    "Ricky Ponting", "Steve Smith (cricketer)", "Pat Cummins", "Ben Stokes",
    "Glenn McGrath", "Muttiah Muralitharan", "Brian Lara", "Jacques Kallis",
    "Ellyse Perry",
    # grounds
    "Melbourne Cricket Ground", "Lord's", "Eden Gardens",
]

text, found, missing = "", 0, []
for title in pages:
    page = wiki.page(title)
    if page.exists():
        text += page.text + "\n\n"
        found += 1
        print(f"✓ {title}: {len(page.text):,} chars")
    else:
        missing.append(title)
        print(f"✗ NOT FOUND: {title}")

with open("cricket.txt", "w", encoding="utf-8") as f:
    f.write(text)

print(f"\n{found}/{len(pages)} pages, {len(text):,} characters saved to cricket.txt")
if missing:
    print("missing:", missing)
