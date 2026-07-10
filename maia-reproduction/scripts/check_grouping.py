"""Check if records from same game_id are consecutive in JSON."""
import json
from collections import defaultdict

with open("data/parquet/records_1100.json") as f:
    recs = json.load(f)

# Check first 10 records
print("First 5 full game_ids:")
for r in recs[:5]:
    print(f"  game_id={r['game_id']}  ply={r['ply']}")

# Check grouping in first 1000 records
g = defaultdict(list)
for r in recs[:1000]:
    g[r["game_id"]].append(r["ply"])

print("\nSample games (first 3):")
for gid, plies in list(g.items())[:3]:
    print(f"  game {gid[:20]}... plies: {len(plies)} moves, range {min(plies)}-{max(plies)}")
    # Check if consecutive
    expected = list(range(min(plies), max(plies) + 1))
    if plies == expected:
        print(f"    -> Consecutive, no gaps")
    else:
        gaps = set(expected) - set(plies)
        print(f"    -> Gaps at plies: {sorted(gaps)[:5]}...")

# Count games vs total
all_games = defaultdict(list)
for r in recs:
    all_games[r["game_id"]].append(r["ply"])

print(f"\nTotal records: {len(recs):,}")
print(f"Total games: {len(all_games):,}")
print(f"Avg moves/game: {len(recs)/len(all_games):.1f}")
print(f"Games with < 10 plies: {sum(1 for v in all_games.values() if len(v) < 10)}")
