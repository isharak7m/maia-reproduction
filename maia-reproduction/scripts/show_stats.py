import pathlib, json
ROOT = pathlib.Path(__file__).parent.parent
with open(ROOT / "data/parquet/dataset_stats.json") as f:
    stats = json.load(f)
print(f"Games scanned: {stats['games_scanned']:,}")
print(f"Time: {stats['elapsed']:.0f}s")
for k, v in sorted(stats["bins"].items(), key=lambda x: int(x[0])):
    print(f"  Bin {k}-{int(k)+99}: {v['games']:,} games, {v['moves']:,} moves")
