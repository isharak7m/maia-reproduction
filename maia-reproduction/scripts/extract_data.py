#!/usr/bin/env python3
"""Two-phase extraction: fast header scan → parse only matching games."""

import json, logging, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import zstandard, chess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("extract")

BINS = {1100: (1100,1199), 1500: (1500,1599), 1900: (1900,1999)}
TARGET_GAMES = 25000
CLOCK_RE = re.compile(r"\{[^}]*%clk\s+(\d+):(\d+):(\d+)[^}]*\}")
EVAL_RE = re.compile(r"\{[^}]*%eval\s+([\d.\-#+]+)[^}]*\}")


def check_headers(headers_text: str) -> tuple:
    """Quick header check. Returns (bin_lower, is_bullet) or (None, False)."""
    elo_w = elo_b = None
    is_bullet_flag = False
    for line in headers_text.split("\n"):
        if line.startswith("[WhiteElo "):
            try: elo_w = int(line.split('"')[1])
            except: pass
        elif line.startswith("[BlackElo "):
            try: elo_b = int(line.split('"')[1])
            except: pass
        elif line.startswith("[TimeControl "):
            tc = line.split('"')[1]
            try:
                parts = tc.split("+")
                init = int(parts[0])
                inc = int(parts[1]) if len(parts) > 1 else 0
                if init + 40*inc < 180:
                    is_bullet_flag = True
            except: pass

    if elo_w is None or elo_b is None:
        return (None, is_bullet_flag)
    for lo, (lo2, hi) in BINS.items():
        if lo2 <= elo_w <= hi and lo2 <= elo_b <= hi:
            return (lo, is_bullet_flag)
    return (None, is_bullet_flag)


def parse_moves(pgn_text: str) -> list | None:
    """Parse moves from a PGN game text. Returns list of record dicts."""
    lines = pgn_text.split("\n")
    header_lines = []
    move_start = 0
    for i, line in enumerate(lines):
        if line.startswith("1."):
            move_start = i
            break
        if line.startswith("["):
            header_lines.append(line)

    headers = {}
    for h in header_lines:
        m = re.match(r'\[(\w+)\s+"(.*)"\]', h)
        if m: headers[m.group(1)] = m.group(2)

    white_elo = int(headers.get("WhiteElo", 0) or 0)
    black_elo = int(headers.get("BlackElo", 0) or 0)
    result = headers.get("Result", "*")
    tc_str = headers.get("TimeControl", "-")
    tc_initial = tc_increment = 0
    if tc_str and tc_str != "-":
        try:
            parts = tc_str.split("+")
            tc_initial, tc_increment = int(parts[0]), int(parts[1]) if len(parts)>1 else 0
        except: pass

    game_id = headers.get("Site", str(hash(pgn_text[:200])))

    # Strip annotations from move text
    move_text = "\n".join(lines[move_start:])
    move_text = re.sub(r'\{[^}]*\}', '', move_text)
    for tok in ("1-0","0-1","1/2-1/2","*"): move_text = move_text.replace(tok, "")
    # Remove move numbers
    move_text = re.sub(r'\b\d+\.\.\.\s*', ' ', move_text)
    move_text = re.sub(r'\b\d+\.\s*', ' ', move_text)
    san_list = [m for m in move_text.split() if m and len(m) <= 8]

    if not san_list or len(san_list) > 300:  # skip very long games
        return None

    # Extract clocks
    clock_map = {}
    for line in lines[move_start:]:
        cm = CLOCK_RE.search(line)
        if cm:
            h, m_, s = int(cm.group(1)), int(cm.group(2)), int(cm.group(3))
            clock_map[len(clock_map)] = h*3600 + m_*60 + s

    eval_map = {}
    for line in lines[move_start:]:
        em = EVAL_RE.search(line)
        if em:
            vs = em.group(1)
            if vs.startswith("#"):
                try: eval_map[len(eval_map)] = 10000 * (1 if int(vs[1:])>0 else -1)
                except: pass
            else:
                try: eval_map[len(eval_map)] = float(vs)
                except: pass

    board = chess.Board()
    records = []
    for ply, san in enumerate(san_list, 1):
        if ply <= 10:
            try: board.push_san(san)
            except: return None
            continue
        try: move = board.parse_san(san)
        except: return None

        clock = clock_map.get(ply-1)
        if clock is not None and clock < 30.0:
            board.push(move)
            continue

        records.append({
            "game_id": game_id, "fen": board.fen(), "move_uci": move.uci(),
            "move_san": san, "ply": ply,
            "white_rating": white_elo, "black_rating": black_elo,
            "side_to_move": "white" if board.turn == chess.WHITE else "black",
            "tc_category": "blitz",
            "tc_initial": tc_initial, "tc_increment": tc_increment,
            "clock_remaining": clock,
            "centipawn_eval": eval_map.get(ply-1),
            "result": result,
        })
        board.push(move)
    return records


def main():
    src = Path("data/pgn/lichess_db_standard_rated_2019-10.pgn.zst")
    if not src.exists():
        logger.error(f"{src} not found"); return

    out_dir = Path("data/parquet"); out_dir.mkdir(parents=True, exist_ok=True)
    records_by_bin = {k: [] for k in BINS}
    matched = {k: 0 for k in BINS}
    total_games = 0
    total_skipped = 0

    logger.info(f"Extracting from {src.name}")

    start = time.time()
    with open(src, "rb") as f:
        dctx = zstandard.ZstdDecompressor()
        reader = dctx.stream_reader(f)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")

        # Phase 1: scan headers for all games, record byte position of matches
        # Since we can't seek in a streaming decompressor, do a single pass:
        # scan headers quickly, and for matches, collect the game text.

        # Use a different approach: iterate through file, collect game text,
        # but do a cheap header check first before parsing moves.

        lines_buf = []
        in_headers = True
        game_buf = []
        rep_time = start

        for line in text_stream:
            if line.startswith("[Event "):
                # New game starting - process previous if we collected it
                if total_games > 0:
                    # Check if this was a matching game
                    header_text = "".join(game_buf)
                    bin_lower, is_bullet_flag = check_headers(header_text)
                    if bin_lower is not None and not is_bullet_flag:
                        parsed = parse_moves("".join(game_buf))
                        if parsed:
                            records_by_bin[bin_lower].extend(parsed)
                            matched[bin_lower] += 1
                    elif is_bullet_flag:
                        total_skipped += 1
                game_buf = []
                total_games += 1
            game_buf.append(line)

            if total_games % 100000 == 0:
                now = time.time()
                rate = total_games/(now-start)
                total_moves = sum(len(v) for v in records_by_bin.values())
                logger.info(
                    f"{total_games:,} games ({rate:.0f}/s) | matched: "
                    + " ".join(f"{k}:{matched[k]:,}g/{len(records_by_bin[k]):,}m"
                               for k in sorted(BINS))
                    + f" | skipped:{total_skipped:,} | {now-rep_time:.0f}s"
                )
                rep_time = now

            if all(matched[k] >= TARGET_GAMES for k in BINS):
                logger.info("All bins reached target!")
                break

        # Process last game
        if game_buf:
            header_text = "".join(game_buf)
            bin_lower, is_bullet_flag = check_headers(header_text)
            if bin_lower is not None and not is_bullet_flag:
                parsed = parse_moves("".join(game_buf))
                if parsed:
                    records_by_bin[bin_lower].extend(parsed)
                    matched[bin_lower] += 1

    elapsed = time.time() - start
    print(f"\n{'='*65}")
    print("DATASET STATS")
    print(f"{'='*65}")
    print(f"Source: {src.name} ({src.stat().st_size/1e9:.1f} GB)")
    print(f"Games scanned: {total_games:,} | Time: {elapsed:.0f}s ({total_games/elapsed:.0f}/s)")
    print(f"Bullet skipped: {total_skipped:,}")
    for b in sorted(records_by_bin):
        r = records_by_bin[b]
        print(f"Bin {b}-{b+99}: {matched[b]:,} games, {len(r):,} moves")
        if r:
            avg = len(r)/matched[b]
            print(f"  Avg moves/game (after ply 10): {avg:.0f}")
    print(f"{'='*65}")

    for bin_lower, recs in records_by_bin.items():
        out = out_dir / f"records_{bin_lower}.json"
        with open(out, "w") as f:
            json.dump(recs, f)
        logger.info(f"Saved {len(recs)} records -> {out}")

    stats = {"games_scanned": total_games, "elapsed": elapsed,
             "bins": {str(k): {"games": matched[k], "moves": len(records_by_bin[k])}
                       for k in sorted(records_by_bin)}}
    with open(out_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    import io
    main()
