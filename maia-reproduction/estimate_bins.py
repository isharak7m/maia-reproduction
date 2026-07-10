import zstandard
import re
import time
import sys

ZST_PATH = "data/pgn/lichess_db_standard_rated_2019-10.pgn.zst"
MAX_GAMES = 500_000
MAX_TIME = 300  # 5 minutes
PRINT_INTERVAL = 100_000

# Rating bins
BINS = {
    "1100-1199": (1100, 1199),
    "1500-1599": (1500, 1599),
    "1900-1999": (1900, 1999),
}

def parse_headers(text):
    """Parse PGN header block into a dict using regex."""
    h = {}
    for m in re.finditer(r'\[(\w+)\s+"(.*?)"\]', text):
        h[m.group(1)] = m.group(2)
    return h

def rating_in_bin(rating_str, bin_lo, bin_hi):
    if not rating_str or rating_str == "?":
        return False
    try:
        r = int(rating_str)
        return bin_lo <= r <= bin_hi
    except ValueError:
        return False

def is_bullet(tc_str):
    """Return True if TimeControl initial time is under 180 seconds."""
    if not tc_str or tc_str == "-" or tc_str == "?":
        return False
    # Format examples: "180+2", "60+1", "300+0", "120"
    try:
        initial = tc_str.split("+")[0]
        return int(initial) < 180
    except (ValueError, IndexError):
        return False

def main():
    total = 0
    with_ratings = 0
    bullet = 0
    bin_counts = {name: 0 for name in BINS}

    start = time.time()
    last_print = 0

    print(f"Opening {ZST_PATH}...", file=sys.stderr)
    fh = open(ZST_PATH, "rb")
    dctx = zstandard.ZstdDecompressor()
    reader = dctx.stream_reader(fh)
    # Wrap for text reading
    text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")

    buffer = []
    in_headers = True
    headers_text = []

    for line in text_stream:
        elapsed = time.time() - start
        if total >= MAX_GAMES or elapsed >= MAX_TIME:
            break

        stripped = line.rstrip("\n\r")

        if in_headers:
            if stripped == "":
                # End of headers section; parse what we collected
                headers = parse_headers("".join(headers_text))
                total += 1
                we = headers.get("WhiteElo")
                be = headers.get("BlackElo")
                tc = headers.get("TimeControl")

                if we and be and we != "?" and be != "?":
                    with_ratings += 1
                    for name, (lo, hi) in BINS.items():
                        if rating_in_bin(we, lo, hi) and rating_in_bin(be, lo, hi):
                            bin_counts[name] += 1

                if tc and is_bullet(tc):
                    bullet += 1

                if total % PRINT_INTERVAL == 0:
                    dur = time.time() - start
                    print(f"[{total} games, {dur:.1f}s] {bin_counts}", file=sys.stderr)

                headers_text = []
                in_headers = False
            else:
                headers_text.append(line)
        else:
            if stripped == "":
                # Blank line after moves -> next game
                in_headers = True

    fh.close()
    dur = time.time() - start
    print(f"\n=== Results after {total} games ({dur:.1f}s) ===", file=sys.stderr)
    print(f"Total: {total}", file=sys.stderr)
    print(f"With both ratings: {with_ratings}", file=sys.stderr)
    print(f"Bullet: {bullet}", file=sys.stderr)
    for name in BINS:
        print(f"  {name}: {bin_counts[name]}", file=sys.stderr)

    # Print final machine-readable summary to stdout
    print(f"total_games={total}")
    print(f"with_both_ratings={with_ratings}")
    print(f"bullet={bullet}")
    for name in BINS:
        print(f"bin_{name}={bin_counts[name]}")

if __name__ == "__main__":
    import io
    main()
