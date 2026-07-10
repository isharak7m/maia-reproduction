#!/usr/bin/env python3
"""Download Lichess PGN dumps for specified date ranges.

Usage:
    python download_data.py --months 2016-01 2016-02 ... --dest data/pgn/
    python download_data.py --all --dest data/pgn/
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.download import download_month_range, PGN_PATTERN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("download_data")


def parse_months(months_str: list[str]) -> list[tuple[int, int]]:
    """Parse "YYYY-MM" strings into (year, month) tuples."""
    result = []
    for s in months_str:
        parts = s.split("-")
        if len(parts) == 2:
            result.append((int(parts[0]), int(parts[1])))
        else:
            logger.warning(f"Can't parse '{s}', expected YYYY-MM")
    return result


def main():
    parser = argparse.ArgumentParser(description="Download Lichess PGN dumps")
    parser.add_argument("--months", nargs="+", help="Months to download (YYYY-MM)")
    parser.add_argument("--start", help="Start month (YYYY-MM)")
    parser.add_argument("--end", help="End month (YYYY-MM)")
    parser.add_argument("--dest", default="data/pgn", help="Destination directory")
    parser.add_argument("--workers", type=int, default=2, help="Max parallel downloads")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")

    args = parser.parse_args()

    months = []
    if args.months:
        months = parse_months(args.months)
    elif args.start and args.end:
        start_year, start_month = map(int, args.start.split("-"))
        end_year, end_month = map(int, args.end.split("-"))
        y, m = start_year, start_month
        while (y, m) <= (end_year, end_month):
            months.append((y, m))
            m += 1
            if m > 12:
                m = 1
                y += 1
    else:
        # Default: download all months 2016-01 through 2019-12
        logger.info("No months specified; downloading 2016-01 through 2019-12")
        for year in range(2016, 2020):
            for month in range(1, 13):
                months.append((year, month))

    logger.info(f"Downloading {len(months)} months to {args.dest}")
    paths = download_month_range(months, args.dest, max_workers=args.workers, force=args.force)
    logger.info(f"Downloaded {len(paths)} files")

    total_gb = sum(p.stat().st_size for p in paths if p.exists()) / 1e9
    logger.info(f"Total size: {total_gb:.1f} GB")


if __name__ == "__main__":
    main()
