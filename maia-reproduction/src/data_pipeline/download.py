"""Download and decompress Lichess PGN dumps."""

import logging
import os
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import zstandard

logger = logging.getLogger(__name__)

LICHESS_BASE_URL = "https://database.lichess.org/standard/"
PGN_PATTERN = "lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"


def download_pgn(
    year: int, month: int, dest_dir: str | Path, force: bool = False
) -> Path:
    """Download a single month's PGN dump from Lichess.

    Returns path to downloaded .zst file.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    filename = PGN_PATTERN.format(year=year, month=month)
    url = f"{LICHESS_BASE_URL}{filename}"
    dest_path = dest_dir / filename

    if dest_path.exists() and not force:
        logger.info(f"{filename} already exists, skipping")
        return dest_path

    logger.info(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"Downloaded {filename} ({dest_path.stat().st_size / 1e9:.1f} GB)")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise
    return dest_path


def decompress_stream(src_path: str | Path):
    """Stream-decompress a .zst file, yielding decompressed chunks.

    Yields bytes objects; this is intended to be used in a pipeline
    with the PGN parser so we never load the whole file into memory.
    """
    src_path = Path(src_path)
    logger.info(f"Decompressing {src_path.name}...")
    with open(src_path, "rb") as f:
        dctx = zstandard.ZstdDecompressor()
        reader = dctx.stream_reader(f)
        while True:
            chunk = reader.read(2 ** 20)  # 1 MB chunks
            if not chunk:
                break
            yield chunk


def decompress_to_text(src_path: str | Path, dest_path: str | Path | None = None) -> Path:
    """Fully decompress a .zst file to plain text .pgn (for testing)."""
    src_path = Path(src_path)
    if dest_path is None:
        dest_path = src_path.with_suffix("")  # remove .zst
    dest_path = Path(dest_path)

    if dest_path.exists():
        logger.info(f"{dest_path} already exists, skipping decompression")
        return dest_path

    logger.info(f"Decompressing {src_path.name} -> {dest_path.name}...")
    with open(src_path, "rb") as f_in:
        dctx = zstandard.ZstdDecompressor()
        with open(dest_path, "wb") as f_out:
            dctx.copy_stream(f_in, f_out)

    logger.info(f"Decompressed to {dest_path}")
    return dest_path


def download_month_range(
    months: list[tuple[int, int]],
    dest_dir: str | Path,
    max_workers: int = 2,
    force: bool = False,
) -> list[Path]:
    """Download multiple months of PGN dumps in parallel.

    Args:
        months: List of (year, month) tuples.
        dest_dir: Directory to save files.
        max_workers: Max parallel downloads.
        force: Re-download if exists.

    Returns list of downloaded file paths.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(download_pgn, year, month, dest_dir, force): (year, month)
            for year, month in months
        }
        for future in as_completed(futures):
            year, month = futures[future]
            try:
                path = future.result()
                paths.append(path)
            except Exception:
                logger.exception(f"Failed for {year}-{month:02d}")
    return paths
