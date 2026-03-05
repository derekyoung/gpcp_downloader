#!/usr/bin/env python3
"""
download_gpcp.py
----------------
Recursively crawls the NOAA GPCP daily precipitation NetCDF archive and
downloads every .nc file to a local directory tree that mirrors the remote
structure.

Base URL:
  https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access/

Usage:
  python download_gpcp.py [--output-dir ./gpcp_data] [--workers 4]
                          [--start-year 1996] [--end-year 2025]
                          [--dry-run]

Options:
  --output-dir   Local directory to save files (default: ./gpcp_data)
  --workers      Number of parallel download threads (default: 4)
  --start-year   First year to download (default: 1996)
  --end-year     Last year to download (default: current year)
  --dry-run      Print URLs without downloading
"""

import argparse
import sys
import os
import re
import time
import datetime
import threading
from pathlib import Path
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    sys.exit(
        "ERROR: 'requests' is not installed.\n"
        "Install it with:  pip install requests"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access/"
NC_PATTERN = re.compile(r'href="([^"]+\.nc)"', re.IGNORECASE)
DIR_PATTERN = re.compile(r'href="(\d{4}/)"')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    """Return a requests Session with retry logic and a polite User-Agent."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": (
            "GPCP-downloader/1.0 "
            "(research use; contact https://scripps.ucsd.edu)"
        )
    })
    return session


_print_lock = threading.Lock()

def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Crawling
# ---------------------------------------------------------------------------

def fetch_nc_urls_for_year(session: requests.Session, year: int) -> list[str]:
    """
    Fetch the listing page for a given year and return absolute URLs of all
    .nc files found.
    """
    url = f"{BASE_URL}{year}/"
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log(f"  [WARN] Could not fetch year {year}: {exc}")
        return []

    html = resp.text
    nc_hrefs = NC_PATTERN.findall(html)

    urls = []
    for href in nc_hrefs:
        # hrefs may be absolute or relative
        if href.startswith("http"):
            abs_url = href
        else:
            abs_url = urljoin(url, href)
        urls.append(abs_url)

    return urls


def collect_all_nc_urls(
    session: requests.Session,
    start_year: int,
    end_year: int,
) -> list[str]:
    """Crawl each year's index page and collect all .nc file URLs."""
    all_urls: list[str] = []
    years = list(range(start_year, end_year + 1))
    log(f"Crawling index pages for {len(years)} year(s): {start_year}–{end_year}")

    for year in years:
        log(f"  Scanning {year}/ ...")
        urls = fetch_nc_urls_for_year(session, year)
        log(f"    Found {len(urls)} file(s)")
        all_urls.extend(urls)
        time.sleep(0.1)  # be polite

    return all_urls


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def local_path_for_url(url: str, output_dir: Path) -> Path:
    """
    Map a remote URL to a local file path, preserving the year sub-directory.
    Example:
      https://.../access/gpcp_v01r03_daily_d19961001_c20170530.nc
        -> <output_dir>/1996/gpcp_v01r03_daily_d19961001_c20170530.nc
    """
    parsed = urlparse(url)
    filename = Path(parsed.path).name

    # Extract year from filename, e.g. d19961001 -> 1996
    m = re.search(r"_d(\d{4})\d{4}_", filename)
    if m:
        year_dir = m.group(1)
    else:
        year_dir = "unknown"

    return output_dir / year_dir / filename


def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    dry_run: bool,
) -> tuple[str, str]:
    """
    Download a single file. Returns (url, status) where status is one of
    'skipped', 'downloaded', 'dry-run', or 'error: <msg>'.
    """
    if dry_run:
        return url, "dry-run"

    if dest.exists():
        # Simple size-based skip: if file exists and has size > 0, skip
        if dest.stat().st_size > 0:
            return url, "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    try:
        with session.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    f.write(chunk)
        tmp.rename(dest)
        return url, "downloaded"
    except Exception as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return url, f"error: {exc}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    current_year = datetime.date.today().year
    parser = argparse.ArgumentParser(
        description="Download all GPCP daily NetCDF files from NOAA."
    )
    parser.add_argument(
        "--output-dir",
        default="./gpcp_data",
        help="Local directory to save files (default: ./gpcp_data)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel download threads (default: 4, max recommended: 8)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1996,
        help="First year to download (default: 1996)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=current_year,
        help=f"Last year to download (default: {current_year})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be downloaded without actually downloading",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        log(f"Output directory : {output_dir}")
    else:
        log("=== DRY RUN — no files will be downloaded ===")

    log(f"Year range       : {args.start_year}–{args.end_year}")
    log(f"Parallel workers : {args.workers}")
    log("")

    session = make_session()

    # Step 1: collect all URLs
    all_urls = collect_all_nc_urls(session, args.start_year, args.end_year)
    log(f"\nTotal files found: {len(all_urls)}")

    if not all_urls:
        log("Nothing to download. Exiting.")
        return

    if args.dry_run:
        log("\nFiles that would be downloaded:")
        for url in all_urls:
            dest = local_path_for_url(url, output_dir)
            log(f"  {url}  ->  {dest}")
        return

    # Step 2: download in parallel
    log(f"\nStarting download with {args.workers} worker(s)...\n")
    counters = {"downloaded": 0, "skipped": 0, "error": 0}

    def task(url: str) -> tuple[str, str]:
        dest = local_path_for_url(url, output_dir)
        return download_file(session, url, dest, dry_run=False)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(task, url): url for url in all_urls}
        for i, future in enumerate(as_completed(futures), start=1):
            url, status = future.result()
            filename = Path(urlparse(url).path).name
            prefix = f"[{i:>5}/{len(all_urls)}]"
            if status == "downloaded":
                counters["downloaded"] += 1
                log(f"{prefix} ✓ {filename}")
            elif status == "skipped":
                counters["skipped"] += 1
                log(f"{prefix} — {filename} (already exists)")
            else:
                counters["error"] += 1
                log(f"{prefix} ✗ {filename} ({status})")

    log(f"\nDone!")
    log(f"  Downloaded : {counters['downloaded']}")
    log(f"  Skipped    : {counters['skipped']}  (already existed)")
    log(f"  Errors     : {counters['error']}")


if __name__ == "__main__":
    main()
