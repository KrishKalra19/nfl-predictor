"""Download raw nflverse data into data/raw/.

Usage:
    python src/download_data.py            # all seasons 2014..current (skips files already cached)
    python src/download_data.py --refresh  # re-download the current season (run weekly)

2014 is pulled so 2015 has a prior season for preseason priors / player ratings.
"""
import argparse
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

FIRST_SEASON = 2014
CURRENT_SEASON = 2026

RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
SCHEDULES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"

# dataset name -> (release tag, filename pattern)
PER_SEASON = {
    "pbp": ("pbp", "play_by_play_{y}.csv.gz"),
    "rosters": ("weekly_rosters", "roster_weekly_{y}.csv.gz"),
    "snaps": ("snap_counts", "snap_counts_{y}.csv.gz"),
    "depth": ("depth_charts", "depth_charts_{y}.csv.gz"),
    "injuries": ("injuries", "injuries_{y}.csv.gz"),
}


def fetch(url: str, dest: Path, force: bool = False, quiet_missing: bool = True) -> bool:
    if dest.exists() and not force:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        if not quiet_missing:
            print(f"  missing ({r.status_code}): {url}")
        return False
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(r.content)
    tmp.replace(dest)
    print(f"  {dest.relative_to(ROOT)}  ({len(r.content) / 1e6:.1f} MB)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download current season files")
    args = ap.parse_args()

    print("schedules")
    fetch(SCHEDULES_URL, RAW / "games.csv", force=True)
    print("players (id map)")
    fetch(f"{RELEASE}/players/players.csv", RAW / "players.csv", force=args.refresh)

    for name, (tag, pattern) in PER_SEASON.items():
        print(name)
        for y in range(FIRST_SEASON, CURRENT_SEASON + 1):
            fname = pattern.format(y=y)
            force = args.refresh and y == CURRENT_SEASON
            if not fetch(f"{RELEASE}/{tag}/{fname}", RAW / name / fname, force=force):
                # some older seasons are only published as plain .csv
                plain = fname.removesuffix(".gz")
                fetch(f"{RELEASE}/{tag}/{plain}", RAW / name / plain, force=force, quiet_missing=False)


if __name__ == "__main__":
    sys.exit(main())
