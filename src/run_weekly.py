"""One command to refresh everything and predict the next week.

    python src/run_weekly.py            # refresh data, rebuild features, predict next unplayed week
    python src/run_weekly.py --no-download

Best run after Friday injury reports (and again Sunday morning for late QB news).
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent


def step(script, *args):
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, str(SRC / script), *args], check=True, cwd=SRC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    args = ap.parse_args()
    if not args.no_download:
        step("download_data.py", "--refresh")
    step("team_games.py")
    step("players.py")
    step("build_features.py")
    extra = ["--season", str(args.season), "--week", str(args.week)] if args.season and args.week else []
    step("predict.py", *extra)
    step("pick_sheet.py", *extra)  # context for the Pick Sheet page (pushed there by Claude)


if __name__ == "__main__":
    main()
