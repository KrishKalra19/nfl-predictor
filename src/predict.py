"""Train the final lineup on every completed game and predict an upcoming week.

    python src/predict.py                     # next unplayed week
    python src/predict.py --season 2026 --week 6

Writes outputs/picks_<season>_wk<week>.csv and .json
"""
import argparse
import json

import numpy as np
import pandas as pd

from backtest import TRAIN_START, feature_sets
from common import OUTPUTS, kickoff_utc, load_games, upcoming_week
from config import LINEUP
from dataset import load_features
from evaluate import logit_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    args = ap.parse_args()

    df = load_features()
    sched = load_games()
    season, week = (args.season, args.week) if args.season else upcoming_week(sched)
    fsets = feature_sets(df)
    games = df[(df["season"] == season) & (df["week"] == week)].copy()
    if games.empty:
        raise SystemExit(f"no games for {season} week {week}")
    # Train only on weeks before this one, so a mid-week rerun never learns from this week's results.
    train = df[df["target"].notna() & (df["season"] >= TRAIN_START) & (df["gameday"] < games["gameday"].min())]

    fitted = {}
    for name, (factory, fs) in LINEUP.items():
        fitted[name] = factory().fit(train, fsets[fs])
        games[name] = fitted[name].predict(games)
        print(f"trained {name}", flush=True)
    from explain import explain
    games["why"] = explain(fitted["logreg"], games)
    games["p_home"] = logit_mean([games[n] for n in LINEUP])

    out = games[["game_id", "gameday", "away_team", "home_team", "p_home", "vegas_prob", "spread_line",
                 "h_qb_rating", "a_qb_rating", "h_qb_changed", "a_qb_changed",
                 "h_rs_missing_off", "a_rs_missing_off", "h_rs_missing_def", "a_rs_missing_def",
                 "h_pr_margin", "a_pr_margin", "why"] + list(LINEUP)].copy()
    out["pick"] = np.where(out["p_home"] >= 0.5, out["home_team"], out["away_team"])
    out["confidence"] = np.maximum(out["p_home"], 1 - out["p_home"])
    out["vegas_pick"] = np.where(out["vegas_prob"] >= 0.5, out["home_team"], out["away_team"])
    out["agrees_with_vegas"] = out["pick"] == out["vegas_pick"]
    out["edge_vs_vegas"] = out["p_home"] - out["vegas_prob"]
    out["gameday"] = out["gameday"].dt.strftime("%Y-%m-%d")
    out = out.sort_values("confidence", ascending=False)

    OUTPUTS.mkdir(exist_ok=True)
    stem = OUTPUTS / f"picks_{season}_wk{week:02d}"
    # Freeze at kickoff: a game that has started keeps the prediction made before it started.
    ko = {r.game_id: kickoff_utc(r.gameday, r.gametime) for r in sched[sched["game_id"].isin(out["game_id"])].itertuples()}
    now = pd.Timestamp.now(tz="UTC")
    started = [gid for gid in out["game_id"] if ko.get(gid) is not None and ko[gid] <= now]
    prev_f = f"{stem}.csv"
    import os
    if started and os.path.exists(prev_f):
        prev = pd.read_csv(prev_f)
        keep = prev[prev["game_id"].isin(started)]
        out = pd.concat([out[~out["game_id"].isin(keep["game_id"])], keep], ignore_index=True)
        out = out.sort_values("confidence", ascending=False)
        print(f"kept pre-kickoff predictions for {len(keep)} started games")
    out.to_csv(f"{stem}.csv", index=False)
    with open(f"{stem}.json", "w") as f:
        json.dump({"season": season, "week": week, "trained_on_games": int(len(train)),
                   "games": out.round(4).to_dict(orient="records")}, f, indent=2, default=str)

    print(f"\n{season} week {week} picks (sorted by confidence)")
    for r in out.itertuples():
        v = f"{r.vegas_prob:.0%}" if not pd.isna(r.vegas_prob) else "n/a"
        print(f"  {r.away_team:>3} @ {r.home_team:<3}  pick {r.pick:<3} {r.confidence:5.1%}   "
              f"(P home {r.p_home:.0%}, Vegas home {v})")


if __name__ == "__main__":
    main()
