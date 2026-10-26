"""Append-only log of live predictions: the model's frozen pre-kickoff number for every 2026 game
since launch, plus the result once known. This is the only truly out-of-sample record.

    python src/live_log.py

Reads outputs/picks_<season>_wk<NN>.csv (predict.py freezes rows at kickoff) and writes
docs/data/live_log.csv. A logged prediction is never changed; only result columns get filled in.
"""
import re

import numpy as np
import pandas as pd

from common import OUTPUTS, ROOT, kickoff_utc, load_games

LOG = ROOT / "docs" / "data" / "live_log.csv"  # under docs/ so it is published and committed with the site
LAUNCH = (2026, 6)  # first week predicted live

# Predictions known not to be genuinely pregame, with the reason (kept out of the record).
EXCLUDE = {
    "2026_06_SEA_DEN": "re-predicted after the game finished, before kickoff freezing existed",
}

COLS = ["season", "week", "game_id", "kickoff", "away_team", "home_team", "p_home", "vegas_home",
        "logged_at", "away_score", "home_score", "home_won"]


def load_scores(games):
    s = {r.game_id: (int(r.away_score), int(r.home_score)) for r in games[games["home_score"].notna()].itertuples()}
    manual = ROOT / "data" / "manual_scores.csv"
    if manual.exists():
        for r in pd.read_csv(manual).itertuples():
            s.setdefault(r.game_id, (int(r.away_score), int(r.home_score)))
    return s


def update():
    games = load_games()
    ko = {r.game_id: kickoff_utc(r.gameday, r.gametime) for r in games.itertuples()}
    scores = load_scores(games)
    now = pd.Timestamp.now(tz="UTC")
    log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=COLS)
    have = set(log["game_id"])

    new = []
    for f in sorted(OUTPUTS.glob("picks_*_wk*.csv")):
        m = re.match(r"picks_(\d{4})_wk(\d{2})\.csv", f.name)
        season, week = int(m.group(1)), int(m.group(2))
        if (season, week) < LAUNCH:
            continue
        for r in pd.read_csv(f).itertuples():
            k = ko.get(r.game_id)
            if r.game_id in have or r.game_id in EXCLUDE or k is None or k > now:
                continue  # only games that have kicked off: their row in the picks file is frozen
            new.append({"season": season, "week": week, "game_id": r.game_id, "kickoff": k.isoformat(),
                        "away_team": r.away_team, "home_team": r.home_team, "p_home": round(float(r.p_home), 4),
                        "vegas_home": None if pd.isna(r.vegas_prob) else round(float(r.vegas_prob), 4),
                        "logged_at": now.isoformat(timespec="seconds")})
    if new:
        log = pd.concat([log, pd.DataFrame(new)], ignore_index=True)

    for i, r in log.iterrows():
        if pd.isna(r["home_won"]) and r["game_id"] in scores:
            a, h = scores[r["game_id"]]
            log.loc[i, ["away_score", "home_score"]] = [a, h]
            log.loc[i, "home_won"] = 1.0 if h > a else 0.0 if h < a else 0.5

    log = log[COLS].sort_values(["kickoff", "game_id"]).reset_index(drop=True)
    log.to_csv(LOG, index=False)
    done = log[log["home_won"].notna()]
    print(f"live log: {len(log)} predictions ({len(new)} new), {len(done)} graded")
    return log


def summary(log=None):
    """Model vs Vegas on graded live games (same games for both)."""
    log = update() if log is None else log
    d = log[log["home_won"].notna() & log["vegas_home"].notna()]
    if d.empty:
        return None
    y = d["home_won"].astype(float).to_numpy()

    def score(p):
        p = np.clip(p.astype(float).to_numpy(), 1e-6, 1 - 1e-6)
        dec = y != 0.5
        return {"right": int(((p > 0.5) == (y == 1))[dec].sum()), "decided": int(dec.sum()),
                "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}
    return {"games": int(len(d)), "since": f"{LAUNCH[0]} week {LAUNCH[1]}",
            "model": score(d["p_home"]), "vegas": score(d["vegas_home"])}


if __name__ == "__main__":
    print(summary())
