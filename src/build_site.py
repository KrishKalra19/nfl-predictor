"""Build the static GitHub Pages site in docs/: this week's model picks + Krish's graded picks.

    python src/build_site.py

Inputs:
    outputs/picks_<season>_wk<NN>.csv          model predictions (frozen at kickoff by predict.py)
    outputs/my_picks/data/users/me/picks-*.json  Krish's picks, exported from the Pick Sheet artifact
    data/raw/games.csv (+ data/manual_scores.csv) final scores
Krish's picks appear only for games that have kicked off. Notes are never published.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUTPUTS, ROOT, kickoff_utc, load_games, upcoming_week

DOCS = ROOT / "docs"
PICKS_DIR = OUTPUTS / "my_picks" / "data" / "users" / "me"


def load_scores(games):
    s = {r.game_id: (int(r.away_score), int(r.home_score)) for r in games[games["home_score"].notna()].itertuples()}
    manual = ROOT / "data" / "manual_scores.csv"
    if manual.exists():
        for r in pd.read_csv(manual).itertuples():
            s.setdefault(r.game_id, (int(r.away_score), int(r.home_score)))
    return s


def brief():
    """Backtest evidence for the technical section (walk-forward, held-out 2022-26)."""
    d = OUTPUTS / "dashboard"
    out = {}
    ms = pd.read_csv(d / "model_scores.csv")
    out["models"] = ms[ms["phase"] == "test"][["model", "kind", "games", "accuracy", "log_loss"]].to_dict(orient="records")
    ss = pd.read_csv(d / "season_scores.csv")
    out["seasons"] = ss[["season", "series", "accuracy", "phase"]].to_dict(orient="records")
    out["tiers"] = pd.read_csv(d / "confidence_tiers.csv")[["tier", "games", "accuracy", "avg_confidence"]].to_dict(orient="records")
    fv = pd.read_csv(d / "feature_value.csv")
    out["features"] = fv[["feature_group", "log_loss_increase"]].to_dict(orient="records")
    return out


def refresh_scores():
    """Pull the latest schedule/scores so finals show up without a full pipeline run."""
    import requests
    from common import RAW
    r = requests.get("https://github.com/nflverse/nfldata/raw/master/data/games.csv", timeout=60)
    if r.status_code == 200 and r.content.startswith(b"game_id"):
        (RAW / "games.csv").write_bytes(r.content)
    else:
        print(f"score refresh failed ({r.status_code}); using cached schedule")


def main():
    refresh_scores()
    import live_log
    live = live_log.summary(live_log.update())
    try:
        import playoffs
        playoffs.main()
    except Exception as e:  # the site must still build if the simulation fails
        print(f"playoff odds skipped: {e}")
    games = load_games()
    now = pd.Timestamp.now(tz="UTC")
    ko = {r.game_id: kickoff_utc(r.gameday, r.gametime) for r in games.itertuples()}
    scores = load_scores(games)
    season, week = upcoming_week(games)

    # this week's model picks
    f = OUTPUTS / f"picks_{season}_wk{week:02d}.csv"
    this_week = []
    if f.exists():
        p = pd.read_csv(f)
        for r in p.itertuples():
            k = ko.get(r.game_id)
            this_week.append({
                "game": f"{r.away_team} @ {r.home_team}", "away": r.away_team, "home": r.home_team,
                "kickoff": k.isoformat() if k is not None else None,
                "pick": r.pick, "confidence": round(float(r.confidence), 4),
                "p_home": round(float(r.p_home), 4),
                "vegas_home": None if pd.isna(r.vegas_prob) else round(float(r.vegas_prob), 4),
                "final": scores.get(r.game_id),
            })
        this_week.sort(key=lambda g: (g["kickoff"] or "", -g["confidence"]))

    # graded picks: every week Krish picked, games that kicked off and have a final
    graded = []
    for pf in sorted(PICKS_DIR.glob("picks-*.json")):
        wid = re.search(r"picks-(\d{4})-(\d{2})", pf.name)
        s, w = int(wid.group(1)), int(wid.group(2))
        mine = json.loads(pf.read_text()).get("picks", {})
        mf = OUTPUTS / f"picks_{s}_wk{w:02d}.csv"
        model = pd.read_csv(mf).set_index("game_id") if mf.exists() else None
        for gid, pk in mine.items():
            if gid not in scores or ko.get(gid) is None or ko[gid] > now:
                continue
            a, h = scores[gid]
            away, home = gid.split("_")[2], gid.split("_")[3]
            y = 1.0 if h > a else 0.0 if h < a else 0.5
            m = model.loc[gid] if model is not None and gid in model.index else None
            graded.append({
                "week": w, "season": s, "game": f"{away} @ {home}", "away": away, "home": home,
                "final": [a, h], "y": y, "you": float(pk["p_home"]), "tags": pk.get("tags", []),
                "model": None if m is None else round(float(m["p_home"]), 4),
                "vegas": None if m is None or pd.isna(m["vegas_prob"]) else round(float(m["vegas_prob"]), 4),
            })
    graded.sort(key=lambda g: (g["season"], g["week"], g["game"]))

    data = {"season": season, "week": week, "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "this_week": this_week, "graded": graded, "brief": brief(), "live": live}
    DOCS.mkdir(exist_ok=True)
    tpl = (ROOT / "site" / "template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", json.dumps(data))
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    (DOCS / ".nojekyll").write_text("")
    print(f"site: week {week} ({len(this_week)} games), {len(graded)} graded picks -> docs/index.html")


if __name__ == "__main__":
    main()
