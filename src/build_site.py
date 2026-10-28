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

import numpy as np
import pandas as pd

from common import OUTPUTS, ROOT, kickoff_utc, load_games, upcoming_week

DOCS = ROOT / "docs"
PICKS_DIR = OUTPUTS / "my_picks" / "data" / "users" / "me"


def model_line(p_home):
    """Home-team point spread implied by the win probability (same 13.45-pt scale used for Vegas lines),
    rounded to the half point. Positive = home favored, matching nflverse's spread_line."""
    from scipy.stats import norm
    p = min(max(float(p_home), 0.01), 0.99)
    return round(13.45 * norm.ppf(p) * 2) / 2


def load_scores(games):
    s = {r.game_id: (int(r.away_score), int(r.home_score)) for r in games[games["home_score"].notna()].itertuples()}
    manual = ROOT / "data" / "manual_scores.csv"
    if manual.exists():
        for r in pd.read_csv(manual).itertuples():
            s.setdefault(r.game_id, (int(r.away_score), int(r.home_score)))
    return s


def rankings(games, season):
    """Power rankings from the Kalman ratings: current (pregame for each team's next game),
    movement vs a week earlier, record, unit ranks, starting QB and playoff odds."""
    import kalman
    from pick_sheet import TEAM_NAMES
    tg = pd.read_csv(ROOT / "data" / "processed" / "team_games.csv", parse_dates=["gameday"])
    kf = kalman.ratings(games, tg).merge(games[["game_id", "season", "week", "gameday"]], on="game_id")
    kf = kf[kf["season"] == season]
    reg = games[(games["season"] == season) & (games["game_type"] == "REG")]
    done = reg[reg["home_score"].notna()]
    upcoming = reg[reg["home_score"].isna()]
    odds_f = DOCS / "data" / "playoff_odds.json"
    odds = {t["team"]: t for t in json.loads(odds_f.read_text())["teams"]} if odds_f.exists() else {}
    rows = []
    for team, d in kf.sort_values("gameday").groupby("team"):
        nxt = d[d["game_id"].isin(upcoming["game_id"])]
        cur = nxt.iloc[0] if len(nxt) else d.iloc[-1]
        before = d[d["gameday"] < cur["gameday"] - pd.Timedelta(days=4)]
        prev = before.iloc[-1] if len(before) else None
        w = l = t = 0
        for g in done[(done["home_team"] == team) | (done["away_team"] == team)].itertuples():
            m = (g.home_score - g.away_score) * (1 if g.home_team == team else -1)
            w, l, t = w + (m > 0), l + (m < 0), t + (m == 0)
        last = games[((games["home_team"] == team) | (games["away_team"] == team)) & (games["season"] == season)]
        last = last[last["gameday"] <= cur["gameday"]].iloc[-1]
        qb = last["home_qb_name"] if last["home_team"] == team else last["away_qb_name"]
        o = odds.get(team, {})
        rows.append({"team": team, "name": TEAM_NAMES.get(team, team), "record": f"{w}-{l}" + (f"-{t}" if t else ""),
                     "rating": round(float(cur["kf_margin"]), 1), "sd": round(float(cur["kf_margin_sd"]), 1),
                     "move": None if prev is None else round(float(cur["kf_margin"] - prev["kf_margin"]), 1),
                     "off": float(cur["kf_off"]), "def": float(cur["kf_def"]),
                     "qb": qb if isinstance(qb, str) else "",
                     "playoff": o.get("playoff"), "division_title": o.get("division_title"),
                     "top_seed": o.get("top_seed"), "proj_wins": o.get("proj_wins")})
    rows.sort(key=lambda r: -r["rating"])
    for k, key in (("off_rank", "off"), ("def_rank", "def")):
        for i, r in enumerate(sorted(rows, key=lambda r: -r[key]), 1):
            r[k] = i
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        del r["off"], r["def"]
    return rows


def recap(week_before, season, graded):
    """Data-only review of the most recent week with graded live predictions."""
    import live_log
    log = pd.read_csv(live_log.LOG)
    log = log[(log["season"] == season) & (log["week"] <= week_before) & log["home_won"].notna()]
    if log.empty:
        return None
    wk = int(log["week"].max())
    d = log[log["week"] == wk].copy()
    d = d[d["home_won"] != 0.5]
    d["winner"] = np.where(d["home_won"] == 1, d["home_team"], d["away_team"])
    d["loser"] = np.where(d["home_won"] == 1, d["away_team"], d["home_team"])
    d["p_winner"] = np.where(d["home_won"] == 1, d["p_home"], 1 - d["p_home"])
    d["v_winner"] = np.where(d["home_won"] == 1, d["vegas_home"], 1 - d["vegas_home"])
    d["score"] = [f"{max(a, h):.0f}-{min(a, h):.0f}" for a, h in zip(d["away_score"], d["home_score"])]
    right = d[d["p_winner"] > 0.5]
    wrong = d[d["p_winner"] < 0.5]
    up = d.loc[d["v_winner"].idxmin()] if d["v_winner"].notna().any() else d.loc[d["p_winner"].idxmin()]
    out = {"week": wk, "games": int(len(d)),
           "model": [int((d["p_winner"] > 0.5).sum()), int((d["p_winner"] < 0.5).sum())],
           "vegas": [int((d["v_winner"] > 0.5).sum()), int((d["v_winner"] < 0.5).sum())],
           "upset": {"winner": up["winner"], "loser": up["loser"], "score": up["score"],
                     "vegas": float(up["v_winner"]) if not pd.isna(up["v_winner"]) else None, "model": float(up["p_winner"])}}
    if len(right):
        b = right.loc[right["p_winner"].idxmax()]
        out["best"] = {"winner": b["winner"], "loser": b["loser"], "score": b["score"], "model": float(b["p_winner"])}
    wrong = wrong[wrong["winner"] != up["winner"]]  # don't repeat the upset as the worst miss
    if len(wrong):
        w = wrong.loc[wrong["p_winner"].idxmin()]
        out["worst"] = {"winner": w["winner"], "loser": w["loser"], "score": w["score"], "model": float(1 - w["p_winner"])}
    mine = [g for g in graded if g["week"] == wk and g["season"] == season and g["y"] != 0.5 and abs(g["you"] - 0.5) > 0.004]
    if mine:
        out["krish"] = [sum((g["you"] > 0.5) == (g["y"] == 1) for g in mine), sum((g["you"] > 0.5) != (g["y"] == 1) for g in mine)]
    return out


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
                "id": r.game_id, "game": f"{r.away_team} @ {r.home_team}", "away": r.away_team, "home": r.home_team,
                "kickoff": k.isoformat() if k is not None else None,
                "pick": r.pick, "confidence": round(float(r.confidence), 4),
                "p_home": round(float(r.p_home), 4),
                "vegas_home": None if pd.isna(r.vegas_prob) else round(float(r.vegas_prob), 4),
                "final": scores.get(r.game_id),
                "model_line": model_line(r.p_home),
                "vegas_line": None if pd.isna(r.spread_line) else float(r.spread_line),
                "why": json.loads(r.why) if isinstance(getattr(r, "why", None), str) else [],
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

    # for grading website picks in the browser: this season's finals and the model's frozen probabilities
    season_scores = {gid: list(v) for gid, v in scores.items() if gid.startswith(f"{season}_")}
    model_p = {}
    for mf in sorted(OUTPUTS.glob(f"picks_{season}_wk*.csv")):
        model_p.update(pd.read_csv(mf).set_index("game_id")["p_home"].round(4).to_dict())

    data = {"season": season, "week": week, "scores": season_scores, "model_p": model_p, "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "this_week": this_week, "graded": graded, "brief": brief(), "live": live,
            "recap": recap(week - 1, season, graded)}
    DOCS.mkdir(exist_ok=True)
    tpl = (ROOT / "site" / "template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", json.dumps(data))
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    import shutil
    for asset in ("style.css", "og.png", "favicon.svg", "picks.js"):
        shutil.copy(ROOT / "site" / asset, DOCS / asset)
    rk = {"season": season, "week": week, "built": data["built"], "teams": rankings(games, season)}
    rtpl = (ROOT / "site" / "rankings.html").read_text(encoding="utf-8")
    (DOCS / "rankings.html").write_text(rtpl.replace("/*__DATA__*/null", json.dumps(rk)), encoding="utf-8")
    (DOCS / ".nojekyll").write_text("")
    print(f"site: week {week} ({len(this_week)} games), {len(graded)} graded picks -> docs/index.html")


if __name__ == "__main__":
    main()
