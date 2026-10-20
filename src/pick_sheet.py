"""Build the weekly pick-sheet context (one JSON document per week) for the Pick Sheet page.

    python src/pick_sheet.py                    # next unplayed week (run predict.py first)
    python src/pick_sheet.py --season 2026 --week 6

Writes outputs/pick_sheets/<season>-<week>.json (the `weeks/<id>` document) and
outputs/pick_sheets/results-<season>.json (final scores, for scoring past picks).
Everything shown before a pick is pregame information. The model's and Vegas's numbers sit
under `reveal` and the page only shows them after a pick is saved, so they can't anchor it.
"""
import argparse
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import OUTPUTS, PROCESSED, upcoming_week, RAW, haversine_miles, load_games, load_per_season, norm_team, team_geo, venue_geo

TEAM_NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers", "CHI": "Bears",
    "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs", "LA": "Rams", "LAC": "Chargers",
    "LV": "Raiders", "MIA": "Dolphins", "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
    "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers",
    "TEN": "Titans", "WAS": "Commanders",
}
STATUS_WHY = {"RES": "Injured reserve", "INA": "Inactive", "CUT": "Released", "PUP": "PUP list",
              "SUS": "Suspended", "DEV": "Practice squad", "RET": "Retired", "EXE": "Exempt",
              "TRD": "Traded", "TRC": "Traded", "NWT": "Not with team", "RSN": "Reserve/non-football"}
ET = ZoneInfo("America/New_York")


def kickoff_label(row):
    t = str(row["gametime"]) if isinstance(row["gametime"], str) else "13:00"
    local = datetime.strptime(f"{row['gameday']:%Y-%m-%d} {t}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    hour = local.hour % 12 or 12
    return (local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            f"{local:%a} {local:%b} {local.day}, {hour}:{local:%M} {'PM' if local.hour >= 12 else 'AM'} ET")


def ranks(tg, season, before):
    """Offense / defense EPA-per-play rank (1 = best) using this season's games before a date."""
    d = tg[(tg["season"] == season) & (tg["gameday"] < before)]
    if d.empty:
        return {}
    agg = d.groupby("team").agg(off=("off_epa", "mean"), deff=("def_epa", "mean"),
                                pf=("pts_for", "mean"), pa=("pts_against", "mean"))
    agg["off_rank"] = agg["off"].rank(ascending=False, method="min").astype(int)
    agg["def_rank"] = agg["deff"].rank(ascending=True, method="min").astype(int)
    return agg.to_dict(orient="index")


def last_games(tg, team, before, n=5):
    d = tg[(tg["team"] == team) & (tg["gameday"] < before)].sort_values("gameday").tail(n)
    out = []
    for r in d.itertuples():
        res = "W" if r.margin > 0 else "L" if r.margin < 0 else "T"
        out.append({"season": int(r.season), "week": int(r.week), "opp": r.opp, "home": bool(r.is_home),
                    "pf": int(r.pts_for), "pa": int(r.pts_against), "result": res})
    return out[::-1]  # most recent first


def record(tg, team, season, before):
    d = tg[(tg["team"] == team) & (tg["season"] == season) & (tg["gameday"] < before) & (tg["game_type"] == "REG")]
    w, l, t = int((d["margin"] > 0).sum()), int((d["margin"] < 0).sum()), int((d["margin"] == 0).sum())
    return f"{w}-{l}" + (f"-{t}" if t else "")


def qb_info(qbg, games, team, qb_id, qb_name, before):
    dates = games.set_index("game_id")["gameday"]
    q = qbg.assign(gameday=qbg["game_id"].map(dates))
    q = q[q["gameday"] < before]
    mine = q[q["qb_id"] == qb_id].sort_values("gameday")
    recent = mine.tail(8)
    plays = recent["dropbacks"].sum() + recent["qb_runs"].sum()
    epa = (recent["qb_epa"].sum() + recent["qb_run_epa"].sum()) / plays if plays else None
    team_q = q[q["team"] == team].sort_values(["gameday", "dropbacks"])
    last = team_q.groupby("game_id").tail(1).sort_values("gameday").tail(1)
    last_id = last["qb_id"].iloc[0] if len(last) else None
    names = games.set_index("home_qb_id")["home_qb_name"].to_dict() | games.set_index("away_qb_id")["away_qb_name"].to_dict()
    return {"name": qb_name if isinstance(qb_name, str) else "TBD",
            "epa_per_play": None if epa is None else round(float(epa), 3),
            "recent_plays": int(plays),
            "changed": bool(last_id is not None and isinstance(qb_id, str) and qb_id != last_id),
            "last_starter": names.get(last_id) if last_id else None}


def roster_notes(season, week, team, before, games):
    """Regulars (>=50% snaps over the team's last 4 games) who are unavailable, plus injury designations."""
    snaps = load_per_season("snaps", "snap_counts", [season - 1, season],
                            usecols=["game_id", "player", "pfr_player_id", "position", "team", "offense_pct", "defense_pct"])
    snaps["team"] = snaps["team"].map(norm_team)
    dates = games.set_index("game_id")["gameday"]
    snaps["gameday"] = snaps["game_id"].map(dates)
    ts = snaps[(snaps["team"] == team) & (snaps["gameday"] < before)]
    last4 = ts.drop_duplicates("game_id").sort_values("gameday").tail(4)["game_id"]
    w = ts[ts["game_id"].isin(last4)]
    reg = w.groupby(["pfr_player_id", "player", "position"])[["offense_pct", "defense_pct"]].sum() / max(len(last4), 1)
    reg["share"] = reg[["offense_pct", "defense_pct"]].max(axis=1)
    reg = reg[reg["share"] >= 0.5].reset_index()
    ids = pd.read_csv(RAW / "players.csv", usecols=["gsis_id", "pfr_id"]).dropna().drop_duplicates("pfr_id")
    reg = reg.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="left")

    ro = load_per_season("rosters", "roster_weekly", [season], usecols=["week", "team", "gsis_id", "status", "full_name"])
    ro["team"] = ro["team"].map(norm_team)
    ro = ro[(ro["week"] == week)]
    on_team = ro[ro["team"] == team].set_index("gsis_id")["status"].to_dict()
    inj = load_per_season("injuries", "injuries", [season],
                          usecols=["week", "team", "gsis_id", "full_name", "position", "report_status", "report_primary_injury"])
    inj["team"] = inj["team"].map(norm_team)
    inj_w = inj[(inj["week"] == week) & (inj["team"] == team) & inj["report_status"].notna()]
    inj_by = inj_w.set_index("gsis_id")[["report_status", "report_primary_injury"]].to_dict(orient="index")

    missing, designated = [], []
    for r in reg.sort_values("share", ascending=False).itertuples():
        st = on_team.get(r.gsis_id)
        ij = inj_by.get(r.gsis_id)
        why = None
        if st is None:
            why = "No longer on roster"
        elif st not in ("ACT", "INA"):
            why = STATUS_WHY.get(st, st)
        elif ij and ij["report_status"] in ("Out", "Doubtful"):
            why = f"{ij['report_status']} ({ij['report_primary_injury']})" if isinstance(ij["report_primary_injury"], str) else ij["report_status"]
        if why:
            missing.append({"name": r.player, "pos": r.position, "snap_share": round(float(r.share), 2), "why": why})
    for gid, ij in inj_by.items():
        if ij["report_status"] == "Questionable":
            row = inj_w[inj_w["gsis_id"] == gid].iloc[0]
            is_reg = gid in set(reg["gsis_id"].dropna())
            designated.append({"name": row["full_name"], "pos": row["position"], "starter": bool(is_reg),
                               "injury": ij["report_primary_injury"] if isinstance(ij["report_primary_injury"], str) else ""})
    designated.sort(key=lambda d: not d["starter"])
    return missing, designated, int(len(inj_w)) > 0


def build(season=None, week=None):
    games = load_games()
    tg = pd.read_csv(PROCESSED / "team_games.csv", parse_dates=["gameday"])
    qbg = pd.read_csv(PROCESSED / "qb_games.csv")
    if season is None:
        season, week = upcoming_week(games)
    wk = games[(games["season"] == season) & (games["week"] == week)]
    picks_f = OUTPUTS / f"picks_{season}_wk{week:02d}.csv"
    picks = pd.read_csv(picks_f).set_index("game_id") if picks_f.exists() else None

    out_games, any_report = [], False
    for _, g in wk.sort_values(["gameday", "gametime"]).iterrows():
        before = g["gameday"]
        rk = ranks(tg, season, before)
        k_iso, k_label = kickoff_label(g)
        vlat, vlon, _ = venue_geo(g)
        teams = {}
        for side in ("away", "home"):
            t = g[f"{side}_team"]
            missing, designated, has_report = roster_notes(season, week, t, before, games)
            any_report |= has_report
            r = rk.get(t, {})
            teams[side] = {
                "abbr": t, "name": TEAM_NAMES.get(t, t), "record": record(tg, t, season, before),
                "last5": last_games(tg, t, before),
                "off_rank": r.get("off_rank"), "def_rank": r.get("def_rank"),
                "ppg": round(r["pf"], 1) if r else None, "papg": round(r["pa"], 1) if r else None,
                "rest_days": None if pd.isna(g[f"{side}_rest"]) else int(g[f"{side}_rest"]),
                "travel_miles": int(round(haversine_miles(team_geo(t, season)[:2], (vlat, vlon)))),
                "qb": qb_info(qbg, games, t, g[f"{side}_qb_id"], g[f"{side}_qb_name"], before),
                "missing": missing, "questionable": designated,
            }
        reveal = None
        if picks is not None and g["game_id"] in picks.index:
            p = picks.loc[g["game_id"]]
            reveal = {"model_home": round(float(p["p_home"]), 4),
                      "vegas_home": None if pd.isna(p["vegas_prob"]) else round(float(p["vegas_prob"]), 4),
                      "spread_home": None if pd.isna(p["spread_line"]) else float(p["spread_line"])}
        out_games.append({
            "game_id": g["game_id"], "kickoff": k_iso, "kickoff_label": k_label,
            "away": g["away_team"], "home": g["home_team"], "venue": g["stadium"] if isinstance(g["stadium"], str) else "",
            "roof": g["roof"] if isinstance(g["roof"], str) else "", "division_game": bool(g["div_game"] == 1),
            "neutral": g["location"] == "Neutral",
            "link": f"https://www.espn.com/nfl/game/_/gameId/{int(g['espn'])}" if not pd.isna(g["espn"]) else None,
            "teams": teams, "reveal": reveal,
        })

    doc = {"id": f"{season}-{week:02d}", "season": season, "week": week,
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "injury_report_posted": any_report, "games": out_games}
    res = games[(games["season"] == season) & games["home_score"].notna()]
    scores = {r.game_id: {"home": int(r.home_score), "away": int(r.away_score)} for r in res.itertuples()}
    # Stopgap finals for games nflverse hasn't posted yet (data/manual_scores.csv); nflverse wins once it has them.
    manual = RAW.parent / "manual_scores.csv"
    if manual.exists():
        for r in pd.read_csv(manual).itertuples():
            if r.game_id.startswith(str(season)) and r.game_id not in scores:
                scores[r.game_id] = {"home": int(r.home_score), "away": int(r.away_score)}
    results = {"season": season, "scores": scores}
    d = OUTPUTS / "pick_sheets"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{doc['id']}.json").write_text(json.dumps(doc, indent=1))
    (d / f"results-{season}.json").write_text(json.dumps(results))
    print(f"pick sheet {doc['id']}: {len(out_games)} games, injury report posted: {any_report}")
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    a = ap.parse_args()
    build(a.season, a.week)
