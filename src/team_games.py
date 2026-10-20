"""Per-team, per-game box stats from play-by-play, plus per-QB game logs.

Writes:
    data/processed/team_games.csv  one row per (game, team) with that team's offense & defense stats
    data/processed/qb_games.csv    one row per (game, passer) with dropbacks / EPA / CPOE
"""
import numpy as np
import pandas as pd

from common import FIRST_SEASON, CURRENT_SEASON, PROCESSED, load_per_season, norm_team

PBP_COLS = [
    "game_id", "season", "week", "posteam", "defteam", "epa", "success", "pass", "rush",
    "qb_dropback", "passer_player_id", "qb_epa", "cpoe", "interception", "fumble_lost", "sack",
    "yards_gained", "wp", "special_teams_play", "play_type", "down", "qb_scramble",
    "rusher_player_id",
]


def load_pbp():
    pbp = load_per_season("pbp", "play_by_play", range(FIRST_SEASON, CURRENT_SEASON + 1), usecols=PBP_COLS)
    for c in ("posteam", "defteam"):
        pbp[c] = pbp[c].map(norm_team)
    return pbp


def offense_stats(pbp):
    """Offensive stats per (game, posteam)."""
    sc = pbp[(pbp["pass"] == 1) | (pbp["rush"] == 1)].copy()
    sc = sc[sc["epa"].notna() & sc["posteam"].notna()]
    sc["is_pass"] = sc["pass"] == 1
    sc["explosive"] = ((sc["is_pass"] & (sc["yards_gained"] >= 20)) |
                       (~sc["is_pass"] & (sc["yards_gained"] >= 10))).astype(float)
    sc["turnover"] = sc["interception"].fillna(0) + sc["fumble_lost"].fillna(0)
    # "Neutral" plays: game still competitive, strips most garbage time
    sc["neutral"] = sc["wp"].between(0.1, 0.9)
    sc["early_down"] = sc["down"].isin([1, 2])

    def agg(d):
        p = d[d.is_pass]
        r = d[~d.is_pass]
        n = d[d.neutral]
        ed = d[d.early_down & d.neutral]
        return pd.Series({
            "plays": len(d),
            "epa": d.epa.mean(),
            "sr": d.success.mean(),
            "pass_epa": p.epa.mean() if len(p) else np.nan,
            "rush_epa": r.epa.mean() if len(r) else np.nan,
            "pass_sr": p.success.mean() if len(p) else np.nan,
            "rush_sr": r.success.mean() if len(r) else np.nan,
            "neutral_epa": n.epa.mean() if len(n) else d.epa.mean(),
            "explosive": d.explosive.mean(),
            "turnovers": d.turnover.sum(),
            "sack_rate": p.sack.fillna(0).mean() if len(p) else np.nan,
            "early_pass_rate": ed.is_pass.mean() if len(ed) else np.nan,
        })

    return sc.groupby(["game_id", "posteam"]).apply(agg, include_groups=False).reset_index()


def special_teams(pbp):
    st = pbp[(pbp["special_teams_play"] == 1) & pbp["epa"].notna() & pbp["posteam"].notna()]
    a = st.groupby(["game_id", "posteam"])["epa"].sum().rename("st_epa_pos").reset_index()
    return a


def qb_games(pbp):
    d = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna() & pbp["qb_epa"].notna()]
    q = d.groupby(["game_id", "season", "week", "posteam", "passer_player_id"]).agg(
        dropbacks=("qb_epa", "size"), qb_epa=("qb_epa", "sum"), cpoe_sum=("cpoe", "sum"),
        cpoe_n=("cpoe", "count")).reset_index()
    # Designed QB runs (not scrambles) also carry QB value
    runs = pbp[(pbp["rush"] == 1) & (pbp["qb_scramble"] != 1) & pbp["rusher_player_id"].notna() & pbp["epa"].notna()]
    runs = runs.groupby(["game_id", "rusher_player_id"]).agg(qb_runs=("epa", "size"), qb_run_epa=("epa", "sum")).reset_index()
    runs = runs.rename(columns={"rusher_player_id": "passer_player_id"})
    q = q.merge(runs, on=["game_id", "passer_player_id"], how="left").fillna({"qb_runs": 0, "qb_run_epa": 0})
    return q.rename(columns={"passer_player_id": "qb_id", "posteam": "team"})


def build(games):
    pbp = load_pbp()
    off = offense_stats(pbp)
    st = special_teams(pbp)

    g = games[games["home_score"].notna()][["game_id", "season", "week", "gameday", "game_type",
                                            "home_team", "away_team", "home_score", "away_score"]]
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        t = g.rename(columns={f"{side}_team": "team", f"{opp}_team": "opp",
                              f"{side}_score": "pts_for", f"{opp}_score": "pts_against"})
        t = t[["game_id", "season", "week", "gameday", "game_type", "team", "opp", "pts_for", "pts_against"]].copy()
        t["is_home"] = int(side == "home")
        rows.append(t)
    tg = pd.concat(rows, ignore_index=True)

    o = off.rename(columns={"posteam": "team"})
    o = o.rename(columns={c: f"off_{c}" for c in o.columns if c not in ("game_id", "team")})
    d = off.rename(columns={"posteam": "opp"})
    d = d.rename(columns={c: f"def_{c}" for c in d.columns if c not in ("game_id", "opp")})
    tg = tg.merge(o, on=["game_id", "team"], how="left").merge(d, on=["game_id", "opp"], how="left")

    s_for = st.rename(columns={"posteam": "team", "st_epa_pos": "st_for"})
    s_against = st.rename(columns={"posteam": "opp", "st_epa_pos": "st_against"})
    tg = tg.merge(s_for, on=["game_id", "team"], how="left").merge(s_against, on=["game_id", "opp"], how="left")
    tg["st_epa"] = tg["st_for"].fillna(0) - tg["st_against"].fillna(0)
    tg = tg.drop(columns=["st_for", "st_against"])

    tg["margin"] = tg["pts_for"] - tg["pts_against"]
    tg["win"] = np.sign(tg["margin"]) * 0.5 + 0.5
    tg["to_margin"] = tg["def_turnovers"] - tg["off_turnovers"]
    tg = tg.sort_values(["gameday", "game_id", "team"]).reset_index(drop=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    tg.to_csv(PROCESSED / "team_games.csv", index=False)
    qb = qb_games(pbp)
    qb.to_csv(PROCESSED / "qb_games.csv", index=False)
    print(f"team_games: {len(tg)} rows, qb_games: {len(qb)} rows")
    return tg, qb


if __name__ == "__main__":
    from common import load_games
    build(load_games())
