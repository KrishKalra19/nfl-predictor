"""Player ratings and pregame roster strength.

Player value (as of kickoff, using only earlier games):
    share   = exponentially weighted snap share over the player's previous appearances
              (offense_pct for offensive players, defense_pct for defenders). Coaches give
              snaps to their best players, so sustained snap share is a robust quality signal.
    talent  = share * draft_value(pick)  -- draft capital as a talent prior.
    Rookies / players with no snap history get a prior share from draft round.

Team pregame roster strength = sums over the *game-day active roster* (weekly roster status
ACT, minus injury-report Out/Doubtful for upcoming games), per unit:
    QB, OL, RB, WR, TE  (offense share)   DL, LB, DB  (defense share)
plus "missing" value: snap share of players who played for the team in its previous 4 games
but are not active today (injury, inactive, released, traded).

Writes data/processed/roster_strength.csv, one row per (game_id, team).
"""
from collections import deque

import numpy as np
import pandas as pd

from common import FIRST_SEASON, CURRENT_SEASON, PROCESSED, RAW, load_per_season, norm_team

OFF_GROUPS = ["QB", "OL", "RB", "WR", "TE"]
DEF_GROUPS = ["DL", "LB", "DB"]
POS_GROUP = {
    "QB": "QB", "RB": "RB", "FB": "RB", "HB": "RB", "WR": "WR", "TE": "TE",
    "OL": "OL", "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL",
    "DL": "DL", "DE": "DL", "DT": "DL", "NT": "DL", "EDGE": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
    "DB": "DB", "CB": "DB", "S": "DB", "SS": "DB", "FS": "DB", "SAF": "DB",
}
SHARE_HALFLIFE = 6.0     # appearances
RECENT_TEAM_GAMES = 4    # window that defines a team's "regulars"


def draft_value(pick):
    """Relative talent prior from overall draft pick (undrafted ~ 0.25 of a top pick)."""
    pick = np.where(np.isnan(pick), 300, pick)
    return 0.25 + 0.75 * np.exp(-pick / 64.0)


def rookie_prior_share(pick):
    pick = np.where(np.isnan(pick), 300, pick)
    return np.clip(0.65 * np.exp(-pick / 70.0), 0.03, None)


def load_snaps():
    s = load_per_season("snaps", "snap_counts", range(FIRST_SEASON, CURRENT_SEASON + 1),
                        usecols=["game_id", "season", "week", "pfr_player_id", "team",
                                 "offense_pct", "defense_pct", "offense_snaps", "defense_snaps", "st_snaps"])
    ids = pd.read_csv(RAW / "players.csv", usecols=["gsis_id", "pfr_id"]).dropna().drop_duplicates("pfr_id")
    s = s.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    s["team"] = s["team"].map(norm_team)
    return s.drop(columns=["pfr_id", "pfr_player_id"])


def load_rosters():
    cols = ["season", "week", "team", "gsis_id", "status", "position", "depth_chart_position",
            "draft_number", "years_exp"]
    r = load_per_season("rosters", "roster_weekly", range(FIRST_SEASON, CURRENT_SEASON + 1), usecols=cols)
    r = r.dropna(subset=["gsis_id"])
    r["team"] = r["team"].map(norm_team)
    r["group"] = r["position"].map(POS_GROUP)
    fallback = r["depth_chart_position"].map(POS_GROUP)
    r["group"] = r["group"].fillna(fallback)
    return r.drop_duplicates(["season", "week", "team", "gsis_id"])


def load_injury_status(statuses):
    cols = ["season", "week", "team", "gsis_id", "report_status"]
    i = load_per_season("injuries", "injuries", range(FIRST_SEASON, CURRENT_SEASON + 1), usecols=cols)
    i["team"] = i["team"].map(norm_team)
    i = i[i["report_status"].isin(statuses)]
    return i[["season", "week", "team", "gsis_id"]].drop_duplicates()


def load_injury_outs():
    return load_injury_status(["Out", "Doubtful"])


def player_share_history(snaps, games):
    """For every player appearance: EW snap share *before* that game (pregame) and after it."""
    dates = games.set_index("game_id")["gameday"]
    s = snaps.copy()
    s["gameday"] = s["game_id"].map(dates)
    s = s.dropna(subset=["gameday"]).sort_values(["gsis_id", "gameday"])
    a = 1 - 0.5 ** (1 / SHARE_HALFLIFE)

    out_off, out_def, n_prev = [], [], []
    for _, grp in s.groupby("gsis_id", sort=False):
        eo = ed = None
        k = 0
        for o, d in zip(grp["offense_pct"].to_numpy(), grp["defense_pct"].to_numpy()):
            out_off.append(eo if eo is not None else np.nan)
            out_def.append(ed if ed is not None else np.nan)
            n_prev.append(k)
            eo = o if eo is None else (1 - a) * eo + a * o
            ed = d if ed is None else (1 - a) * ed + a * d
            k += 1
    s["pre_off"] = out_off
    s["pre_def"] = out_def
    s["n_prev"] = n_prev
    # post-game value = what a *future* game sees as this player's latest rating
    s["post_off"] = s.groupby("gsis_id")["pre_off"].shift(-1)
    s["post_def"] = s.groupby("gsis_id")["pre_def"].shift(-1)
    last = s.groupby("gsis_id").tail(1).index
    eo_last, ed_last = _final_ew(s, a)
    s.loc[last, "post_off"] = eo_last
    s.loc[last, "post_def"] = ed_last
    return s


def _final_ew(s, a):
    eo_l, ed_l = [], []
    for _, grp in s.groupby("gsis_id", sort=False):
        eo = ed = None
        for o, d in zip(grp["offense_pct"].to_numpy(), grp["defense_pct"].to_numpy()):
            eo = o if eo is None else (1 - a) * eo + a * o
            ed = d if ed is None else (1 - a) * ed + a * d
        eo_l.append(eo)
        ed_l.append(ed)
    return eo_l, ed_l


QUALITY_STATS = ["q_rec", "q_rush", "q_pressure", "q_cover"]
QUALITY_HALFLIFE = 8.0   # appearances
QUALITY_PRIOR_GAMES = 4.0


def player_production(games):
    """Per (game, player) production credited from play-by-play.

    q_rec: EPA on targets (receivers)       q_rush: EPA on carries (non-QB runners handled at team level)
    q_pressure: sacks + 0.5 * QB hits       q_cover: interceptions + 0.5 * passes defensed
    """
    from common import CURRENT_SEASON as CS
    cols = ["game_id", "epa", "pass", "rush", "receiver_player_id", "rusher_player_id", "sack_player_id",
            "qb_hit_1_player_id", "qb_hit_2_player_id", "interception_player_id",
            "pass_defense_1_player_id", "pass_defense_2_player_id"]
    pbp = load_per_season("pbp", "play_by_play", range(FIRST_SEASON, CS + 1), usecols=cols)
    parts = []
    t = pbp[(pbp["pass"] == 1) & pbp["receiver_player_id"].notna() & pbp["epa"].notna()]
    parts.append(t.groupby(["game_id", "receiver_player_id"])["epa"].sum().rename("q_rec").rename_axis(["game_id", "gsis_id"]))
    t = pbp[(pbp["rush"] == 1) & pbp["rusher_player_id"].notna() & pbp["epa"].notna()]
    parts.append(t.groupby(["game_id", "rusher_player_id"])["epa"].sum().rename("q_rush").rename_axis(["game_id", "gsis_id"]))
    ev = []
    for c, w in (("sack_player_id", 1.0), ("qb_hit_1_player_id", 0.5), ("qb_hit_2_player_id", 0.5)):
        ev.append(pbp[["game_id", c]].dropna().rename(columns={c: "gsis_id"}).assign(v=w))
    parts.append(pd.concat(ev).groupby(["game_id", "gsis_id"])["v"].sum().rename("q_pressure"))
    ev = []
    for c, w in (("interception_player_id", 1.0), ("pass_defense_1_player_id", 0.5), ("pass_defense_2_player_id", 0.5)):
        ev.append(pbp[["game_id", c]].dropna().rename(columns={c: "gsis_id"}).assign(v=w))
    parts.append(pd.concat(ev).groupby(["game_id", "gsis_id"])["v"].sum().rename("q_cover"))
    return pd.concat(parts, axis=1).fillna(0).reset_index()


def player_quality_history(snaps, games):
    """Per-appearance production, exponentially weighted; post-game values for as-of joins.

    Appearances come from snap counts, so games with zero production count as zeros.
    """
    dates = games.set_index("game_id")["gameday"]
    app = snaps[["game_id", "gsis_id"]].drop_duplicates()
    app = app.merge(player_production(games), on=["game_id", "gsis_id"], how="left").fillna(0)
    app["gameday"] = app["game_id"].map(dates)
    app = app.dropna(subset=["gameday"]).sort_values(["gsis_id", "gameday"]).reset_index(drop=True)
    a = 1 - 0.5 ** (1 / QUALITY_HALFLIFE)
    X = app[QUALITY_STATS].to_numpy()
    out = np.zeros_like(X)
    n_eff = np.zeros(len(app))
    for _, idx in app.groupby("gsis_id", sort=False).indices.items():
        s = np.zeros(X.shape[1]); n = 0.0
        for i in idx:
            s = (1 - a) * s + X[i]
            n = (1 - a) * n + 1
            out[i] = s
            n_eff[i] = n
    # shrunk per-appearance rate toward zero (replacement level)
    post = out / (n_eff + QUALITY_PRIOR_GAMES * a * 10)[:, None]
    q = pd.DataFrame(post, columns=[f"post_{c}" for c in QUALITY_STATS])
    q["gsis_id"] = app["gsis_id"]
    q["gameday"] = app["gameday"]
    return q.sort_values("gameday")


def team_game_index(games):
    """(season, week, team) -> game_id, gameday for both sides of every scheduled game."""
    rows = []
    for side in ("home", "away"):
        t = games[["game_id", "season", "week", "gameday", f"{side}_team"]].rename(columns={f"{side}_team": "team"})
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def build(games):
    snaps = load_snaps()
    hist = player_share_history(snaps, games)
    rosters = load_rosters()
    outs = load_injury_outs()
    tgi = team_game_index(games)

    r = rosters.merge(tgi, on=["season", "week", "team"], how="inner")

    # Latest rating strictly before this game's date: as-of join on each player's post-game values.
    ratings = hist[["gsis_id", "gameday", "post_off", "post_def", "n_prev"]].copy()
    ratings["n_games"] = ratings["n_prev"] + 1
    ratings = ratings.drop(columns="n_prev").sort_values("gameday")
    r = r.sort_values("gameday")
    r = pd.merge_asof(r, ratings, on="gameday", by="gsis_id", allow_exact_matches=False, direction="backward")
    r["n_games"] = r["n_games"].fillna(0)

    no_hist = r["post_off"].isna()
    prior = rookie_prior_share(r["draft_number"].to_numpy(dtype=float))
    is_off = r["group"].isin(OFF_GROUPS)
    r["share"] = np.where(is_off, r["post_off"], r["post_def"])
    r.loc[no_hist, "share"] = prior[no_hist.to_numpy()]
    # few appearances -> shrink toward the draft prior
    w = r["n_games"] / (r["n_games"] + 3.0)
    r["share"] = w * r["share"].fillna(0) + (1 - w) * prior
    r["talent"] = r["share"] * draft_value(r["draft_number"].to_numpy(dtype=float))

    r = r.merge(outs.assign(listed_out=1), on=["season", "week", "team", "gsis_id"], how="left")
    # Friday view, identical for history and live picks: on the 53-man roster (ACT, or INA where
    # game-day inactives are recorded) and not listed Out/Doubtful on the injury report.
    r["active"] = r["status"].isin(["ACT", "INA"]) & r["listed_out"].isna()

    agg = aggregate(r, snaps, games)
    act = r[r["active"] & r["group"].notna()]

    # Candidate (not adopted): game-day view = players who actually took a snap in this game. An upper
    # bound on what Sunday inactives could add; prefixed rsg_ and fenced out of the live model.
    played = snaps.loc[(snaps["offense_snaps"] + snaps["defense_snaps"] + snaps["st_snaps"]) > 0, ["game_id", "gsis_id"]]
    gd = r.merge(played.drop_duplicates().assign(took_snap=1), on=["game_id", "gsis_id"], how="left")
    has_snaps = gd["game_id"].isin(set(played["game_id"]))
    gd["active"] = np.where(has_snaps, gd["status"].isin(["ACT", "INA"]) & gd["took_snap"].notna(), gd["active"])
    g2 = aggregate(gd, snaps, games)
    g2.columns = [c.replace("rs_", "rsg_", 1) if c.startswith("rs_") else c for c in g2.columns]
    agg = agg.merge(g2, on=["game_id", "team"], how="left")

    # Candidate: the realistic game-day view = Friday view minus Questionable players who didn't play.
    # Matches what ESPN publishes ~90 minutes before kickoff (injured players flipped to Inactive); see gameday.py.
    q = load_injury_status(["Questionable"]).assign(questionable=1)
    gq = gd.merge(q, on=["season", "week", "team", "gsis_id"], how="left")
    sat = gq["questionable"].notna() & gq["took_snap"].isna() & has_snaps
    gq["active"] = r["active"].to_numpy() & ~sat.to_numpy()
    g3 = aggregate(gq, snaps, games)
    g3.columns = [c.replace("rs_", "rsq_", 1) if c.startswith("rs_") else c for c in g3.columns]
    agg = agg.merge(g3, on=["game_id", "team"], how="left")

    # Production quality of the available roster (expected per-game contribution)
    qh = player_quality_history(snaps, games)
    rq = pd.merge_asof(act[["game_id", "team", "gsis_id", "gameday", "group"]].sort_values("gameday"), qh,
                       on="gameday", by="gsis_id", allow_exact_matches=False, direction="backward").fillna(0)
    units = {"q_rec": ["WR", "TE", "RB"], "q_rush": ["RB"], "q_pressure": ["DL", "LB"], "q_cover": ["DB", "LB"]}
    for stat, groups in units.items():
        x = rq[rq["group"].isin(groups)].groupby(["game_id", "team"])[f"post_{stat}"].sum().rename(f"rs_{stat}")
        agg = agg.merge(x.reset_index(), on=["game_id", "team"], how="left")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    agg.to_csv(PROCESSED / "roster_strength.csv", index=False)
    print(f"roster_strength: {len(agg)} rows, {agg.shape[1]} cols")
    return agg


def aggregate(r, snaps, games):
    """Per (game, team) unit strength over rows of r flagged active, plus missing regulars."""
    act = r[r["active"] & r["group"].notna()]
    agg = act.pivot_table(index=["game_id", "team"], columns="group", values=["share", "talent"],
                          aggfunc="sum", fill_value=0)
    agg.columns = [f"rs_{v}_{g}" for v, g in agg.columns]
    agg = agg.reset_index()

    # Starter-level strength: top-11 players on each side by share
    def topk(d, groups, k=11):
        x = d[d["group"].isin(groups)].groupby(["game_id", "team"])["share"].apply(lambda v: v.nlargest(k).sum())
        return x
    agg = agg.merge(topk(act, OFF_GROUPS).rename("rs_off_top11").reset_index(), on=["game_id", "team"], how="left")
    agg = agg.merge(topk(act, DEF_GROUPS).rename("rs_def_top11").reset_index(), on=["game_id", "team"], how="left")
    tal = act.groupby(["game_id", "team", act["group"].isin(OFF_GROUPS)])["talent"].sum().unstack(fill_value=0)
    tal.columns = ["rs_def_talent" if not c else "rs_off_talent" for c in tal.columns]
    agg = agg.merge(tal.reset_index(), on=["game_id", "team"], how="left")
    exp = act.groupby(["game_id", "team"]).apply(
        lambda d: np.average(d["years_exp"].fillna(0), weights=d["share"] + 1e-6), include_groups=False)
    agg = agg.merge(exp.rename("rs_exp").reset_index(), on=["game_id", "team"], how="left")

    agg = agg.merge(missing_regulars(snaps, r, games), on=["game_id", "team"], how="left")
    return agg


def missing_regulars(snaps, r, games):
    """Snap share of a team's recent regulars who are NOT active for this game, split off/def."""
    dates = games.set_index("game_id")["gameday"]
    sn = snaps.copy()
    sn["gameday"] = sn["game_id"].map(dates)
    played = sn[(sn["offense_snaps"] + sn["defense_snaps"]) > 0]
    # regulars per (team, game): avg pct over the team's last N games
    team_games = played[["team", "game_id", "gameday"]].drop_duplicates().sort_values("gameday")
    by_game = {k: v for k, v in played.groupby(["team", "game_id"])}

    act_sets = r[r["active"]].groupby(["game_id", "team"])["gsis_id"].apply(set).to_dict()
    sched = team_game_index(games).sort_values("gameday")

    rows = []
    for team, tg in sched.groupby("team"):
        past = team_games[team_games["team"] == team]
        window = deque(maxlen=RECENT_TEAM_GAMES)
        pi = 0
        past_ids = past["game_id"].tolist()
        past_dates = past["gameday"].tolist()
        for gid, gday in zip(tg["game_id"], tg["gameday"]):
            while pi < len(past_ids) and past_dates[pi] < gday:
                window.append(by_game[(team, past_ids[pi])])
                pi += 1
            active = act_sets.get((gid, team))
            if not window or active is None:
                rows.append((gid, team, np.nan, np.nan))
                continue
            w = pd.concat(window)
            reg = w.groupby("gsis_id")[["offense_pct", "defense_pct"]].sum() / len(window)
            miss = reg[~reg.index.isin(active)]
            rows.append((gid, team, miss["offense_pct"].sum(), miss["defense_pct"].sum()))
    return pd.DataFrame(rows, columns=["game_id", "team", "rs_missing_off", "rs_missing_def"])


if __name__ == "__main__":
    from common import load_games
    build(load_games())
