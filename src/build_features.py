"""Assemble the game-level modeling table. Every feature uses only pre-kickoff information.

Writes data/processed/features.csv: one row per game (2015+ incl. unplayed games that have
rosters/QBs), with h_* (home team), a_* (away team), ctx_* (game context), target and Vegas benchmark.
"""
import math

import numpy as np
import pandas as pd

from common import FIRST_MODEL_SEASON, PROCESSED, RAW, haversine_miles, load_games, team_geo, venue_geo

ROLL_STATS = [
    "off_epa", "off_sr", "off_pass_epa", "off_rush_epa", "off_neutral_epa", "off_explosive",
    "off_turnovers", "off_sack_rate", "off_early_pass_rate", "off_pass_sr", "off_rush_sr",
    "def_epa", "def_sr", "def_pass_epa", "def_rush_epa", "def_neutral_epa", "def_explosive",
    "def_turnovers", "def_sack_rate", "def_pass_sr", "def_rush_sr",
    "st_epa", "margin", "win", "to_margin", "pts_for", "pts_against",
    "adj_off_epa", "adj_def_epa", "adj_margin",
]
# (name, half-life in games, weight multiplier per season boundary, pseudo-games of league-average prior)
WINDOWS = [("s", 2.0, 0.15, 0.5), ("l", 8.0, 0.55, 2.0)]

ELO_K, ELO_HFA, ELO_REVERT = 20.0, 40.0, 1 / 3


# ------------------------------------------------------------------ rolling team form

def pregame_means(rows, source, stats):
    """League average of `stats` over `source` games played strictly before each row's date
    (rows aligned to rows.index). Using a whole-dataset mean here would leak future seasons into
    early-season priors. Rows before any data fall back to the first day's average."""
    src = source[["gameday"] + stats].copy()
    by_day = src.groupby("gameday")[stats]
    sums, counts = by_day.sum(min_count=1).fillna(0).sort_index(), by_day.count().sort_index()
    cs, cc = sums.cumsum(), counts.cumsum()
    days = cs.index.to_numpy()
    pos = np.searchsorted(days, rows["gameday"].to_numpy(), side="left") - 1  # last day strictly before
    first = (sums.iloc[0] / counts.iloc[0].replace(0, np.nan)).to_numpy()
    out = np.empty((len(rows), len(stats)))
    valid = pos >= 0
    out[valid] = (cs.to_numpy()[pos[valid]] / np.where(cc.to_numpy()[pos[valid]] == 0, np.nan, cc.to_numpy()[pos[valid]]))
    out[~valid] = first
    return out


def weighted_history(tg, stats, halflife, carry, prior_n, means):
    """Pregame recency-weighted mean of `stats` for each team-game (uses strictly earlier games).
    `means` is the league-average prior: one row per tg row (pregame), or a single vector."""
    means = np.asarray(means, dtype=float)
    if means.ndim == 1:
        means = np.tile(means, (len(tg), 1))
    out = np.full((len(tg), len(stats)), np.nan)
    tg = tg.sort_values(["team", "gameday"])
    for _, grp in tg.groupby("team", sort=False):
        X = grp[stats].to_numpy(dtype=float)
        seasons = grp["season"].to_numpy()
        n = len(grp)
        idx = np.arange(n)
        for j in range(n):
            m = means[grp.index[j]]
            if j == 0:
                out[grp.index[j]] = m
                continue
            i = idx[:j]
            w = 0.5 ** ((j - 1 - i) / halflife) * carry ** (seasons[j] - seasons[i])
            Xi = X[:j]
            mask = ~np.isnan(Xi)
            num = np.nansum(Xi * w[:, None], axis=0) + prior_n * m
            den = (mask * w[:, None]).sum(axis=0) + prior_n
            out[grp.index[j]] = num / den
    return out


def add_opponent_adjusted(tg):
    """Opponent-adjust EPA & margin using opponents' *pregame* long-window form (no leakage)."""
    base = ["off_epa", "def_epa", "margin"]
    tg = tg.reset_index(drop=True)
    means = pregame_means(tg, tg, base)
    pre = weighted_history(tg, base, 8.0, 0.55, 2.0, means)
    pre = pd.DataFrame(pre, columns=[f"pre_{c}" for c in base])
    tg = pd.concat([tg, pre, pd.DataFrame(means[:, :2], columns=["lg_off", "lg_def"])], axis=1)
    opp = tg[["game_id", "team", "pre_off_epa", "pre_def_epa", "pre_margin"]].rename(
        columns={"team": "opp", "pre_off_epa": "opp_pre_off", "pre_def_epa": "opp_pre_def", "pre_margin": "opp_pre_margin"})
    tg = tg.merge(opp, on=["game_id", "opp"], how="left")
    # good offense vs a stingy defense (low def_epa allowed) deserves credit
    tg["adj_off_epa"] = tg["off_epa"] - (tg["opp_pre_def"] - tg["lg_def"])
    tg["adj_def_epa"] = tg["def_epa"] - (tg["opp_pre_off"] - tg["lg_off"])
    tg["adj_margin"] = tg["margin"] + tg["opp_pre_margin"]
    return tg.drop(columns=["pre_off_epa", "pre_def_epa", "pre_margin", "opp_pre_off", "opp_pre_def", "opp_pre_margin",
                            "lg_off", "lg_def"])


def team_form(tg, games):
    """Pregame form features for every scheduled (game, team), incl. unplayed games."""
    sched = []
    for side in ("home", "away"):
        t = games[["game_id", "season", "week", "gameday", f"{side}_team"]].rename(columns={f"{side}_team": "team"})
        sched.append(t)
    sched = pd.concat(sched, ignore_index=True)
    # Unplayed games get NaN stats so they receive pregame features but contribute nothing.
    full = sched.merge(tg.drop(columns=["season", "week", "gameday"]), on=["game_id", "team"], how="left")
    full = full.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    means = pregame_means(full, tg, ROLL_STATS)

    feats = [full[["game_id", "team"]]]
    for name, hl, carry, pn in WINDOWS:
        arr = weighted_history(full, ROLL_STATS, hl, carry, pn, means)
        feats.append(pd.DataFrame(arr, columns=[f"{name}_{c}" for c in ROLL_STATS]))
    f = pd.concat(feats, axis=1)

    # Momentum: streaks / recent record (current season only), and trend = short - long
    mom = []
    for _, grp in full.sort_values("gameday").groupby("team", sort=False):
        streak, last = 0, []
        prev_season = None
        for gid, season, win in zip(grp["game_id"], grp["season"], grp["win"]):
            if season != prev_season:
                streak, last, played, wins = 0, [], 0, 0.0
                prev_season = season
            mom.append((gid, grp["team"].iloc[0], streak, np.mean(last[-3:]) if last else 0.5,
                        played, wins / played if played else 0.5))
            if not np.isnan(win):
                streak = (streak + 1 if streak > 0 else 1) if win == 1 else ((streak - 1 if streak < 0 else -1) if win == 0 else 0)
                last.append(win)
                played += 1
                wins += win
    mom = pd.DataFrame(mom, columns=["game_id", "team", "streak", "last3_win", "games_played", "season_win_pct"])
    f = f.merge(mom, on=["game_id", "team"], how="left")
    for c in ("off_epa", "def_epa", "margin", "adj_margin"):
        f[f"trend_{c}"] = f[f"s_{c}"] - f[f"l_{c}"]
    return f


# ------------------------------------------------------------------ Elo

def elo_ratings(games):
    elo = {}
    last_season = {}
    rows = []
    for g in games.itertuples():
        for t in (g.home_team, g.away_team):
            if t not in elo:
                elo[t] = 1500.0
            if last_season.get(t) not in (None, g.season):
                elo[t] = elo[t] * (1 - ELO_REVERT) + 1500 * ELO_REVERT
            last_season[t] = g.season
        hfa = 0 if g.location == "Neutral" else ELO_HFA
        rh, ra = elo[g.home_team], elo[g.away_team]
        rows.append((g.game_id, rh, ra))
        if pd.isna(g.home_score):
            continue
        p = 1 / (1 + 10 ** (-(rh + hfa - ra) / 400))
        m = g.home_score - g.away_score
        s = 1.0 if m > 0 else 0.0 if m < 0 else 0.5
        winner_diff = (rh + hfa - ra) if m > 0 else (ra - rh - hfa)
        mult = math.log(abs(m) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
        delta = ELO_K * mult * (s - p)
        elo[g.home_team] += delta
        elo[g.away_team] -= delta
    return pd.DataFrame(rows, columns=["game_id", "h_elo", "a_elo"])


# ------------------------------------------------------------------ ridge power ratings

# Tuned on 2016-2021 (corr of rating diff with margin): long memory wins; a medium-memory
# version is kept as an opponent-adjusted "recent form" signal.
PR_CONFIGS = {"pr": (200.0, 0.7, 10.0), "prm": (56.0, 0.5, 10.0)}  # prefix: (half-life days, season carry, ridge lambda)


def _ridge(X, y, w, lam):
    Xw = X * w[:, None]
    A = X.T @ Xw + lam * np.eye(X.shape[1])
    A[0, 0] -= lam  # don't shrink the intercept
    return np.linalg.solve(A, Xw.T @ y)


def power_ratings(games, tg, halflife=200.0, carry=0.7, lam=10.0, prefix="pr"):
    """Opponent-adjusted team ratings fit on all games before each week's first kickoff.

    margin: margin = hfa + r_home - r_away           -> pr_margin
    epa:    off_epa(team) = mu + o_team + d_opp     -> pr_off (higher = better offense),
                                                       pr_def (higher = better defense)
    pass/rush: same split by play type              -> pr_pass_off/def, pr_rush_off/def
    """
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    ti = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    played = games[games["home_score"].notna()].copy()
    tgx = tg.set_index(["game_id", "team"])
    rows = []
    for (season, week), wk in games.groupby(["season", "week"], sort=True):
        cutoff = wk["gameday"].min()
        past = played[played["gameday"] < cutoff]
        if len(past) < 50:
            for t in teams:
                rows.append({"season": season, "week": week, "team": t})
            continue
        days = (cutoff - past["gameday"]).dt.days.to_numpy()
        w = 0.5 ** (days / halflife) * carry ** (season - past["season"].to_numpy())
        hi = past["home_team"].map(ti).to_numpy()
        ai = past["away_team"].map(ti).to_numpy()
        n = len(past)
        # margin model: [1(hfa), teams...]
        X = np.zeros((n, T + 1))
        X[:, 0] = (past["location"] != "Neutral").to_numpy()
        X[np.arange(n), 1 + hi] = 1
        X[np.arange(n), 1 + ai] = -1
        y = (past["home_score"] - past["away_score"]).to_numpy(dtype=float)
        rm = _ridge(X, y, w, lam)[1:]

        # epa models: one row per offense: [1, off_team..., def_team...]
        def unit(col):
            rr = []
            for side, oside in (("home", "away"), ("away", "home")):
                key = list(zip(past["game_id"], past[f"{side}_team"]))
                vals = tgx.reindex(key)[col].to_numpy(dtype=float)
                rr.append((past[f"{side}_team"].map(ti).to_numpy(), past[f"{oside}_team"].map(ti).to_numpy(), vals))
            o = np.concatenate([r[0] for r in rr]); d = np.concatenate([r[1] for r in rr])
            v = np.concatenate([r[2] for r in rr]); ww = np.concatenate([w, w])
            ok = ~np.isnan(v)
            m = ok.sum()
            Xe = np.zeros((m, 2 * T + 1))
            Xe[:, 0] = 1
            Xe[np.arange(m), 1 + o[ok]] = 1
            Xe[np.arange(m), 1 + T + d[ok]] = 1
            b = _ridge(Xe, v[ok], ww[ok], lam)
            return b[1:T + 1], -b[T + 1:]

        po, pd_ = unit("off_epa")
        ppo, ppd = unit("off_pass_epa")
        pro, prd = unit("off_rush_epa")
        for t, i in ti.items():
            rows.append({"season": season, "week": week, "team": t, "margin": rm[i], "off": po[i],
                         "def": pd_[i], "pass_off": ppo[i], "pass_def": ppd[i],
                         "rush_off": pro[i], "rush_def": prd[i]})
    out = pd.DataFrame(rows)
    return out.rename(columns={c: f"{prefix}_{c}" for c in out.columns if c not in ("season", "week", "team")})


# ------------------------------------------------------------------ QB

QB_DECAY_GAME = 0.95     # per QB game played
QB_DECAY_SEASON = 0.75   # extra per offseason
QB_PRIOR_PLAYS = 120.0


def qb_features(games, qbg):
    players = pd.read_csv(RAW / "players.csv", usecols=["gsis_id", "draft_pick", "draft_round"])
    pick = players.set_index("gsis_id")["draft_pick"].to_dict()
    dates = games.set_index("game_id")["gameday"]
    q = qbg.copy()
    q["gameday"] = q["game_id"].map(dates)
    q = q.dropna(subset=["gameday"]).sort_values("gameday")
    q["plays"] = q["dropbacks"] + q["qb_runs"]
    q["val"] = q["qb_epa"] + q["qb_run_epa"]

    # per-QB running state: decayed EPA sum, decayed plays, decayed cpoe, career plays
    state = {}
    snapshots = {}  # (qb, game_id) not needed; we snapshot by date via ordered events

    def prior_mean(qb):
        p = pick.get(qb)
        p = 300 if p is None or np.isnan(p) else p
        return -0.12 + 0.12 * math.exp(-p / 40.0)

    def rating(qb):
        s = state.get(qb)
        pm = prior_mean(qb)
        if s is None:
            return pm, 0.0, 0.0, 0.0
        val, n, cp, cn, career, _ = s
        return ((val + QB_PRIOR_PLAYS * pm) / (n + QB_PRIOR_PLAYS), n, (cp / cn if cn else 0.0), career)

    # team starter history (by date) to detect changes / backups
    team_hist = {}  # team -> list of (gameday, qb_id, dropbacks)

    out = []
    events = q.groupby("game_id", sort=False)
    q_by_game = {gid: d for gid, d in events}
    for g in games.itertuples():
        for side in ("home", "away"):
            team = getattr(g, f"{side}_team")
            hist = team_hist.get(team, [])
            starter = getattr(g, f"{side}_qb_id")
            if not isinstance(starter, str):
                starter = hist[-1][1] if hist else None
            r, n, cpoe, career = rating(starter) if starter else (-0.1, 0, 0, 0)
            prev_starter = hist[-1][1] if hist else None
            recent = hist[-8:]
            tot = sum(h[2] for h in recent)
            share = sum(h[2] for h in recent if h[1] == starter) / tot if tot else 0.0
            # rating of the team's "usual" QB (most dropbacks over the last 8 games)
            if recent:
                usual = pd.Series({h[1]: 0 for h in recent})
                for h in recent:
                    usual[h[1]] += h[2]
                usual_qb = usual.idxmax()
                usual_r = rating(usual_qb)[0]
            else:
                usual_r = r
            out.append({"game_id": g.game_id, "side": side, "qb_rating": r, "qb_plays_decayed": n,
                        "qb_cpoe": cpoe, "qb_career_plays": career,
                        "qb_changed": int(prev_starter is not None and starter != prev_starter),
                        "qb_team_share": share, "qb_vs_usual": r - usual_r})
        # update with this game's results
        d = q_by_game.get(g.game_id)
        if d is None:
            continue
        for row in d.itertuples():
            s = state.get(row.qb_id)
            if s is None:
                s = [0.0, 0.0, 0.0, 0.0, 0.0, g.season]
            if s[5] != g.season:
                s[0] *= QB_DECAY_SEASON; s[1] *= QB_DECAY_SEASON; s[2] *= QB_DECAY_SEASON; s[3] *= QB_DECAY_SEASON
                s[5] = g.season
            s[0] = s[0] * QB_DECAY_GAME + row.val
            s[1] = s[1] * QB_DECAY_GAME + row.plays
            if row.cpoe_n:
                s[2] = s[2] * QB_DECAY_GAME + row.cpoe_sum
                s[3] = s[3] * QB_DECAY_GAME + row.cpoe_n
            s[4] += row.plays
            state[row.qb_id] = s
            team_hist.setdefault(row.team, []).append((g.gameday, row.qb_id, row.dropbacks))
    df = pd.DataFrame(out)
    h = df[df.side == "home"].drop(columns="side").add_prefix("h_").rename(columns={"h_game_id": "game_id"})
    a = df[df.side == "away"].drop(columns="side").add_prefix("a_").rename(columns={"a_game_id": "game_id"})
    return h.merge(a, on="game_id")


# ------------------------------------------------------------------ playoff stakes

DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SEA", "SF"],
}
TEAM_DIV = {t: d for d, ts in DIVISIONS.items() for t in ts}


def stakes_features(games):
    """Approximate playoff picture before each regular-season week (ignores tiebreakers).

    cushion  = (team wins - wins of the first team out) in games, positive if currently in;
               for teams out, minus the deficit to the last team in.
    clinched / eliminated = cushion exceeds the games left to play.
    bye_locked = lead over the next seed for the bye exceeds the games left (rest-starters risk).
    """
    rows = []
    reg = games[games["game_type"] == "REG"]
    for season, sg in reg.groupby("season"):
        n_games = 17 if season >= 2021 else 16
        n_spots = 7 if season >= 2020 else 6
        n_byes = 1 if season >= 2020 else 2
        for week, wk in sg.groupby("week"):
            cutoff = wk["gameday"].min()
            past = sg[(sg["gameday"] < cutoff) & sg["home_score"].notna()]
            wins = {t: 0.0 for t in TEAM_DIV}
            played = {t: 0 for t in TEAM_DIV}
            for g in past.itertuples():
                m = g.home_score - g.away_score
                wins[g.home_team] += 1.0 if m > 0 else 0.5 if m == 0 else 0.0
                wins[g.away_team] += 1.0 if m < 0 else 0.5 if m == 0 else 0.0
                played[g.home_team] += 1
                played[g.away_team] += 1
            for conf in ("AFC", "NFC"):
                teams = [t for t in TEAM_DIV if TEAM_DIV[t].startswith(conf)]
                # division leaders get the top seeds, wildcards fill the rest
                leaders = [max(DIVISIONS[d], key=lambda t: wins[t]) for d in DIVISIONS if d.startswith(conf)]
                leaders.sort(key=lambda t: -wins[t])
                others = sorted([t for t in teams if t not in leaders], key=lambda t: -wins[t])
                seeds = leaders + others
                last_in = wins[seeds[n_spots - 1]]
                first_out = wins[seeds[n_spots]]
                for t in teams:
                    left = n_games - played[t]
                    cushion = wins[t] - first_out if seeds.index(t) < n_spots else wins[t] - last_in
                    bye_lead = (wins[t] - wins[seeds[n_byes]]) if seeds.index(t) < n_byes else -1.0
                    rows.append({"season": season, "week": week, "team": t,
                                 "stakes_cushion": cushion / max(left, 1),
                                 "stakes_clinched": int(cushion > left),
                                 "stakes_eliminated": int(-cushion > left),
                                 "stakes_bye_locked": int(bye_lead > left),
                                 "stakes_late": int(left <= 3)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ context & Vegas

def qb_adjusted(df, games, qbg, form):
    """qbx_rating: QB EPA per play adjusted for each opponent's pregame pass defense (long window),
    decayed and shrunk like qb_rating; qbx_age and qbx_age2 (years at kickoff, centered at 28)."""
    from common import RAW
    birth = pd.read_csv(RAW / "players.csv", usecols=["gsis_id", "birth_date"]).dropna()
    birth = dict(zip(birth["gsis_id"], pd.to_datetime(birth["birth_date"], errors="coerce")))
    opp_def = {}
    lf = form.set_index(["game_id", "team"])["l_def_pass_epa"].to_dict()
    gm = games.set_index("game_id")
    league = form["l_def_pass_epa"].mean()
    for gid, r in gm[["home_team", "away_team"]].iterrows():
        opp_def[(gid, r["home_team"])] = lf.get((gid, r["away_team"]), league) - league
        opp_def[(gid, r["away_team"])] = lf.get((gid, r["home_team"]), league) - league
    q = qbg.copy()
    q["gameday"] = q["game_id"].map(gm["gameday"])
    q = q.dropna(subset=["gameday"])
    q["plays"] = q["dropbacks"] + q["qb_runs"]
    # a QB facing a defense that allows more EPA than average gets less credit
    q["val"] = q["qb_epa"] + q["qb_run_epa"] - q["plays"] * [opp_def.get((g, t), 0.0) for g, t in zip(q["game_id"], q["team"])]
    by_game = {g: d for g, d in q.groupby("game_id")}
    state, last = {}, {}
    out = {}
    for g in games.sort_values(["gameday", "gametime", "game_id"]).itertuples():
        for side in ("home", "away"):
            team = getattr(g, f"{side}_team")
            qb = getattr(g, f"{side}_qb_id")
            if not isinstance(qb, str):
                qb = last.get(team)
            s_ = state.get(qb)
            rating = -0.08 if s_ is None else (s_[0] + 120 * -0.08) / (s_[1] + 120)
            b = birth.get(qb)
            age = (pd.Timestamp(g.gameday) - b).days / 365.25 - 28 if b is not None and not pd.isna(b) else 0.0
            out[(g.game_id, side)] = (rating, age, age * age)
        d = by_game.get(g.game_id)
        if d is None:
            continue
        for r in d.itertuples():
            s_ = state.get(r.qb_id, [0.0, 0.0, g.season])
            if s_[2] != g.season:
                s_ = [s_[0] * 0.75, s_[1] * 0.75, g.season]
            state[r.qb_id] = [s_[0] * 0.95 + r.val, s_[1] * 0.95 + r.plays, g.season]
            last[r.team] = r.qb_id
    for side, pfx in (("home", "h"), ("away", "a")):
        v = [out.get((gid, side), (np.nan, np.nan, np.nan)) for gid in df["game_id"]]
        df[f"{pfx}_qbx_rating"] = [x[0] for x in v]
        df[f"{pfx}_qbx_age"] = [x[1] for x in v]
        df[f"{pfx}_qbx_age2"] = [x[2] for x in v]
    return df


def coach_features(df):
    """Per-team coach features from the full schedule history (1999+), using only earlier games:
    coach_new (first season with this team), coach_tenure (log games with team), coach_winpct
    (career win rate shrunk toward .500 with 16 pseudo-games)."""
    from common import RAW, norm_team
    g = pd.read_csv(RAW / "games.csv")
    for c in ("home_team", "away_team"):
        g[c] = g[c].map(norm_team)
    g = g.sort_values(["gameday", "gametime", "game_id"])
    career, tenure, first_season = {}, {}, {}
    feats = {}
    for r in g.itertuples():
        for side in ("home", "away"):
            coach, team = getattr(r, f"{side}_coach"), getattr(r, f"{side}_team")
            if not isinstance(coach, str):
                continue
            w, n = career.get(coach, (0.0, 0))
            key = (coach, team)
            first_season.setdefault(key, r.season)
            feats[(r.game_id, side)] = (int(first_season[key] == r.season), np.log1p(tenure.get(key, 0)),
                                        (w + 8) / (n + 16))
        if pd.isna(r.home_score):
            continue
        m = r.home_score - r.away_score
        for side, res in (("home", 1.0 if m > 0 else 0.0 if m < 0 else 0.5), ("away", 1.0 if m < 0 else 0.0 if m > 0 else 0.5)):
            coach, team = getattr(r, f"{side}_coach"), getattr(r, f"{side}_team")
            if not isinstance(coach, str):
                continue
            w, n = career.get(coach, (0.0, 0))
            career[coach] = (w + res, n + 1)
            tenure[(coach, team)] = tenure.get((coach, team), 0) + 1
    for side, pfx in (("home", "h"), ("away", "a")):
        v = [feats.get((gid, side), (np.nan, np.nan, np.nan)) for gid in df["game_id"]]
        df[f"{pfx}_coach_new"] = [x[0] for x in v]
        df[f"{pfx}_coach_tenure"] = [x[1] for x in v]
        df[f"{pfx}_coach_winpct"] = [x[2] for x in v]
    return df


def kicker_features(df, games):
    """k_fgoe: the team's most recent kicker's field goals made over expected per attempt (expected
    from a league logistic curve on distance), exponentially weighted and shrunk; earlier games only."""
    from common import CURRENT_SEASON, FIRST_SEASON, load_per_season, norm_team
    from sklearn.linear_model import LogisticRegression
    k = load_per_season("pbp", "play_by_play", range(FIRST_SEASON, CURRENT_SEASON + 1),
                        usecols=["game_id", "posteam", "kicker_player_id", "field_goal_result", "kick_distance"])
    k = k[k["field_goal_result"].isin(["made", "missed", "blocked"]) & k["kick_distance"].notna()].copy()
    k["posteam"] = k["posteam"].map(norm_team)
    k["made"] = (k["field_goal_result"] == "made").astype(float)
    curve = LogisticRegression().fit(k[["kick_distance"]], k["made"])  # distance curve is stable over eras
    k["oe"] = k["made"] - curve.predict_proba(k[["kick_distance"]])[:, 1]
    dates = games.set_index("game_id")["gameday"]
    k["gameday"] = k["game_id"].map(dates)
    k = k.dropna(subset=["gameday"]).sort_values("gameday")
    state, last_kicker, rating_before = {}, {}, {}
    # walk games in date order: snapshot each team's kicker rating before the game, then update
    by_game = {gid: d for gid, d in k.groupby("game_id")}
    for g in games.sort_values("gameday").itertuples():
        for side in ("home", "away"):
            t = getattr(g, f"{side}_team")
            kid = last_kicker.get(t)
            s_, n_ = state.get(kid, (0.0, 0.0)) if kid else (0.0, 0.0)
            rating_before[(g.game_id, side)] = s_ / (n_ + 15.0)
        d = by_game.get(g.game_id)
        if d is None:
            continue
        for r in d.itertuples():
            s_, n_ = state.get(r.kicker_player_id, (0.0, 0.0))
            state[r.kicker_player_id] = (0.97 * s_ + r.oe, 0.97 * n_ + 1)
            last_kicker[r.posteam] = r.kicker_player_id
    for side, pfx in (("home", "h"), ("away", "a")):
        df[f"{pfx}_k_fgoe"] = [rating_before.get((gid, side), 0.0) for gid in df["game_id"]]
    return df


def add_weather(df, games):
    """Kickoff weather (src/weather.py cache) as shared + per-team context features."""
    from common import RAW
    f = RAW / "weather_games.csv"
    if not f.exists():
        return df
    w = pd.read_csv(f)[["game_id", "wx_temp", "wx_wind", "wx_precip", "wx_indoor"]]
    df = df.merge(w, on="game_id", how="left")
    df["ctx_wx_wind"] = df["wx_wind"]
    df["ctx_wx_cold"] = (45 - df["wx_temp"]).clip(lower=0)
    df["ctx_wx_precip"] = np.log1p(df["wx_precip"])
    # Does each team normally play indoors? (its usual home roof that season)
    home = games.dropna(subset=["roof"]).assign(ind=lambda d: d["roof"].isin(["dome", "closed"]))
    dome_team = home.groupby(["season", "home_team"])["ind"].mean().gt(0.5).to_dict()
    for side, pfx in (("home", "h"), ("away", "a")):
        dt = np.array([dome_team.get((s_, t), False) for s_, t in zip(df["season"], df[f"{side}_team"])])
        df[f"ctx_{pfx}_dome_cold"] = np.where(dt & (df["wx_indoor"] == 0), df["ctx_wx_cold"], 0.0)
        df[f"ctx_{pfx}_wind_pass"] = df["ctx_wx_wind"] * df[f"{pfx}_l_off_pass_epa"]
    return df.drop(columns=["wx_temp", "wx_wind", "wx_precip", "wx_indoor"])


def context(games):
    rows = []
    for g in games.itertuples():
        gd = g._asdict()
        vlat, vlon, vtz = venue_geo(gd)
        hlat, hlon, htz = team_geo(g.home_team, g.season)
        alat, alon, atz = team_geo(g.away_team, g.season)
        try:
            kick = int(str(g.gametime)[:2]) + int(str(g.gametime)[3:5]) / 60
        except ValueError:
            kick = 13.0
        rows.append({
            "game_id": g.game_id,
            "ctx_neutral": int(g.location == "Neutral"),
            "ctx_playoff": int(g.game_type != "REG"),
            "ctx_div": int(g.div_game == 1),
            "ctx_dome": int(g.roof in ("dome", "closed")),
            "ctx_week": min(g.week, 18) / 18,
            "ctx_season": g.season,
            "ctx_h_rest": min(g.home_rest, 14) if not pd.isna(g.home_rest) else 7,
            "ctx_a_rest": min(g.away_rest, 14) if not pd.isna(g.away_rest) else 7,
            "ctx_h_travel": haversine_miles((hlat, hlon), (vlat, vlon)) / 1000,
            "ctx_a_travel": haversine_miles((alat, alon), (vlat, vlon)) / 1000,
            "ctx_a_tz_shift": vtz - atz,
            "ctx_h_tz_shift": vtz - htz,
            # kickoff on each team's body clock (hours, local time of team's home)
            "ctx_a_body_kick": kick + atz,
            "ctx_h_body_kick": kick + htz,
            "ctx_primetime": int(kick >= 19.5),
        })
    c = pd.DataFrame(rows)
    c["ctx_rest_diff"] = c["ctx_h_rest"] - c["ctx_a_rest"]
    return c


def ml_to_prob(ml):
    ml = np.asarray(ml, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def vegas(games):
    ph = ml_to_prob(games["home_moneyline"])
    pa = ml_to_prob(games["away_moneyline"])
    from scipy.stats import norm
    out = pd.DataFrame({"game_id": games["game_id"],
                        "vegas_ml_prob": ph / (ph + pa),
                        "vegas_spread_prob": norm.cdf(games["spread_line"] / 13.45),
                        "spread_line": games["spread_line"]})
    out["vegas_prob"] = out["vegas_ml_prob"].fillna(out["vegas_spread_prob"])
    return out.drop(columns=["vegas_ml_prob", "vegas_spread_prob"])


# ------------------------------------------------------------------ assemble

def build(games=None):
    games = load_games() if games is None else games
    tg = pd.read_csv(PROCESSED / "team_games.csv", parse_dates=["gameday"])
    qbg = pd.read_csv(PROCESSED / "qb_games.csv")
    rs = pd.read_csv(PROCESSED / "roster_strength.csv")

    tg = add_opponent_adjusted(tg)
    form = team_form(tg, games)
    side_feats = form.merge(rs, on=["game_id", "team"], how="left")

    base = games[["game_id", "season", "week", "gameday", "game_type", "home_team", "away_team",
                  "home_score", "away_score"]].copy()
    h = side_feats.rename(columns={"team": "home_team"})
    h = h.rename(columns={c: f"h_{c}" for c in h.columns if c not in ("game_id", "home_team")})
    a = side_feats.rename(columns={"team": "away_team"})
    a = a.rename(columns={c: f"a_{c}" for c in a.columns if c not in ("game_id", "away_team")})
    df = base.merge(h, on=["game_id", "home_team"], how="left").merge(a, on=["game_id", "away_team"], how="left")
    for prefix, (hl, carry, lam) in PR_CONFIGS.items():
        pr = power_ratings(games, tg, hl, carry, lam, prefix)
        prc = [c for c in pr.columns if c.startswith(prefix + "_")]
        for side, pfx in (("home", "h"), ("away", "a")):
            df = df.merge(pr.rename(columns={"team": f"{side}_team", **{c: f"{pfx}_{c}" for c in prc}}),
                          on=["season", "week", f"{side}_team"], how="left")
    stakes = stakes_features(games)
    stc = [c for c in stakes.columns if c.startswith("stakes_")]
    for side, pfx in (("home", "h"), ("away", "a")):
        df = df.merge(stakes.rename(columns={"team": f"{side}_team", **{c: f"{pfx}_{c}" for c in stc}}),
                      on=["season", "week", f"{side}_team"], how="left")
        df[[f"{pfx}_{c}" for c in stc]] = df[[f"{pfx}_{c}" for c in stc]].fillna(0)  # playoffs: all-in
    df = df.merge(elo_ratings(games), on="game_id", how="left")
    df = df.merge(qb_features(games, qbg), on="game_id", how="left")
    df = df.merge(context(games), on="game_id", how="left")
    df = add_weather(df, games)
    df = coach_features(df)
    df = kicker_features(df, games)
    df = qb_adjusted(df, games, qbg, form)
    import kalman
    kf = kalman.ratings(games, tg)
    for side, pfx in (("home", "h"), ("away", "a")):
        df = df.merge(kf.rename(columns={"team": f"{side}_team", **{c: f"{pfx}_{c}" for c in kf.columns if c.startswith("kf_")}}),
                      on=["game_id", f"{side}_team"], how="left")
    df = df.merge(vegas(games), on="game_id", how="left")

    m = df["home_score"] - df["away_score"]
    df["target"] = np.where(m > 0, 1.0, np.where(m < 0, 0.0, 0.5))
    df.loc[df["home_score"].isna(), "target"] = np.nan
    df["margin"] = m

    df = df[df["season"] >= FIRST_MODEL_SEASON].reset_index(drop=True)
    df.to_csv(PROCESSED / "features.csv", index=False)
    print(f"features: {df.shape}, played: {df['target'].notna().sum()}")
    return df


if __name__ == "__main__":
    build()
