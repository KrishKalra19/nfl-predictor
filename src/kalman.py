"""State-space (Kalman filter) team ratings.

Each team has latent strengths that drift week to week. Two filters run jointly over all teams:

  margin model   state r_t (points)         obs: home_margin = hfa + r_home - r_away + e,  e ~ N(0, 13.5^2)
  EPA model      state (o_t, d_t) EPA/play  obs: off_epa(A vs B) = mu + o_A - d_B + e,      e ~ N(0, 0.17^2)

Between weeks every rating gets process noise (ratings drift); between seasons the mean regresses
toward 0 and uncertainty widens (rosters turn over). For every game the pregame snapshot is taken
for the whole week before any of that week's results are applied, matching how live picks work.
Outputs per team-game: kf_margin, kf_margin_sd, kf_off, kf_def (higher = better defense), kf_off_sd.
"""
import numpy as np
import pandas as pd

M_OBS_SD, M_WEEK_SD, M_INIT_SD, M_SEASON_KEEP, M_SEASON_SD, HFA = 13.5, 1.0, 6.0, 0.75, 3.0, 1.8
E_OBS_SD, E_WEEK_SD, E_INIT_SD, E_SEASON_KEEP, E_SEASON_SD = 0.17, 0.012, 0.08, 0.7, 0.05


class Filter:
    def __init__(self, n, init_sd):
        self.x = np.zeros(n)
        self.P = np.eye(n) * init_sd ** 2

    def drift(self, sd):
        self.P += np.eye(len(self.x)) * sd ** 2

    def regress(self, keep, sd):
        self.x *= keep
        self.P = self.P * keep ** 2 + np.eye(len(self.x)) * sd ** 2

    def update(self, h, y, obs_sd):
        s = h @ self.P @ h + obs_sd ** 2
        k = self.P @ h / s
        self.x = self.x + k * (y - h @ self.x)
        self.P = self.P - np.outer(k, h @ self.P)


def ratings(games, tg):
    """games: schedule (normalized teams); tg: team_games (per team-game off_epa)."""
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    ti = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    fm = Filter(T, M_INIT_SD)
    fe = Filter(2 * T, E_INIT_SD)          # [o_0..o_T-1, d_0..d_T-1]
    off = tg.set_index(["game_id", "team"])["off_epa"].to_dict()
    ep_sum, ep_n = 0.0, 0  # league-average EPA from games seen so far (a whole-dataset mean would leak)

    rows, prev_season = [], None
    for (season, week), wk in games.sort_values("gameday").groupby(["season", "week"], sort=True):
        if prev_season is not None and season != prev_season:
            fm.regress(M_SEASON_KEEP, M_SEASON_SD)
            fe.regress(E_SEASON_KEEP, E_SEASON_SD)
        elif prev_season is not None:
            fm.drift(M_WEEK_SD)
            fe.drift(E_WEEK_SD)
        prev_season = season
        msd = np.sqrt(np.diag(fm.P))
        esd = np.sqrt(np.diag(fe.P))
        for g in wk.itertuples():
            for side in ("home", "away"):
                i = ti[getattr(g, f"{side}_team")]
                rows.append({"game_id": g.game_id, "team": teams[i], "kf_margin": fm.x[i], "kf_margin_sd": msd[i],
                             "kf_off": fe.x[i], "kf_def": -fe.x[T + i], "kf_off_sd": esd[i]})
        wk_vals = [off.get((g.game_id, t)) for g in wk.itertuples() for t in (g.home_team, g.away_team)]
        wk_vals = [v for v in wk_vals if v is not None and not np.isnan(v)]
        ep_sum, ep_n = ep_sum + sum(wk_vals), ep_n + len(wk_vals)
        mu = ep_sum / ep_n if ep_n else 0.0
        for g in wk.itertuples():
            if pd.isna(g.home_score):
                continue
            h, a = ti[g.home_team], ti[g.away_team]
            hv = np.zeros(T)
            hv[h], hv[a] = 1, -1
            fm.update(hv, (g.home_score - g.away_score) - (0 if g.location == "Neutral" else HFA), M_OBS_SD)
            for o, d, team in ((h, a, g.home_team), (a, h, g.away_team)):
                y = off.get((g.game_id, team))
                if y is None or np.isnan(y):
                    continue
                he = np.zeros(2 * T)
                he[o], he[T + d] = 1, 1
                fe.update(he, y - mu, E_OBS_SD)
    return pd.DataFrame(rows)
