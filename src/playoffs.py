"""Monte Carlo playoff odds for the current season.

    python src/playoffs.py      # writes docs/data/playoff_odds.json

Remaining regular-season games are simulated SIMS times. Win probabilities come from the model's
frozen predictions where they exist (this week), otherwise from the Kalman margin ratings:
P(home) = Phi((r_home - r_away + hfa) / sqrt(13.5^2 + sd_home^2 + sd_away^2)).
Seeding: 4 division winners then 3 wild cards per conference by wins; ties broken at random
(real NFL tiebreakers are not modeled). Ties in games count as half a win.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from build_features import DIVISIONS, TEAM_DIV
from common import OUTPUTS, PROCESSED, ROOT, load_games, upcoming_week
import kalman

SIMS = 10000
OUT = ROOT / "docs" / "data" / "playoff_odds.json"


def main(seed=7):
    games = load_games()
    season, week = upcoming_week(games)
    reg = games[(games["season"] == season) & (games["game_type"] == "REG")]
    done = reg[reg["home_score"].notna()]
    left = reg[reg["home_score"].isna()].reset_index(drop=True)
    teams = sorted(TEAM_DIV)
    ti = {t: i for i, t in enumerate(teams)}

    base = np.zeros(len(teams))
    for g in done.itertuples():
        m = g.home_score - g.away_score
        base[ti[g.home_team]] += 1.0 if m > 0 else 0.5 if m == 0 else 0.0
        base[ti[g.away_team]] += 1.0 if m < 0 else 0.5 if m == 0 else 0.0

    tg = pd.read_csv(PROCESSED / "team_games.csv", parse_dates=["gameday"])
    kf = kalman.ratings(games, tg)
    latest = kf[kf["game_id"].isin(left["game_id"])].drop_duplicates("team", keep="first").set_index("team")
    model = {}
    f = OUTPUTS / f"picks_{season}_wk{week:02d}.csv"
    if f.exists():
        model = pd.read_csv(f).set_index("game_id")["p_home"].to_dict()

    p = np.empty(len(left))
    for i, g in left.iterrows():
        if g["game_id"] in model:
            p[i] = model[g["game_id"]]
            continue
        h, a = latest.loc[g["home_team"]], latest.loc[g["away_team"]]
        hfa = 0 if g["location"] == "Neutral" else kalman.HFA
        sd = np.sqrt(kalman.M_OBS_SD ** 2 + h["kf_margin_sd"] ** 2 + a["kf_margin_sd"] ** 2)
        p[i] = norm.cdf((h["kf_margin"] - a["kf_margin"] + hfa) / sd)

    rng = np.random.default_rng(seed)
    home_win = rng.random((SIMS, len(left))) < p
    wins = np.tile(base, (SIMS, 1))
    hi = left["home_team"].map(ti).to_numpy()
    ai = left["away_team"].map(ti).to_numpy()
    for j in range(len(left)):
        wins[:, hi[j]] += home_win[:, j]
        wins[:, ai[j]] += ~home_win[:, j]
    noisy = wins + rng.random(wins.shape) * 1e-3  # random tiebreak

    playoff = np.zeros(len(teams)); div = np.zeros(len(teams)); top = np.zeros(len(teams))
    for conf in ("AFC", "NFC"):
        divs = [d for d in DIVISIONS if d.startswith(conf)]
        idx = {d: np.array([ti[t] for t in DIVISIONS[d]]) for d in divs}
        winners = np.stack([idx[d][np.argmax(noisy[:, idx[d]], axis=1)] for d in divs], axis=1)  # SIMS x 4
        conf_teams = np.array([ti[t] for t in teams if TEAM_DIV[t].startswith(conf)])
        for s in range(SIMS):
            w = winners[s]
            div[w] += 1
            order_w = w[np.argsort(-noisy[s, w])]
            top[order_w[0]] += 1
            rest = np.setdiff1d(conf_teams, w)
            wc = rest[np.argsort(-noisy[s, rest])][:3]
            playoff[w] += 1
            playoff[wc] += 1

    rows = []
    for t, i in ti.items():
        rows.append({"team": t, "division": TEAM_DIV[t], "wins_now": float(base[i]),
                     "proj_wins": round(float(wins[:, i].mean()), 1),
                     "playoff": round(playoff[i] / SIMS, 4), "division_title": round(div[i] / SIMS, 4),
                     "top_seed": round(top[i] / SIMS, 4)})
    rows.sort(key=lambda r: -r["playoff"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"season": season, "as_of_week": week, "sims": SIMS,
                               "games_left": int(len(left)), "teams": rows}, indent=1))
    print(f"playoff odds: season {season}, {len(left)} games left, {SIMS} sims -> {OUT.relative_to(ROOT)}")
    return rows


if __name__ == "__main__":
    for r in main()[:10]:
        print(r)
