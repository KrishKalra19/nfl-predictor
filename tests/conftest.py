"""Shared fixtures: a synthetic two-season league, so tests run without the (gitignored) nflverse data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

TEAMS = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
         "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"]
STAT_COLS = ["off_epa", "off_sr", "off_pass_epa", "off_rush_epa", "off_neutral_epa", "off_explosive", "off_turnovers",
             "off_sack_rate", "off_early_pass_rate", "off_pass_sr", "off_rush_sr", "st_epa"]


@pytest.fixture(scope="session")
def league():
    """games (schedule with scores) and tg (per team-game stats) for seasons 2021-2022, 17 weeks each."""
    rng = np.random.default_rng(0)
    strength = dict(zip(TEAMS, rng.normal(0, 5, len(TEAMS))))
    games, rows = [], []
    for season in (2021, 2022):
        start = pd.Timestamp(f"{season}-09-12")
        for week in range(1, 18):
            order = rng.permutation(TEAMS)
            day = start + pd.Timedelta(days=7 * (week - 1))
            for i in range(16):
                h, a = order[2 * i], order[2 * i + 1]
                m = strength[h] - strength[a] + 2 + rng.normal(0, 13)
                hs = int(max(0, round(21 + m / 2)))
                as_ = int(max(0, round(21 - m / 2)))
                gid = f"{season}_{week:02d}_{a}_{h}"
                games.append({"game_id": gid, "season": season, "week": week, "gameday": day, "gametime": "13:00",
                              "game_type": "REG", "location": "Home", "home_team": h, "away_team": a,
                              "home_score": hs, "away_score": as_, "div_game": 0, "roof": "outdoors",
                              "stadium_id": None, "home_rest": 7, "away_rest": 7})
                for team, opp, pf, pa, home in ((h, a, hs, as_, 1), (a, h, as_, hs, 0)):
                    r = {"game_id": gid, "season": season, "week": week, "gameday": day, "game_type": "REG",
                         "team": team, "opp": opp, "pts_for": pf, "pts_against": pa, "is_home": home,
                         "margin": pf - pa, "win": 1.0 if pf > pa else 0.0 if pf < pa else 0.5}
                    for c in STAT_COLS:
                        r[c] = rng.normal(0.01 * strength[team], 0.2)
                    rows.append(r)
    games = pd.DataFrame(games)
    tg = pd.DataFrame(rows)
    # defensive stats are the opponent's offensive stats
    opp = tg[["game_id", "team"] + [c for c in STAT_COLS if c.startswith("off_")]].rename(columns={"team": "opp"})
    opp = opp.rename(columns={c: "def_" + c[4:] for c in opp.columns if c.startswith("off_")})
    tg = tg.merge(opp, on=["game_id", "opp"])
    tg["to_margin"] = tg["def_turnovers"] - tg["off_turnovers"]
    return games, tg


def truncate(games, tg, cutoff):
    """The league as it looked before `cutoff`: later scores unknown, later team-game stats absent."""
    g = games.copy()
    later = g["gameday"] >= cutoff
    g.loc[later, ["home_score", "away_score"]] = np.nan
    return g, tg[tg["gameday"] < cutoff].copy()
