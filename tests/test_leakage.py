"""No-leakage tests: every pregame feature for a week must be identical whether or not that week's
(and later weeks') results exist. Run on a synthetic league, so CI needs no data."""
import numpy as np
import pytest

import build_features as B
import kalman
from conftest import truncate

CUTOFF_WEEK = (2022, 6)


def cutoff_of(games):
    s, w = CUTOFF_WEEK
    return games[(games["season"] == s) & (games["week"] == w)]["gameday"].min()


def week_ids(games):
    s, w = CUTOFF_WEEK
    return set(games[(games["season"] == s) & (games["week"] == w)]["game_id"])


def assert_same(full, trunc, ids, key="game_id", cols=None):
    f = full[full[key].isin(ids)].sort_values(list(full.columns[:2])).reset_index(drop=True)
    t = trunc[trunc[key].isin(ids)].sort_values(list(trunc.columns[:2])).reset_index(drop=True)
    cols = cols or [c for c in f.columns if f[c].dtype.kind in "fi"]
    assert len(f) == len(t) and len(f) > 0
    np.testing.assert_allclose(f[cols].to_numpy(float), t[cols].to_numpy(float), rtol=1e-9, atol=1e-9)


def test_elo_is_pregame(league):
    games, _ = league
    g2, _ = truncate(games, league[1], cutoff_of(games))
    assert_same(B.elo_ratings(games), B.elo_ratings(g2), week_ids(games))


def test_power_ratings_are_pregame(league):
    games, tg = league
    g2, tg2 = truncate(games, tg, cutoff_of(games))
    s, w = CUTOFF_WEEK
    full = B.power_ratings(games, B.add_opponent_adjusted(tg))
    trunc = B.power_ratings(g2, B.add_opponent_adjusted(tg2))
    pick = lambda d: d[(d["season"] == s) & (d["week"] == w)].sort_values("team").reset_index(drop=True)
    cols = [c for c in full.columns if c.startswith("pr_")]
    np.testing.assert_allclose(pick(full)[cols].to_numpy(float), pick(trunc)[cols].to_numpy(float), atol=1e-9)


def test_team_form_is_pregame(league):
    games, tg = league
    g2, tg2 = truncate(games, tg, cutoff_of(games))
    full = B.team_form(B.add_opponent_adjusted(tg), games)
    trunc = B.team_form(B.add_opponent_adjusted(tg2), g2)
    cols = [c for c in full.columns if c.startswith(("s_", "l_", "trend_")) or c in ("streak", "last3_win", "games_played")]
    ids = week_ids(games)
    f = full[full["game_id"].isin(ids)].sort_values(["game_id", "team"]).reset_index(drop=True)
    t = trunc[trunc["game_id"].isin(ids)].sort_values(["game_id", "team"]).reset_index(drop=True)
    np.testing.assert_allclose(f[cols].to_numpy(float), t[cols].to_numpy(float), atol=1e-9)


def test_standings_are_pregame(league):
    games, _ = league
    g2, _ = truncate(games, league[1], cutoff_of(games))
    s, w = CUTOFF_WEEK
    pick = lambda d: d[(d["season"] == s) & (d["week"] == w)].sort_values("team").reset_index(drop=True)
    full, trunc = pick(B.stakes_features(games)), pick(B.stakes_features(g2))
    cols = [c for c in full.columns if c.startswith("stakes_")]
    np.testing.assert_allclose(full[cols].to_numpy(float), trunc[cols].to_numpy(float), atol=1e-9)


def test_kalman_is_pregame(league):
    games, tg = league
    g2, tg2 = truncate(games, tg, cutoff_of(games))
    ids = week_ids(games)
    f = kalman.ratings(games, tg)
    t = kalman.ratings(g2, tg2)
    f = f[f["game_id"].isin(ids)].sort_values(["game_id", "team"]).reset_index(drop=True)
    t = t[t["game_id"].isin(ids)].sort_values(["game_id", "team"]).reset_index(drop=True)
    cols = [c for c in f.columns if c.startswith("kf_")]
    np.testing.assert_allclose(f[cols].to_numpy(float), t[cols].to_numpy(float), atol=1e-9)


def test_leak_detector_catches_a_leak(league):
    """Sanity check on the test itself: a feature that uses the game's own result must fail."""
    games, _ = league
    g2, _ = truncate(games, league[1], cutoff_of(games))
    leaky = lambda g: g[["game_id"]].assign(x=(g["home_score"] - g["away_score"]).fillna(0))
    with pytest.raises(AssertionError):
        assert_same(leaky(games), leaky(g2), week_ids(games), cols=["x"])
