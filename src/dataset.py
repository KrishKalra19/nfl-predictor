"""Turn features.csv into model matrices.

Each game is seen from "team 1" vs "team 2". Training data is augmented with the mirrored view
(team 1 = away team, label flipped, home flag = -1), which forces every model to be symmetric
and learn home-field advantage only through the explicit `home` feature.
"""
import numpy as np
import pandas as pd

from common import PROCESSED

SHARED = ["ctx_playoff", "ctx_div", "ctx_dome", "ctx_week", "ctx_primetime"]
SIDE_CTX = ["rest", "travel", "tz_shift", "body_kick"]
DROP = {"elo"}  # elo is handled as its own side column below (h_elo/a_elo), listed for clarity


def load_features():
    df = pd.read_csv(PROCESSED / "features.csv", parse_dates=["gameday"])
    return df


def team_columns(df):
    """Per-team feature names (without h_/a_ prefix) present for both sides."""
    names = []
    for c in df.columns:
        if c.startswith("h_") and f"a_{c[2:]}" in df.columns:
            names.append(c[2:])
    return names


def side_matrix(df, prefix, names):
    cols = [f"{prefix}_{n}" for n in names] + [f"ctx_{prefix}_{c}" for c in SIDE_CTX]
    return df[cols].to_numpy(dtype=np.float32)


def views(df, names):
    """Return (team1, team2, shared, home_flag) for the home-perspective view."""
    t1 = side_matrix(df, "h", names)
    t2 = side_matrix(df, "a", names)
    shared = df[SHARED].to_numpy(dtype=np.float32)
    home = np.where(df["ctx_neutral"].to_numpy() == 1, 0.0, 1.0).astype(np.float32)
    return t1, t2, shared, home


class Prep:
    """Median-impute + standardize using training rows only (fit on both views of train)."""

    def fit(self, t1, t2, shared):
        both = np.vstack([t1, t2])
        self.t_med = np.nanmedian(both, axis=0)
        filled = np.where(np.isnan(both), self.t_med, both)
        self.t_mu, self.t_sd = filled.mean(0), filled.std(0) + 1e-6
        self.s_med = np.nanmedian(shared, axis=0)
        sf = np.where(np.isnan(shared), self.s_med, shared)
        self.s_mu, self.s_sd = sf.mean(0), sf.std(0) + 1e-6
        return self

    def team(self, t):
        t = np.where(np.isnan(t), self.t_med, t)
        return ((t - self.t_mu) / self.t_sd).astype(np.float32)

    def shared(self, s):
        s = np.where(np.isnan(s), self.s_med, s)
        return ((s - self.s_mu) / self.s_sd).astype(np.float32)


MARGIN_SCALE = 14.0  # ~1 SD of NFL margins; keeps the margin target on a unit scale


def make_xy(df, names, prep, augment, with_margin=False):
    """Standardized (t1, t2, shared, home, y[, margin]). With augment, append the mirrored view."""
    t1, t2, sh, home = views(df, names)
    t1, t2, sh = prep.team(t1), prep.team(t2), prep.shared(sh)
    y = df["target"].to_numpy(dtype=np.float32)
    m = (df["margin"].to_numpy(dtype=np.float32) / MARGIN_SCALE) if "margin" in df else np.zeros_like(y)
    if not augment:
        out = (t1, t2, sh, home, y)
        return out + (m,) if with_margin else out
    out = (np.vstack([t1, t2]), np.vstack([t2, t1]), np.vstack([sh, sh]),
           np.concatenate([home, -home]), np.concatenate([y, 1 - y]))
    return out + (np.concatenate([m, -m]),) if with_margin else out


def diff_features(t1, t2, sh, home):
    """Flat representation for LR / GBM: team1 - team2, shared context, home flag."""
    return np.hstack([t1 - t2, sh, home[:, None]])


def expand_ties(X, y):
    """sklearn needs hard labels: a tie becomes half a win + half a loss."""
    tie = y == 0.5
    if not tie.any():
        return X, y, np.ones_like(y)
    X2 = np.vstack([X[~tie], X[tie], X[tie]])
    y2 = np.concatenate([y[~tie], np.ones(tie.sum()), np.zeros(tie.sum())])
    w2 = np.concatenate([np.ones((~tie).sum()), np.full(2 * tie.sum(), 0.5)])
    return X2, y2, w2
