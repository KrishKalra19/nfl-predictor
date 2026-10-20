"""GRU sequence model: each team is encoded from its last K games (in order) plus static features.

Lets the network learn momentum / form patterns directly instead of from hand-built rolling means.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from build_features import add_opponent_adjusted
from common import PROCESSED, load_games
from models import TorchModel

K = 12
SEQ_STATS = ["off_epa", "def_epa", "adj_off_epa", "adj_def_epa", "margin", "adj_margin", "off_sr", "def_sr",
             "off_pass_epa", "def_pass_epa", "off_rush_epa", "def_rush_epa", "st_epa", "to_margin", "win", "is_home"]
F = len(SEQ_STATS) + 2  # + prior-season flag + valid flag
SEQ_NAMES = [f"seq_{k}_{j}" for k in range(K) for j in range(F)]


def build_sequences():
    """DataFrame keyed by (game_id, team) with the K previous games flattened (oldest first)."""
    games = load_games()
    tg = pd.read_csv(PROCESSED / "team_games.csv", parse_dates=["gameday"])
    tg = add_opponent_adjusted(tg).sort_values("gameday")
    sched = pd.concat([games[["game_id", "season", "gameday", f"{s}_team"]].rename(columns={f"{s}_team": "team"})
                       for s in ("home", "away")]).sort_values("gameday")
    rows, keys = [], []
    for team, sg in sched.groupby("team"):
        hist = tg[tg["team"] == team]
        X = hist[SEQ_STATS].to_numpy(dtype=np.float32)
        hd = hist["gameday"].to_numpy()
        hs = hist["season"].to_numpy()
        for gid, season, gday in zip(sg["game_id"], sg["season"], sg["gameday"]):
            n = np.searchsorted(hd, np.datetime64(gday))  # games strictly before this date
            lo = max(0, n - K)
            seq = np.full((K, F), np.nan, dtype=np.float32)
            take = n - lo
            if take:
                seq[K - take:, :len(SEQ_STATS)] = X[lo:n]
                seq[K - take:, len(SEQ_STATS)] = (hs[lo:n] < season).astype(np.float32)
                seq[K - take:, len(SEQ_STATS) + 1] = 1.0
            seq[:K - take, len(SEQ_STATS) + 1] = 0.0
            seq[:K - take, len(SEQ_STATS)] = 1.0
            rows.append(seq.reshape(-1))
            keys.append((gid, team))
    df = pd.DataFrame(np.vstack(rows), columns=SEQ_NAMES)
    df[["game_id", "team"]] = pd.DataFrame(keys)
    return df


class GRUNet(nn.Module):
    def __init__(self, d_team, d_shared, emb=16, hidden=32, dropout=0.5, gru=32):
        super().__init__()
        self.d_seq = K * F
        d_static = d_team - self.d_seq
        self.gru = nn.GRU(F, gru, batch_first=True)
        self.enc = nn.Sequential(nn.Linear(gru + d_static, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, emb), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * emb + d_shared + 1, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 2))
        self.skip = nn.Linear(d_static, 2, bias=False)
        self.hfa = nn.Linear(1, 2)

    def encode(self, t):
        seq = t[:, :self.d_seq].reshape(-1, K, F)
        _, h = self.gru(seq)
        return self.enc(torch.cat([h[-1], t[:, self.d_seq:]], 1))

    def forward(self, t1, t2, sh, home):
        e1, e2 = self.encode(t1), self.encode(t2)
        z = torch.cat([e1 - e2, e1 * e2, sh, home[:, None]], 1)
        return self.head(z) + self.skip(t1[:, self.d_seq:] - t2[:, self.d_seq:]) + self.hfa(home[:, None])


_SEQ_CACHE = {}


def with_sequences(df):
    if "seq" not in _SEQ_CACHE:
        _SEQ_CACHE["seq"] = build_sequences()
    seq = _SEQ_CACHE["seq"]
    out = df
    for side, p in (("home", "h"), ("away", "a")):
        s = seq.rename(columns={"team": f"{side}_team", **{c: f"{p}_{c}" for c in SEQ_NAMES}})
        out = out.merge(s, on=["game_id", f"{side}_team"], how="left")
    return out


class GRU(TorchModel):
    net_cls = GRUNet

    def _build(self, d_team, d_shared):
        p = self.params
        kw = {k: p[k] for k in ("hidden", "dropout", "emb", "gru") if k in p}
        return GRUNet(d_team, d_shared, **kw)

    def fit(self, df, names):
        return super().fit(with_sequences(df), SEQ_NAMES + list(names))

    def predict(self, df):
        return super().predict(with_sequences(df))
