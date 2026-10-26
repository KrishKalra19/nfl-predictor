"""Loop item 3: does a calibration layer help? Walk-forward honest test on dev seasons.

For each dev season s, fit Platt scaling (logit(p') = a * logit(p) + b) on the ensemble's
walk-forward predictions from seasons before s, then apply it to s.
Writes outputs/oof_ensemble.csv (walk-forward member + ensemble predictions, 2016-2021).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from backtest import feature_sets, metrics, walk_forward
from common import OUTPUTS
from config import LINEUP
from dataset import load_features
from evaluate import logit_mean

SEASONS = [2016, 2017, 2018, 2019, 2020, 2021]


def oof():
    f = OUTPUTS / "oof_ensemble.csv"
    if f.exists():
        return pd.read_csv(f)
    df = load_features()
    fs = feature_sets(df)
    out = None
    for name, (factory, feat) in LINEUP.items():
        p = walk_forward(factory, df, SEASONS, fs[feat])
        if out is None:
            out = p.drop(columns="pred")
        out[name] = p["pred"].to_numpy()
        print("done", name, flush=True)
    out["ensemble"] = logit_mean([out[n] for n in LINEUP])
    out.to_csv(f, index=False)
    return out


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    d = oof()
    d = d[d["target"] != 0.5]
    raw, cal = [], []
    for s in (2018, 2019, 2020, 2021):
        tr, te = d[d["season"] < s], d[d["season"] == s]
        m = LogisticRegression(C=1e6).fit(lg(tr["ensemble"].to_numpy())[:, None], tr["target"])
        p = m.predict_proba(lg(te["ensemble"].to_numpy())[:, None])[:, 1]
        print(s, "slope", round(m.coef_[0][0], 3), "intercept", round(m.intercept_[0], 3))
        raw.append(te.assign(p=te["ensemble"]))
        cal.append(te.assign(p=p))
    raw, cal = pd.concat(raw), pd.concat(cal)
    print("dev raw       ", {k: round(v, 4) for k, v in metrics(raw["target"], raw["p"]).items()})
    print("dev calibrated", {k: round(v, 4) for k, v in metrics(cal["target"], cal["p"]).items()})


if __name__ == "__main__":
    main()
