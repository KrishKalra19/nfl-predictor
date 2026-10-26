"""Dev-season A/B test for a feature addition: full lineup ensemble with vs without.

    python src/exp_features.py weather
Prints dev (2018-21) metrics for the ensemble and each member, baseline vs candidate.
"""
import sys

import dataset
from backtest import DEV_SEASONS, feature_sets, metrics, walk_forward
from config import LINEUP
from evaluate import logit_mean

BASE_SHARED = ["ctx_playoff", "ctx_div", "ctx_dome", "ctx_week", "ctx_primetime"]
BASE_SIDE = ["rest", "travel", "tz_shift", "body_kick"]

CANDIDATES = {
    # name: (extra shared ctx columns, extra side ctx suffixes, team-feature prefixes to include)
    "coach": ([], [], ["coach_"]),
    "kicker": ([], [], ["k_fgoe"]),
    "kalman": ([], [], ["kf_"]),
    "kalman_swap": ([], [], ["kf_"], ["pr_", "prm_"]),  # replace ridge ratings with Kalman ratings
    "weather": (["ctx_wx_wind", "ctx_wx_cold", "ctx_wx_precip"], ["dome_cold", "wind_pass"], []),
}


def run(shared, side, extra_prefixes, exclude_prefixes):
    dataset.SHARED[:] = shared
    dataset.SIDE_CTX[:] = side
    df = dataset.load_features()
    fs = feature_sets(df)
    preds, out = {}, None
    for name, (factory, feat) in LINEUP.items():
        names = [n for n in fs[feat] if not n.startswith(tuple(exclude_prefixes))] if exclude_prefixes else list(fs[feat])
        names = [n for n in names if not n.startswith(tuple(extra_prefixes))] if extra_prefixes else names
        names += [n for n in dataset.team_columns(df) if n.startswith(tuple(extra_prefixes)) and n not in names] if extra_prefixes else []
        p = walk_forward(factory, df, DEV_SEASONS, names)
        preds[name] = p["pred"].to_numpy()
        out = p
    ens = logit_mean(list(preds.values()))
    res = {n: round(metrics(out["target"], v)["log_loss"], 4) for n, v in preds.items()}
    m = metrics(out["target"], ens)
    res["ENSEMBLE"] = (round(m["log_loss"], 4), round(m["accuracy"], 4))
    return res


def main(cand):
    sh, sd, pref = CANDIDATES[cand][:3]
    excl = CANDIDATES[cand][3] if len(CANDIDATES[cand]) > 3 else []
    print("baseline ", run(BASE_SHARED, BASE_SIDE, [], pref), flush=True)
    print(cand.ljust(9), run(BASE_SHARED + sh, BASE_SIDE + sd, pref, excl), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
