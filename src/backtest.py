"""Walk-forward backtest: for each test season, train on all earlier seasons, predict that season.

    python src/backtest.py --phase dev   # 2018-2021: used for model/feature selection
    python src/backtest.py --phase test  # 2022-2026: final, untouched evaluation
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from common import OUTPUTS
from dataset import load_features, team_columns

DEV_SEASONS = [2018, 2019, 2020, 2021]
TEST_SEASONS = [2022, 2023, 2024, 2025, 2026]
TRAIN_START = 2015


def metrics(y, p):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    decided = y != 0.5
    return {
        "n": int(len(y)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p[decided] > 0.5) == (y[decided] == 1))),
    }


def walk_forward(make_model, df, seasons, names, train_start=TRAIN_START):
    played = df[df["target"].notna()]
    preds = []
    for s in seasons:
        tr = played[(played["season"] >= train_start) & (played["season"] < s)]
        te = played[played["season"] == s]
        m = make_model().fit(tr, names)
        preds.append(te[["game_id", "season", "week", "home_team", "away_team", "target", "vegas_prob"]]
                     .assign(pred=m.predict(te)))
    return pd.concat(preds, ignore_index=True)


def summarize(pred):
    out = {"overall": metrics(pred["target"], pred["pred"])}
    for s, d in pred.groupby("season"):
        out[int(s)] = metrics(d["target"], d["pred"])
    return out


def feature_sets(df):
    # Not adopted (no dev gain) -- see LOOP_LOG.md. New candidates stay fenced out until they earn a place.
    NOT_ADOPTED = ("rs_q_", "coach_")
    allf = [f for f in team_columns(df) if not f.startswith(NOT_ADOPTED)]
    return {
        "all": allf,
        "no_roster": [f for f in allf if not f.startswith("rs_")],
        "no_qb": [f for f in allf if not f.startswith("qb_")],
        "long_only": [f for f in allf if not f.startswith("s_") and not f.startswith("trend_")],
        "compact": [f for f in allf if f.startswith(("l_adj", "l_off_epa", "l_def_epa", "l_margin", "elo",
                                                      "qb_", "rs_off_top11", "rs_def_top11", "rs_missing",
                                                      "rs_off_talent", "rs_def_talent", "s_adj", "trend_",
                                                      "last3", "streak", "l_st_epa", "l_win", "pr_", "prm_", "stakes_"))],
        "core": [f for f in allf if f.startswith(("pr_", "prm_", "elo", "qb_", "rs_off_top11", "rs_def_top11",
                                                   "rs_missing", "rs_off_talent", "rs_def_talent", "rs_exp",
                                                   "l_st_epa", "streak", "last3", "stakes_"))],
    }


def baselines(df, seasons):
    played = df[df["target"].notna() & df["season"].isin(seasons)]
    res = {}
    home = played.assign(pred=0.5 + 0.0 * played["season"])
    # always-home: use historical home win rate as the probability
    res["always_home"] = summarize(home.assign(pred=np.where(home["ctx_neutral"] == 1, 0.5, 0.565)))
    res["vegas"] = summarize(played.assign(pred=played["vegas_prob"]))
    elo_p = 1 / (1 + 10 ** (-(played["h_elo"] + np.where(played["ctx_neutral"] == 1, 0, 40) - played["a_elo"]) / 400))
    res["elo"] = summarize(played.assign(pred=elo_p))
    return res


def main():
    import models as M
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "test"], default="dev")
    ap.add_argument("--models", default="logreg,gbm,mlp,siamese")
    ap.add_argument("--features", default="all")
    args = ap.parse_args()
    seasons = DEV_SEASONS if args.phase == "dev" else TEST_SEASONS

    df = load_features()
    names = feature_sets(df)[args.features]
    zoo = {
        "logreg": lambda: M.LogReg(C=0.02),
        "gbm": lambda: M.GBM(),
        "mlp": lambda: M.MLP(),
        "siamese": lambda: M.Siamese(),
    }
    results = {"baselines": baselines(df, seasons)}
    for name in args.models.split(","):
        t = time.time()
        pred = walk_forward(zoo[name], df, seasons, names)
        results[name] = summarize(pred)
        o = results[name]["overall"]
        print(f"{name:10s} ll={o['log_loss']:.4f} brier={o['brier']:.4f} acc={o['accuracy']:.4f} ({time.time() - t:.0f}s)")
    for b, r in results["baselines"].items():
        o = r["overall"]
        print(f"{b:10s} ll={o['log_loss']:.4f} brier={o['brier']:.4f} acc={o['accuracy']:.4f}")
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / f"backtest_{args.phase}_{args.features}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
