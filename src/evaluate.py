"""Walk-forward evaluation of the final lineup + ensembles + baselines.

    python src/evaluate.py --phase dev    # 2018-2021 (model selection seasons)
    python src/evaluate.py --phase test   # 2022-2026 (held out; run once)

Writes outputs/preds_<phase>.csv (one row per game, every model) and outputs/metrics_<phase>.json.
"""
import argparse
import json

import numpy as np

from backtest import DEV_SEASONS, TEST_SEASONS, baselines, feature_sets, metrics, walk_forward
from common import OUTPUTS
from config import LINEUP, NEURAL
from dataset import load_features


def logit_mean(P):
    P = np.clip(np.asarray(P), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-np.mean(np.log(P / (1 - P)), axis=0)))


def by_season(y, p, season):
    return {int(s): metrics(y[season == s], p[season == s]) for s in np.unique(season)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "test"], default="dev")
    args = ap.parse_args()
    seasons = DEV_SEASONS if args.phase == "dev" else TEST_SEASONS

    df = load_features()
    fsets = feature_sets(df)
    preds = None
    for name, (factory, fs) in LINEUP.items():
        p = walk_forward(factory, df, seasons, fsets[fs])
        if preds is None:
            preds = p.drop(columns="pred")
            preds = preds.merge(df[["game_id", "game_type", "spread_line", "margin", "ctx_neutral", "h_elo", "a_elo"]],
                                on="game_id", how="left")
        preds[name] = p["pred"].to_numpy()
        print(f"{name:13s} {metrics(preds['target'], preds[name])}", flush=True)

    preds["ensemble"] = logit_mean([preds[n] for n in LINEUP])
    preds["nn_ensemble"] = logit_mean([preds[n] for n in NEURAL])

    y, s = preds["target"].to_numpy(), preds["season"].to_numpy()
    out = {"seasons": seasons, "models": {}, "baselines": baselines(df, seasons)}
    for name in list(LINEUP) + ["ensemble", "nn_ensemble"]:
        p = preds[name].to_numpy()
        out["models"][name] = {"overall": metrics(y, p), "by_season": by_season(y, p, s)}
    has_v = preds["vegas_prob"].notna()
    out["vegas_same_games"] = metrics(y[has_v], preds.loc[has_v, "vegas_prob"])
    out["ensemble_same_games"] = metrics(y[has_v], preds.loc[has_v, "ensemble"])

    for name in ("ensemble", "nn_ensemble"):
        print(f"{name:13s} {out['models'][name]['overall']}")
    for b, r in out["baselines"].items():
        print(f"{b:13s} {r['overall']}")

    OUTPUTS.mkdir(exist_ok=True)
    preds.to_csv(OUTPUTS / f"preds_{args.phase}.csv", index=False)
    (OUTPUTS / f"metrics_{args.phase}.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
