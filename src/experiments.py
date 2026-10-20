"""Dev-phase experiment runner. Appends one row per config to outputs/experiments.csv.

    python src/experiments.py <grid-name>
"""
import itertools
import sys
import time

import pandas as pd

import models as M
from backtest import DEV_SEASONS, feature_sets, summarize, walk_forward
from common import OUTPUTS
from dataset import load_features

GRIDS = {
    "features": [
        (m, fs, {}) for m in ("logreg", "mlp") for fs in ("all", "no_roster", "no_qb", "long_only", "compact")
    ],
    "logreg": [("logreg", fs, {"C": c}) for fs in ("all", "long_only", "compact")
               for c in (0.0005, 0.002, 0.005, 0.02)],
    "mlp": [("mlp", fs, {"hidden": h, "dropout": d, "wd": wd})
            for fs in ("long_only", "compact")
            for h, d, wd in itertools.product([(32,), (64, 32), (128, 64)], [0.3, 0.5], [1e-2, 1e-1])],
    "recency": [(m, fs, {"recency": r}) for m in ("logreg", "mlp") for fs in ("long_only", "compact")
                for r in (1.0, 0.9, 0.75)],
    "gbm": [("gbm", fs, {"lr": lr, "iters": it, "depth": d})
            for fs in ("long_only", "compact")
            for lr, it, d in [(0.02, 200, 2), (0.02, 400, 2), (0.02, 300, 3), (0.01, 600, 2)]],
    "siamese": [("siamese", fs, {"emb": e, "hidden": h, "dropout": d})
                for fs in ("long_only", "compact")
                for e, h, d in [(8, 32, 0.3), (16, 32, 0.5), (16, 64, 0.5), (32, 64, 0.5)]],
}

GRIDS["round2"] = (
    [("logreg", fs, {"C": c}) for fs in ("all", "compact", "core") for c in (0.0002, 0.0005, 0.002)]
    + [("mlp", fs, {"hidden": (128, 64), "dropout": 0.5, "wd": 0.1}) for fs in ("all", "compact", "core")]
    + [("gbm", fs, {"lr": 0.02, "iters": 200, "depth": 2}) for fs in ("compact", "core")]
    + [("siamese", fs, {"emb": 16, "hidden": 64, "dropout": 0.5}) for fs in ("all", "compact", "core")]
)

GRIDS["margin"] = (
    [("ridge", fs, {"alpha": a}) for fs in ("all", "compact") for a in (300, 1000, 3000, 10000)]
    + [("mlp", fs, {"hidden": (128, 64), "dropout": 0.5, "wd": 0.1, "margin_weight": mw})
       for fs in ("all", "compact") for mw in (0.0, 0.5, 2.0)]
    + [("siamese", "compact", {"emb": 16, "hidden": 64, "dropout": 0.5, "margin_weight": mw}) for mw in (0.5, 2.0)]
)

GRIDS["round3"] = [
    ("logreg", "all", {"C": 0.0005}), ("logreg", "all", {"C": 0.001}),
    ("ridge", "all", {"alpha": 10000}), ("ridge", "all", {"alpha": 5000}),
    ("mlp", "all", {"hidden": (128, 64), "dropout": 0.5, "wd": 0.1, "margin_weight": 0.5}),
    ("mlp", "all", {"hidden": (256, 128), "dropout": 0.6, "wd": 0.1, "margin_weight": 0.5}),
    ("gbm", "compact", {"lr": 0.02, "iters": 200, "depth": 2}),
    ("gbm", "all", {"lr": 0.02, "iters": 200, "depth": 2}),
    ("siamese", "compact", {"emb": 16, "hidden": 64, "dropout": 0.5, "margin_weight": 2.0}),
]

GRIDS["nn_epochs"] = [
    ("mlp", "all", {"hidden": (128, 64), "dropout": 0.5, "wd": 0.1, "margin_weight": mw, "epochs": e, "seeds": 10})
    for mw in (0.0, 0.5) for e in (5, 10, 20, 40)
]

GRIDS["nn_l2"] = [
    ("mlp", "all", {"hidden": h, "dropout": 0.5, "wd": 0.0, "l2": l2, "input_dropout": idp, "margin_weight": 0.5,
                    "lr": 2e-3, "batch": 256, "epochs": e, "seeds": 6})
    for l2 in (0.003, 0.01, 0.03) for h, idp, e in [((64,), 0.2, 40), ((128, 64), 0.2, 40), ((64,), 0.2, 80)]
]

GRIDS["nn_l2b"] = [
    ("mlp", "all", {"hidden": (64,), "dropout": 0.5, "l2": l2, "l2_skip": l2s, "input_dropout": 0.2,
                    "margin_weight": 0.5, "lr": 2e-3, "batch": 256, "epochs": 40, "seeds": 6})
    for l2, l2s in [(0.1, 0.1), (0.3, 0.3), (0.01, 0.3), (0.03, 0.3), (0.03, 1.0)]
]

GRIDS["siam_l2"] = [
    ("siamese", fs, {"emb": 16, "hidden": 64, "dropout": 0.5, "l2": l2, "l2_skip": 1.0, "margin_weight": 0.5,
                     "lr": 2e-3, "batch": 256, "epochs": 40, "seeds": 6})
    for fs in ("all", "compact") for l2 in (0.01, 0.03, 0.1)
]

GRIDS["gru"] = [
    ("gru", fs, {"emb": 16, "hidden": 64, "gru": g, "dropout": 0.5, "l2": l2, "l2_skip": 1.0, "margin_weight": 0.5,
                 "lr": 2e-3, "batch": 256, "epochs": 40, "seeds": 4})
    for fs in ("core", "all") for g, l2 in [(16, 0.01), (32, 0.03), (32, 0.1)]
]

from seq_model import GRU  # noqa: E402

CLASSES = {"logreg": M.LogReg, "gbm": M.GBM, "mlp": M.MLP, "siamese": M.Siamese, "ridge": M.RidgeMargin, "gru": GRU}


def run(grid):
    df = load_features()
    fsets = feature_sets(df)
    out = OUTPUTS / "experiments.csv"
    for model, fs, params in GRIDS[grid]:
        t = time.time()
        pred = walk_forward(lambda: CLASSES[model](**params), df, DEV_SEASONS, fsets[fs])
        o = summarize(pred)["overall"]
        row = {"grid": grid, "model": model, "features": fs, "params": repr(params),
               "log_loss": round(o["log_loss"], 5), "brier": round(o["brier"], 5),
               "accuracy": round(o["accuracy"], 4), "secs": round(time.time() - t)}
        print(row, flush=True)
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    for g in sys.argv[1:]:
        run(g)
