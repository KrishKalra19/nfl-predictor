"""Dev-season test of adding or swapping ensemble members (bigger / sequence architectures).

    python src/exp_members.py
Prints dev log-loss of each new member alone, and of the lineup ensemble with it added.
"""
import numpy as np

import dataset
import models as M
from backtest import DEV_SEASONS, feature_sets, metrics, walk_forward
from config import LINEUP, NN_COMMON
from evaluate import logit_mean
import seq_model

NEW = {
    "transformer": (lambda: seq_model.Transformer(emb=16, hidden=32, dropout=0.5, gru=32, l2=0.03, l2_skip=1.0,
                                                  **NN_COMMON), "compact"),
    "mlp_deep": (lambda: M.MLP(hidden=(128, 64), dropout=0.5, l2=0.03, l2_skip=1.0, input_dropout=0.2, **NN_COMMON), "all"),
}


def main():
    df = dataset.load_features()
    fs = feature_sets(df)
    preds, out = {}, None
    for name, (factory, feat) in {**LINEUP, **NEW}.items():
        out = walk_forward(factory, df, DEV_SEASONS, fs[feat])
        preds[name] = out["pred"].to_numpy()
        print(name.ljust(12), round(metrics(out["target"], preds[name])["log_loss"], 4), flush=True)
    y = out["target"]
    base = [preds[n] for n in LINEUP]
    print("ENSEMBLE    ", round(metrics(y, logit_mean(base))["log_loss"], 4))
    for n in NEW:
        print(f"+{n}".ljust(12), round(metrics(y, logit_mean(base + [preds[n]]))["log_loss"], 4))
        sw = [preds[k] if k != "mlp" else preds[n] for k in LINEUP]
        print(f"mlp->{n}".ljust(12), round(metrics(y, logit_mean(sw))["log_loss"], 4))
    np.save("../outputs/exp/member_preds.npy", preds)


if __name__ == "__main__":
    main()
