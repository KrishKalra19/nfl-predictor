"""Dev-season test of a longer training history (built with NFL_FIRST_SEASON=2006 into data/processed_2006).

    NFL_PROCESSED=data/processed_2006 python src/exp_history.py
Prints dev (2018-21) ensemble log-loss for several first training seasons. Snap counts start in 2012,
so roster features are imputed for earlier games.
"""
import dataset
from backtest import DEV_SEASONS, feature_sets, metrics, walk_forward
from config import LINEUP
from evaluate import logit_mean

STARTS = [2015, 2012, 2010, 2007]


def main():
    df = dataset.load_features()
    fs = feature_sets(df)
    for start in STARTS:
        preds, out = [], None
        res = {}
        for name, (factory, feat) in LINEUP.items():
            out = walk_forward(factory, df, DEV_SEASONS, fs[feat], train_start=start)
            preds.append(out["pred"].to_numpy())
            res[name] = round(metrics(out["target"], preds[-1])["log_loss"], 4)
        m = metrics(out["target"], logit_mean(preds))
        res["ENSEMBLE"] = (round(m["log_loss"], 4), round(m["accuracy"], 4))
        print(f"train from {start}", res, flush=True)


if __name__ == "__main__":
    main()
