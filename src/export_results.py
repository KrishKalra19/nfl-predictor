"""Export backtest results for the site/dashboard from outputs/preds_{dev,test}.csv.

    python src/export_results.py

The held-out set is fixed at the games available when the lineup was locked (2022 through 2026 week 5,
1,218 games); later 2026 games belong to the live log instead. Writes outputs/dashboard/*.csv.
"""
import numpy as np
import pandas as pd

from backtest import metrics
from common import OUTPUTS

HELDOUT_END = (2026, 5)  # last (season, week) in the held-out test set
D = OUTPUTS / "dashboard"
NAMES = {"ensemble": "Ensemble (5 models)", "logreg": "Logistic regression", "ridge_margin": "Ridge margin",
         "gbm": "Gradient boosting", "mlp": "MLP neural net", "siamese": "Siamese neural net",
         "nn_ensemble": "Neural nets only"}
KIND = {"ensemble": "Ensemble", "nn_ensemble": "Neural", "mlp": "Neural", "siamese": "Neural",
        "logreg": "Classical", "ridge_margin": "Classical", "gbm": "Classical"}


def load(phase):
    d = pd.read_csv(OUTPUTS / f"preds_{phase}.csv")
    if phase == "test":
        d = d[(d["season"] < HELDOUT_END[0]) | (d["week"] <= HELDOUT_END[1])]
    return d


def baselines(d):
    home = np.where(d["ctx_neutral"] == 1, 0.5, 0.565)
    elo = 1 / (1 + 10 ** (-(d["h_elo"] + np.where(d["ctx_neutral"] == 1, 0, 40) - d["a_elo"]) / 400))
    return {"Vegas moneyline": d["vegas_prob"], "Elo rating": elo, "Always pick home": pd.Series(home, index=d.index)}


def main():
    D.mkdir(exist_ok=True)
    rows, seasons = [], []
    for phase in ("test", "dev"):
        d = load(phase)
        for k, label in NAMES.items():
            o = metrics(d["target"], d[k])
            rows.append({"phase": phase, "model": label, "kind": KIND[k], "games": o["n"], "accuracy": round(o["accuracy"], 4),
                         "log_loss": round(o["log_loss"], 4), "brier": round(o["brier"], 4)})
        for label, p in baselines(d).items():
            ok = p.notna()
            o = metrics(d.loc[ok, "target"], p[ok])
            rows.append({"phase": phase, "model": label, "kind": "Baseline", "games": o["n"], "accuracy": round(o["accuracy"], 4),
                         "log_loss": round(o["log_loss"], 4), "brier": round(o["brier"], 4)})
        b = baselines(d)
        for s, g in d.groupby("season"):
            for series, p in (("Ensemble", g["ensemble"]), ("Vegas", b["Vegas moneyline"][g.index]), ("Elo", b["Elo rating"][g.index])):
                o = metrics(g["target"], p)
                seasons.append({"season": int(s), "series": series, "accuracy": round(o["accuracy"], 4),
                                "log_loss": round(o["log_loss"], 4), "games": o["n"], "phase": phase})
    pd.DataFrame(rows).to_csv(D / "model_scores.csv", index=False)
    pd.DataFrame(seasons).sort_values(["series", "season"]).to_csv(D / "season_scores.csv", index=False)

    t = load("test")
    p = t["ensemble"]
    t = t.assign(conf=np.maximum(p, 1 - p), correct=((p > 0.5) == (t["target"] == 1)).astype(float))
    t.loc[t["target"] == 0.5, "correct"] = np.nan
    labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75%+"]
    t["tier"] = pd.cut(t["conf"], [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1.0], labels=labels, include_lowest=True)
    tiers = t.groupby("tier", observed=True).agg(games=("correct", "size"), accuracy=("correct", "mean"),
                                                 avg_confidence=("conf", "mean")).reset_index().round(4)
    tiers["order"] = range(len(tiers))
    tiers.to_csv(D / "confidence_tiers.csv", index=False)
    print(pd.DataFrame(rows)[lambda x: x.phase == "test"][["model", "games", "accuracy", "log_loss"]].to_string(index=False))


if __name__ == "__main__":
    main()
