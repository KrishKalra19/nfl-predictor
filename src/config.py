"""Final model lineup, chosen on the 2018-2021 dev seasons (see outputs/experiments.csv)."""
import models as M

NN_COMMON = {"margin_weight": 0.5, "lr": 2e-3, "batch": 256, "epochs": 40, "seeds": 8}

# name -> (factory, feature set)
LINEUP = {
    "logreg": (lambda: M.LogReg(C=0.0005), "all"),
    "ridge_margin": (lambda: M.RidgeMargin(alpha=10000), "all"),
    "gbm": (lambda: M.GBM(lr=0.02, iters=200, depth=2), "compact"),
    "mlp": (lambda: M.MLP(hidden=(64,), dropout=0.5, l2=0.03, l2_skip=1.0, input_dropout=0.2, **NN_COMMON), "all"),
    "siamese": (lambda: M.Siamese(emb=16, hidden=64, dropout=0.5, l2=0.1, l2_skip=1.0, **NN_COMMON), "all"),
}
NEURAL = ("mlp", "siamese")
