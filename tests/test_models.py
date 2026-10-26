"""Model contract tests on synthetic matchups: symmetry on a neutral field, valid probabilities."""
import numpy as np
import pandas as pd
import pytest

import dataset
import models as M

NAMES = [f"f{i}" for i in range(6)]


def matchup_frame(n=400, seed=0, neutral=False):
    rng = np.random.default_rng(seed)
    d = {"season": rng.integers(2018, 2022, n), "ctx_neutral": np.ones(n) if neutral else np.zeros(n)}
    for side in ("h", "a"):
        for f in NAMES:
            d[f"{side}_{f}"] = rng.normal(size=n)
        for c in dataset.SIDE_CTX:
            d[f"ctx_{side}_{c}"] = rng.normal(size=n)
    for c in dataset.SHARED:
        d[c] = rng.normal(size=n)
    df = pd.DataFrame(d)
    signal = (df[[f"h_{f}" for f in NAMES]].to_numpy() - df[[f"a_{f}" for f in NAMES]].to_numpy()) @ rng.normal(size=len(NAMES))
    df["margin"] = 3 * signal + 2 + rng.normal(0, 13, n)
    df["target"] = np.where(df["margin"] > 0, 1.0, 0.0)
    return df


def swap_sides(df):
    out = df.copy()
    for f in NAMES:
        out[f"h_{f}"], out[f"a_{f}"] = df[f"a_{f}"], df[f"h_{f}"]
    for c in dataset.SIDE_CTX:
        out[f"ctx_h_{c}"], out[f"ctx_a_{c}"] = df[f"ctx_a_{c}"], df[f"ctx_h_{c}"]
    return out


FAST = {"epochs": 3, "seeds": 1, "batch": 128}


@pytest.mark.parametrize("make", [
    lambda: M.LogReg(C=0.05),
    lambda: M.RidgeMargin(alpha=100),
    lambda: M.GBM(iters=30),
    lambda: M.MLP(hidden=(16,), dropout=0.1, **FAST),
    lambda: M.Siamese(emb=4, hidden=8, dropout=0.1, **FAST),
], ids=["logreg", "ridge", "gbm", "mlp", "siamese"])
def test_neutral_site_symmetry(make):
    train = matchup_frame(400, seed=1)
    test = matchup_frame(60, seed=2, neutral=True)
    model = make().fit(train, NAMES)
    p = model.predict(test)
    q = model.predict(swap_sides(test))
    assert np.all((p > 0) & (p < 1))
    np.testing.assert_allclose(p + q, 1.0, atol=1e-5)


def test_home_field_is_learned_and_signed():
    """With a home edge in the data, the home side of an otherwise even game is favored."""
    train = matchup_frame(2000, seed=3)
    model = M.LogReg(C=1.0).fit(train, NAMES)
    even = matchup_frame(50, seed=4)
    for f in NAMES:
        even[f"a_{f}"] = even[f"h_{f}"]
    for c in dataset.SIDE_CTX:
        even[f"ctx_a_{c}"] = even[f"ctx_h_{c}"]
    assert (model.predict(even) > 0.5).all()
