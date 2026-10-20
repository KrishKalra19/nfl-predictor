"""Models. Every model exposes fit(train_df, names) and predict(df) -> P(home team wins).

All neural nets are trained with binary cross-entropy (log-loss): a proper scoring rule, so
being confidently right is rewarded and confidently wrong is punished hard -- the model
is paid to be exactly as confident as the evidence allows.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from dataset import Prep, diff_features, expand_ties, make_xy, views

torch.set_num_threads(4)


def recency_weights(df, decay):
    last = df["season"].max()
    return decay ** (last - df["season"].to_numpy(dtype=float))


class SymmetricModel:
    """Fits on both views; predicts the average of P(home) and 1 - P(mirrored)."""

    def __init__(self, recency=1.0, **params):
        self.recency = recency
        self.params = params

    def fit(self, df, names):
        self.names = names
        t1, t2, sh, _ = views(df, names)
        self.prep = Prep().fit(t1, t2, sh)
        w = recency_weights(df, self.recency)
        data = make_xy(df, names, self.prep, augment=True)
        self._fit(*data, np.concatenate([w, w]).astype(np.float32))
        return self

    def predict(self, df):
        t1, t2, sh, home, _ = make_xy(df.assign(target=0.0), self.names, self.prep, augment=False)
        p = self._predict(t1, t2, sh, home)
        q = self._predict(t2, t1, sh, -home)
        return 0.5 * (p + (1 - q))


class LogReg(SymmetricModel):
    def _fit(self, t1, t2, sh, home, y, w):
        X = diff_features(t1, t2, sh, home)
        X, yy, tw = expand_ties(X, y)
        ww = np.concatenate([w[y != 0.5], w[y == 0.5], w[y == 0.5]]) * tw
        self.m = LogisticRegression(C=self.params.get("C", 0.05), max_iter=2000)
        self.m.fit(X, yy, sample_weight=ww)

    def _predict(self, t1, t2, sh, home):
        return self.m.predict_proba(diff_features(t1, t2, sh, home))[:, 1]


class RidgeMargin(SymmetricModel):
    """Predict point margin with ridge regression, convert to P(win) = Phi(margin / sigma)."""

    def _fit(self, t1, t2, sh, home, y, w):
        from scipy.optimize import minimize_scalar
        from scipy.stats import norm
        from sklearn.linear_model import Ridge
        X = diff_features(t1, t2, sh, home)
        self.m = Ridge(alpha=self.params.get("alpha", 1000.0)).fit(X, self._margin, sample_weight=w)
        mu = self.m.predict(X)

        def nll(s):
            p = np.clip(norm.cdf(mu / s), 1e-6, 1 - 1e-6)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        self.sigma = minimize_scalar(nll, bounds=(0.2, 3.0), method="bounded").x

    def fit(self, df, names):
        self.names = names
        t1, t2, sh, _ = views(df, names)
        self.prep = Prep().fit(t1, t2, sh)
        w = recency_weights(df, self.recency)
        *data, m = make_xy(df, names, self.prep, augment=True, with_margin=True)
        self._margin = m
        self._fit(*data, np.concatenate([w, w]))
        return self

    def _predict(self, t1, t2, sh, home):
        from scipy.stats import norm
        return norm.cdf(self.m.predict(diff_features(t1, t2, sh, home)) / self.sigma)


class GBM(SymmetricModel):
    def _fit(self, t1, t2, sh, home, y, w):
        X = diff_features(t1, t2, sh, home)
        X, yy, tw = expand_ties(X, y)
        ww = np.concatenate([w[y != 0.5], w[y == 0.5], w[y == 0.5]]) * tw
        p = self.params
        self.m = HistGradientBoostingClassifier(
            learning_rate=p.get("lr", 0.03), max_iter=p.get("iters", 300), max_depth=p.get("depth", 3),
            min_samples_leaf=p.get("leaf", 40), l2_regularization=p.get("l2", 1.0), random_state=0)
        self.m.fit(X, yy, sample_weight=ww)

    def _predict(self, t1, t2, sh, home):
        return self.m.predict_proba(diff_features(t1, t2, sh, home))[:, 1]


# ------------------------------------------------------------------ neural nets

class MLPNet(nn.Module):
    """Feed-forward net on the matchup difference vector."""

    def __init__(self, d_team, d_shared, hidden=(64, 32), dropout=0.3, input_dropout=0.0):
        super().__init__()
        self.in_drop = nn.Dropout(input_dropout)
        layers, d = [], d_team + d_shared + 1
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(d, 2)
        self.skip = nn.Linear(d_team + d_shared + 1, 2)  # linear path so the net is never worse than LR

    def forward(self, t1, t2, sh, home):
        """Returns (n, 2): column 0 = win logit, column 1 = scaled margin."""
        x = self.in_drop(torch.cat([t1 - t2, sh, home[:, None]], 1))
        return self.out(self.body(x)) + self.skip(x)


class SiameseNet(nn.Module):
    """Shared team encoder -> matchup head. Learns 'team strength' embeddings and how they interact."""

    def __init__(self, d_team, d_shared, emb=16, hidden=32, dropout=0.3):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_team, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, emb), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * emb + d_shared + 1, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 2))
        self.skip = nn.Linear(d_team, 2, bias=False)
        self.hfa = nn.Linear(1, 2)

    def forward(self, t1, t2, sh, home):
        e1, e2 = self.enc(t1), self.enc(t2)
        z = torch.cat([e1 - e2, e1 * e2, sh, home[:, None]], 1)
        return self.head(z) + self.skip(t1 - t2) + self.hfa(home[:, None])


class TorchModel(SymmetricModel):
    net_cls = MLPNet

    def _build(self, d_team, d_shared):
        p = self.params
        kw = {k: p[k] for k in ("hidden", "dropout", "emb", "input_dropout") if k in p}
        return self.net_cls(d_team, d_shared, **kw)

    def _train(self, data, epochs, seed, val=None):
        t1, t2, sh, home, y, mg, w = [torch.tensor(a) for a in data]
        alpha = self.params.get("margin_weight", 0.0)
        l2 = self.params.get("l2", 0.0)
        l2_skip = self.params.get("l2_skip", l2)
        torch.manual_seed(seed)
        net = self._build(t1.shape[1], sh.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=self.params.get("lr", 1e-3),
                                weight_decay=self.params.get("wd", 1e-2))
        n, bs = len(y), self.params.get("batch", 128)
        g = torch.Generator().manual_seed(seed)
        best, best_ep, hist = 1e9, epochs, []
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                out = net(t1[b], t2[b], sh[b], home[b])
                bce = nn.functional.binary_cross_entropy_with_logits(out[:, 0], y[b], reduction="none")
                mse = (out[:, 1] - mg[b]) ** 2
                loss = ((bce + alpha * mse) * w[b]).mean()
                if l2:
                    pen = sum((q ** 2).sum() for nm, q in net.named_parameters()
                              if "weight" in nm and not nm.startswith("skip"))
                    loss = loss + l2 * pen
                if l2_skip:
                    loss = loss + l2_skip * (net.skip.weight ** 2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            if val is not None:
                vl = self._val_loss(net, val)
                hist.append(vl)
                if vl < best - 1e-5:
                    best, best_ep = vl, ep + 1
                elif ep + 1 - best_ep >= self.params.get("patience", 25):
                    break
        return net, best_ep, hist

    @staticmethod
    def _val_loss(net, val):
        net.eval()
        with torch.no_grad():
            t1, t2, sh, home, y = [torch.tensor(a) for a in val[:5]]
            return nn.functional.binary_cross_entropy_with_logits(net(t1, t2, sh, home)[:, 0], y).item()

    def fit(self, df, names):
        self.names = names
        p = self.params
        # 1) pick epoch count by early stopping on the most recent training season
        last = df["season"].max()
        inner_tr, inner_va = df[df["season"] < last], df[df["season"] == last]
        epochs = p.get("epochs")
        if epochs is None:
            t1, t2, sh, _ = views(inner_tr, names)
            prep = Prep().fit(t1, t2, sh)
            w = recency_weights(inner_tr, self.recency)
            tr = make_xy(inner_tr, names, prep, augment=True, with_margin=True) + (np.concatenate([w, w]).astype(np.float32),)
            va = make_xy(inner_va, names, prep, augment=False)
            eps = [self._train(tr, p.get("max_epochs", 300), s, val=va)[1] for s in range(2)]
            epochs = int(np.mean(eps))
        self.epochs_ = epochs
        # 2) refit on all training seasons, ensemble of seeds
        t1, t2, sh, _ = views(df, names)
        self.prep = Prep().fit(t1, t2, sh)
        w = recency_weights(df, self.recency)
        data = make_xy(df, names, self.prep, augment=True, with_margin=True) + (np.concatenate([w, w]).astype(np.float32),)
        self.nets = [self._train(data, epochs, 100 + s)[0] for s in range(p.get("seeds", 5))]
        return self

    def _predict(self, t1, t2, sh, home):
        args = [torch.tensor(a) for a in (t1, t2, sh, home)]
        out = []
        for net in self.nets:
            net.eval()
            with torch.no_grad():
                out.append(torch.sigmoid(net(*args)[:, 0]).numpy())
        return np.mean(out, axis=0)


class MLP(TorchModel):
    net_cls = MLPNet


class Siamese(TorchModel):
    net_cls = SiameseNet


class Ensemble:
    def __init__(self, members, weights=None):
        self.members = members
        self.weights = weights

    def fit(self, df, names):
        for m in self.members:
            m.fit(df, names)
        return self

    def predict(self, df):
        P = np.array([m.predict(df) for m in self.members])
        # average in log-odds space
        L = np.log(np.clip(P, 1e-6, 1 - 1e-6) / (1 - np.clip(P, 1e-6, 1 - 1e-6)))
        w = np.ones(len(P)) / len(P) if self.weights is None else np.asarray(self.weights)
        return 1 / (1 + np.exp(-(w[:, None] * L).sum(0)))
