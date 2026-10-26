"""Plain-language reasons behind each pick.

The ensemble's logistic-regression member is linear in (team1 - team2) feature differences, so its
log-odds split exactly into per-feature contributions. Contributions are summed into readable groups;
the biggest groups (with the team each one favors) become the "why" shown on the site.
Shared game context (week, division game...) is skipped: in the symmetric model it doesn't favor a side.
"""
import json

from dataset import SIDE_CTX, make_xy

GROUPS = [
    ("Quarterback", ("qb_",)),
    ("Roster & injuries", ("rs_",)),
    ("Special teams", ("l_st_epa", "s_st_epa")),
    ("Team strength rating", ("pr_", "prm_", "elo")),
    ("Recent form", ("s_", "trend_", "streak", "last3")),
    ("Offense this season", ("l_off_", "l_adj_off")),
    ("Defense this season", ("l_def_", "l_adj_def")),
    ("Results this season", ("l_margin", "l_adj_margin", "l_win", "l_pts_", "l_to_margin", "season_win_pct", "games_played")),
    ("Rest & travel", ("ctx_",)),
]


def group_of(name):
    for label, prefixes in GROUPS:
        if name.startswith(prefixes):
            return label
    return None


def explain(logreg, games, top=3, min_logodds=0.04):
    """logreg: a fitted models.LogReg. Returns a JSON string per game:
    [{"factor": label, "favors": team, "strength": "strong|moderate|slight", "logodds": x}, ...]."""
    t1, t2, sh, home, _ = make_xy(games.assign(target=0.0), logreg.names, logreg.prep, augment=False)
    coef = logreg.m.coef_[0]
    nt = t1.shape[1]
    labels = list(logreg.names) + [f"ctx_{c}" for c in SIDE_CTX]
    contrib = (t1 - t2) * coef[:nt]                   # home-view log-odds contributions
    home_c = home * coef[-1]
    groups = [group_of(n) for n in labels]
    out = []
    for i, g in enumerate(games.itertuples()):
        agg = {}
        for j, grp in enumerate(groups):
            if grp:
                agg[grp] = agg.get(grp, 0.0) + contrib[i, j]
        if home[i] != 0:
            agg["Home field"] = float(home_c[i])
        items = sorted(agg.items(), key=lambda kv: -abs(kv[1]))
        reasons = []
        for label, v in items[:top]:
            if abs(v) < min_logodds:
                continue
            a = abs(v)
            reasons.append({"factor": label, "favors": g.home_team if v > 0 else g.away_team,
                            "strength": "strong" if a >= 0.25 else "moderate" if a >= 0.1 else "slight",
                            "logodds": round(float(v), 3)})
        out.append(json.dumps(reasons))
    return out
