# NFL Win Predictor — Overnight Report (2026-10-14)

## Bottom line

On **1,218 held-out games (2022 – 2026 week 5)** the final ensemble picked **66.3%** of winners,
with a log-loss of **0.624**. The model never saw these games during tuning.

| Model | Accuracy | Log-loss | Brier |
|---|---|---|---|
| **Ensemble (5 models)** | 66.3% | 0.6242 | 0.2163 |
| MLP neural net | 66.9% | 0.6245 | 0.2163 |
| Siamese neural net | 65.7% | 0.6274 | 0.2178 |
| Logistic regression | 67.0% | 0.6238 | 0.2161 |
| Ridge margin | 67.1% | 0.6242 | 0.2162 |
| Gradient boosting | 65.7% | 0.6309 | 0.2193 |
| *Vegas moneyline* | *67.4%* | *0.6103* | *0.2106* |
| *Elo rating* | *63.9%* | *0.6349* | *0.2214* |
| *Always pick home* | *55.0%* | *0.6881* | *0.2467* |

- **The model clearly beats the simple baselines.** It is +2.4 points over Elo and +11 over picking the home team.
- **It comes close to Vegas but doesn't beat it.** Accuracy is about 1 point lower, and log-loss is 0.014 higher. Vegas prices in news and market information that public stats don't contain. The model did beat Vegas in 2024 (71.9% vs 70.5%).
- **The results generalize.** The dev seasons (2018–21) scored 0.623 log-loss and the held-out seasons scored 0.624, so the model selection didn't overfit.

## Confidence

The networks are trained on log-loss, which is the "confidence-weighted payout" you asked for: confident and right earns a lot, confident and wrong costs a lot. On held-out games the confidence levels mean what they say:

| Model confidence | Games | Picked right |
|---|---|---|
| 50–55% | 242 | 54.5% |
| 55–60% | 261 | 63.7% |
| 60–65% | 216 | 63.7% |
| 65–70% | 168 | 68.3% |
| 70–75% | 150 | 73.3% |
| 75%+ | 181 | **81.2%** |

When the model and Vegas pick different winners (170 held-out games), Vegas was right 53.5% of the time and the model 45.9%; the remainder were ties. **Use the model to rank confidence and to flag games, and lean toward Vegas when the two disagree.**

## What mattered (dev seasons, drop-one-group ablation)

| Feature group | Log-loss worse without it |
|---|---|
| Starting QB rating & QB changes | +0.0034 |
| Season-long team form (opponent-adjusted EPA, margin) | +0.0021 |
| **Pregame roster ratings** (your idea) | +0.0013 |
| Momentum (last 2–3 games) | +0.0008 |
| Playoff stakes, Elo, ridge power ratings, rest/travel | ~0 (already covered by other features) |

- **Both of your roster ideas paid off.** Rating players and summing the pregame active roster adds information that nothing else covers, especially for injuries and roster turnover.
- **Momentum is real but small.** Long-term strength is worth about 3× more. When I tuned the power ratings, a 200-day memory beat a 56-day memory by a wide margin.
- **The QB features carry the most unique value.** The QB rating is a shrunk EPA per play that carries across teams and seasons, plus "different QB than last start" and "backup share" flags.

## How the roster ratings work

- **Player value** is each player's exponentially weighted snap share over his previous appearances (half-life 6 games). It uses offense snaps for offensive players and defense snaps for defenders. Coaches give snaps to their best players, so sustained playing time is a solid quality signal.
- **Rookies and players with little history** are shrunk toward a prior based on draft pick. Draft position also scales a separate "talent" score.
- **Team strength** sums player value over the **pregame active roster**, by unit (QB, OL, RB, WR, TE, DL, LB, DB), plus top-11 starter strength and experience.
- **Missing regulars** counts the snap share of players who played for the team in its last 4 games but aren't available now. This is how injuries and cuts show up.
- **Preseason priors** come out of the same calculation. At week 1, each player's rating comes from last season, even if he played for another team, so free agency, trades and draft classes are reflected automatically.
- **Availability is defined the same way for history and live picks:** on the 53-man roster and not listed Out or Doubtful. Using game-day inactives in training would have leaked information the model won't have when making real picks.

## Neural network findings

1. **Overfitting was the main problem.** With about 2,000–3,000 training games and around 250 features, the first networks overfit within about 5 epochs, and their scores swung by roughly 0.007 between runs. An explicit L2 penalty fixed it, strongest on the network's linear skip path, plus input dropout. After that the networks were stable.
2. **Multi-task training helps a little.** Predicting point margin alongside win/loss improved log-loss.
3. **The signal is mostly linear.** After regularization the MLP ties logistic regression. The Siamese matchup network and a GRU over each team's last 12 games were close but not better, and the GRU didn't improve the ensemble, so it's left out.
4. **Symmetry is built in.** Every game is trained from both sides, and predictions average the two views, so a model can't favor whichever team is listed first.

## 2026 week 6 picks

| Game | Pick | Confidence | Vegas (pick's side) | Note |
|---|---|---|---|---|
| ARI @ LA | LA | 79.1% | 86% | |
| LAC @ KC | KC | 78.4% | 80% | |
| WAS @ SF | SF | 78.0% | 78% | |
| BUF @ LV | BUF | 71.4% | 69% | LV: different QB than last start |
| BAL @ CLE | BAL | 70.9% | 68% | BAL: different QB than last start |
| NYJ @ NE | NE | 70.2% | 74% | |
| TEN @ IND | IND | 69.4% | 74% | |
| NO @ NYG | **NYG** | 61.6% | 49% | **differs from Vegas** |
| HOU @ JAX | JAX | 61.5% | 57% | |
| PIT @ TB | PIT | 54.8% | 55% | |
| CAR @ PHI | PHI | 52.7% | 51% | |
| SEA @ DEN (Thu) | SEA | 51.5% | 56% | SEA: different QB than last start |
| CHI @ ATL | **ATL** | 50.7% | 41% | **differs from Vegas** |
| DAL @ GB | **DAL** | 50.5% | 47% | **differs from Vegas** |

Week 6 injury designations weren't published yet when these picks were made. Re-run `python src/run_weekly.py` after Friday's reports.

## Limitations and next ideas

- **Missing information.** The biggest remaining gap is information Vegas has and the model doesn't: late injury news, weather forecasts, coaching changes, and market sentiment. Weather is left out entirely because forecasts weren't available in the data at prediction time.
- **Per-player production ratings didn't help.** I tested quality ratings on top of snap share: receiver and RB EPA, pass-rush pressure (sacks + hits), and coverage plays (INTs + passes defensed), summed over the available roster. Receiver quality alone correlates 0.28 with margin, but dev log-loss didn't move (0.62342 vs 0.62344), because team EPA form already carries the same information. The code stays in `players.py` (`rs_q_*` columns), but the features are excluded from the models.
- **Week 18 resting starters** is only approximated by the "clinched / bye locked" standings flags, which ignore tiebreakers.
- **2015 is the weakest training season** because player snap history only starts in 2014.
- **Possible improvements:** live weather forecasts, a proper playoff-odds simulation, and quality-based player ratings.

All experiment results are in `outputs/experiments.csv`. Per-game predictions are in `outputs/preds_dev.csv` and `outputs/preds_test.csv`.
