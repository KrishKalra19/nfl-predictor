# Model card: NFL win-probability ensemble

## What it does
Estimates the probability that each team wins an NFL game, using only information available before kickoff.
Outputs a home-win probability, a pick, an implied point spread and a short explanation ("Grounds").

## Intended use
- Personal analysis and entertainment: weekly picks, power rankings and playoff odds on the project site.
- A benchmark for human predictions (the site grades human picks against the model and the betting market).

Not intended for wagering decisions. It doesn't beat the betting market (see Evaluation), and it carries no
information about late-breaking news after its last refresh.

## Data
- [nflverse](https://github.com/nflverse): play-by-play (2014–present), schedules and results, weekly rosters, snap counts, injury reports, player IDs. Training uses 2015 onward; 2014 seeds the ratings.
- Vegas moneylines and spreads from the same schedule data are used **only** as a benchmark, never as inputs.

## Model
- ~250 pregame features per game:
  - opponent-adjusted EPA and success rates over short and long windows;
  - ridge power ratings and Elo;
  - a shrunk QB EPA rating plus QB-change flags;
  - roster strength over the pregame active roster;
  - rest, travel and body clock.
- Five members averaged in log-odds space:
  - an MLP (multi-task: win logit + point margin);
  - a Siamese matchup network;
  - L2 logistic regression;
  - ridge margin regression (probability via a normal CDF);
  - shallow gradient boosting.
- Each neural net is an 8-seed ensemble, heavily L2-regularized.
- Every game is trained from both sides, so predictions are exactly symmetric on a neutral field (tested).

## Evaluation
Walk-forward by season: each season is predicted by models trained only on earlier seasons.
Hyperparameters and features were chosen on 2018–21; 2022–26 (through week 5, 1,218 games) was scored after the lineup was locked.

| Held-out 2022–26 | Accuracy | Log-loss | Brier |
|---|---|---|---|
| Ensemble | 66.2% | 0.624 | 0.217 |
| Vegas moneyline | 67.4% | 0.610 | 0.211 |
| Elo | 63.9% | 0.635 | 0.221 |
| Always home | 55.0% | 0.688 | 0.247 |

- **Calibration:** held-out picks at 75%+ confidence won 81%; the 70–75% band won 73%. A temperature-scaling layer was tested and didn't improve out-of-sample log-loss.
- **Live:** every prediction since 2026 week 6 is frozen at kickoff and logged in `docs/data/live_log.csv`.

## Leakage controls
- Features are as of kickoff. Unit tests rebuild Elo, ratings, form, standings and Kalman ratings with later results removed and require identical values. These tests found and fixed a leak of whole-dataset league averages; the measured impact was 0.0002 log-loss.
- Player availability uses the Friday view (53-man roster minus Out/Doubtful), identical in backtests and live.
- Live models train only on weeks before the target week, and predictions freeze at kickoff.

## Limitations and known biases
- **It doesn't beat the market.** Blended with Vegas, the model gets about 8% of the weight. Public stats are largely priced in.
- **Late news** (Sunday inactives, surprise QB changes after the last refresh) isn't reflected.
- **Week 18** rest and motivation effects are only approximated, and NFL tiebreakers aren't modeled in the playoff odds.
- **Early-season uncertainty** is higher; ratings carry over from the prior season with regression to the mean.
- **Roster ratings measure playing time and draft capital, not play quality.** A per-player production version was tested and added nothing.
- **Explanations come from the linear member only.** They approximate, but don't exactly decompose, the full ensemble.
