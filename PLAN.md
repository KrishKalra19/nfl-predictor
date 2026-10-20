# NFL Predictor — Plan (agreed 2026-10-14)

## Goal
Neural network that outputs **win probability** for NFL games, built for week-to-week use.

## Decisions
- **Target:** home-team win probability. Trained/scored with log-loss + Brier
  (confidence-weighted: confidently right earns a lot, confidently wrong costs a lot).
- **Vegas lines:** benchmark only — never a model input.
- **Seasons:** 2015+ (modern era). Walk-forward evaluation: train on all prior seasons,
  test on the next season, repeat.
- **Data:** nflverse play-by-play, schedules, weekly rosters, snap counts, depth charts.
- **Leakage rule:** every feature for a game uses only info available before kickoff,
  including the *pregame roster* for that week.
- **Priority:** max accuracy.

## Features
- Pregame roster strength from custom player ratings (prior snap share, production,
  positional value) — drives early-season priors and injury/absence effects.
- QB changes (starter vs. usual starter, QB rating).
- Momentum / recency-weighted team form (EPA, success rate, pass/rush splits).
- Rest, bye, short week, travel distance, dome/outdoor, temperature, wind.
- Preseason priors: last season strength blended with roster changes, decaying over weeks 1–4+.

## Models
- Baselines: always-home, logistic regression, gradient boosting, Vegas implied prob.
- Neural nets: MLP, Siamese matchup net, GRU over recent games. Keep the best.

## Deliverables
1. This week's picks
2. Results dashboard (private artifact)
3. Clean reusable scripts (refresh data → retrain → predict)
4. Written report
