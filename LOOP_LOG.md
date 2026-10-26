# Improvement loop log

Started 2026-10-26. Plan: LOOP_PLAN.md. One entry per item.

## 1. Live prediction log — done
- Added `src/live_log.py`. It's append-only: frozen pre-kickoff predictions for every game since 2026 week 6, plus results. It's stored at `docs/data/live_log.csv`, so it's published and committed by the existing scheduled tasks with no task changes needed.
- `build_site.py` refreshes it on every run.
- Excluded `2026_06_SEA_DEN`: its last stored prediction was recomputed after the game ended, before kickoff freezing existed. All week 7 predictions were confirmed frozen pre-kickoff (the Sunday refresh ran at 13:08 ET and kept earlier numbers for the 1 PM games).
- First reading: 26 graded live games. Model 15–11 (log-loss 0.715), Vegas 17–9 (0.701) on the same games. That's far too few games to mean anything yet.
- Verification: a site-only change, so I ran `build_site.py` and checked the data embeds. The model pipeline is unchanged.

## 2. "Since launch" record on the site — done
- The brief now has a "Live since launch" panel: model vs Vegas on the same live games, with log-loss, the game count, a small-sample caveat and a link to download `data/live_log.csv`.
- Current: 26 games, model 15–11 (0.715), Vegas 17–9 (0.701).
- Verified the render in the browser pane with no console errors.

## 3. Calibration layer — no gain, not adopted
- Built walk-forward ensemble predictions for 2016–2021 (`outputs/oof_ensemble.csv`, `src/exp_calibration.py`). Each dev season is calibrated only from earlier seasons.
- Dev log-loss: raw 0.6227. Platt with intercept 0.6250 (worse: it carries an outdated home-field bias from 2016–17 into later seasons, when home advantage had shrunk). Slope-only temperature 0.6233 (fitted slopes 1.11–1.23 confirm mild under-confidence, but correcting it doesn't help out of sample).

## 4. Stacking weights — no gain, not adopted
- Non-negative log-odds weights fit walk-forward on earlier seasons. Unshrunk weights favor ridge_margin and the Siamese net but swing by season: dev 0.6256. Shrunk toward equal weights: 0.6237 / 0.6231. The flat average wins at 0.6227.
- Takeaway: member strength isn't stable enough across seasons to exploit, so the flat log-odds average stays.
