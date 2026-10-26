# Improvement loop log

Started 2026-10-26. Plan: LOOP_PLAN.md. One entry per item.

## 1. Live prediction log — done
- Added `src/live_log.py`. It's append-only: frozen pre-kickoff predictions for every game since 2026 week 6, plus results. It's stored at `docs/data/live_log.csv`, so it's published and committed by the existing scheduled tasks with no task changes needed.
- `build_site.py` refreshes it on every run.
- Excluded `2026_06_SEA_DEN`: its last stored prediction was recomputed after the game ended, before kickoff freezing existed. All week 7 predictions were confirmed frozen pre-kickoff (the Sunday refresh ran at 13:08 ET and kept earlier numbers for the 1 PM games).
- First reading: 26 graded live games. Model 15–11 (log-loss 0.715), Vegas 17–9 (0.701) on the same games. That's far too few games to mean anything yet.
- Verification: a site-only change, so I ran `build_site.py` and checked the data embeds. The model pipeline is unchanged.
