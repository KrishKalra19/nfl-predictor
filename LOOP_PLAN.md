# Improvement loop plan (8+ hours)

The loop works through this list top to bottom. Each iteration takes the first unchecked item,
finishes it (or marks it blocked with a reason), logs it in `LOOP_LOG.md`, commits, and moves on.
Priorities: accuracy that's real, then resume value, then polish.

## Ground rules (every iteration)

1. **Honest evaluation.**
   - Model changes are judged on the dev seasons (2018–21) only.
   - A change is adopted only if dev log-loss improves by at least 0.002, or it ties and simplifies.
   - After adopting, re-run the 2022–26 evaluation once and label it *post-hoc* in REPORT.md. The original held-out numbers stay recorded.
   - Live 2026 games, from the first frozen prediction onward, are the only truly fresh test.
2. **Never break the live system.**
   - Before any push, run `python src/run_weekly.py --no-download` and `python src/build_site.py` end to end, and check the site renders with no console errors.
   - Scheduled tasks also push to `main`, so always `git pull --rebase` before pushing. Never force-push.
3. **Frozen predictions stay frozen.** Games that have kicked off keep their pre-kickoff numbers, and history files are never rewritten.
4. **Privacy.** Never write under `data/users/` in the pick sheet. No surname in site copy. Krish's notes are never published.
5. **Small commits**, one item per commit, with the Co-Authored-By trailer.
6. **Time box.** Stop an item after about 90 minutes. Log what was learned and mark it `[~]` partial or `[x] no gain`.
7. **No new accounts, paid services or API keys.** Only free, keyless data sources.

## Phase 1 — Live track record and calibration (≈1.5 h)

- [x] **1. Live prediction log.** Append every frozen pre-kickoff prediction to `outputs/live_log.csv`: game, timestamp, model probability, Vegas probability, result once known. Backfill weeks 6–7 from the existing frozen picks files. This is the fresh out-of-sample record.
- [x] **2. "Since launch" record on the site.** Show the live 2026 record next to the backtest in The brief, as model vs Vegas on the same games, with the game count shown prominently.
- [x] **3. Calibration layer** (no gain).** The model is slightly under-confident (prediction spread 0.168 vs Vegas 0.191). Fit temperature or Platt scaling on dev predictions and adopt it only if dev log-loss improves.
- [x] **4. Stacking weights** (no gain).** Learn ensemble member weights on dev predictions instead of a flat average. Adopt only if dev improves.

## Phase 2 — New information the model doesn't have (≈3 h)

- [x] **5. Weather.** (no gain for winners)
  - Training: kickoff-hour wind, temperature and precipitation from Open-Meteo's free historical archive (no key), keyed by stadium coordinates.
  - Live: Open-Meteo's forecast for upcoming games.
  - Domes get zeros.
  - Expect small gains, and test on dev.
- [x] **6. Coaching.** (no gain) Head coach from `games.csv`: new-coach flag, coach tenure, and the coach's prior win rate with shrinkage.
- [x] **7. Kicking and special teams quality.** (no gain) The kicker's field-goal percentage over expected (from play-by-play), weighted by availability, plus punt and return EPA.
- [x] **8. Kalman-filter team strength.** (no model gain; reused for rankings) A state-space rating per team (offense and defense EPA as latent states with weekly drift), as a principled replacement for or addition to the ridge ratings. This also has strong resume value.
- [x] **9. QB model upgrade.** (near miss: −0.0006, under bar) Opponent-adjust QB EPA, add a backup-QB quality prior based on the snap-weighted QB2, and test an age curve.
- [x] **10. Proper playoff-stakes simulation.** (site feature) Monte Carlo the remaining schedule each week (using the model's own probabilities) to get each team's playoff odds and whether it has clinched a seed. Replaces the crude stakes flags and also feeds a site feature (#15).

After phase 2: re-tune the lineup on dev with any adopted features, run the post-hoc test evaluation, and update REPORT.md and the site's brief numbers.

## Phase 3 — Website features (≈2.5 h)

- [x] **11. "Why this pick" on each case.** Expand a case card to show the top 3–4 factors behind the ruling: the logistic model's coefficient × feature difference, in plain words like "QB edge: CIN" or "Rest: +3 days BAL". Explainability is the biggest resume win.
- [ ] **12. Model spread line.** Convert the ridge-margin model into a predicted point spread on each case ("Model line: BAL −2.5"), shown next to Vegas's.
- [ ] **13. Firm rankings page.** Power rankings for all 32 teams (rating, offense and defense EPA rank, QB, trend arrow since last week), styled as "The firm's client list". Ships as a second page, `docs/rankings.html`.
- [ ] **14. Weekly recap.** Auto-generated each week from data: biggest upset, the model's best and worst calls, and Krish vs the model that week. No made-up narrative.
- [ ] **15. Playoff odds table** from #10.
- [ ] **16. Polish.** Favicon (the K&A seal), social preview image and Open Graph tags, accessibility pass (contrast, focus, landmarks), mobile check at 375px, a print-free resume-friendly layout for The brief, and a footer link to the repo.

## Phase 4 — Engineering quality (≈1.5 h, high resume value)

- [ ] **17. Tests.**
  - Leakage tests: every feature row for a game equals what you'd compute with all later games deleted. Run on a sample of games.
  - Symmetry tests: P(A beats B) + P(B beats A) = 1.
  - Freeze tests: predictions for started games don't change.
- [ ] **18. GitHub Actions CI.** Run lint (ruff) and the tests on every push, with a CI badge in the README.
- [ ] **19. Repo polish.**
  - README rewrite as a project landing page: results, an architecture diagram in mermaid, how to run, and a link to the site.
  - Pin versions in `requirements.txt`.
  - Add a LICENSE (MIT).
  - Remove scratch outputs like `exp_log1.txt` and `dev_preds.csv`.
- [ ] **20. Model card** (`MODEL_CARD.md`): data, intended use, evaluation, limitations, known biases.

## Phase 5 — Human alpha (only if time remains, ≈0.5 h)

- [ ] **21. "Krish + Model" blend.** After kickoff, show what a 50/50 log-odds blend of Krish and the model would have scored, as a teaser for the stacking idea. It stays descriptive until there are about 200 picks.
- [ ] **22. Pick analysis by reason tag** (accuracy and log-loss per tag). Only displayed once a tag has at least 30 graded picks.

## Blocked: needs Krish

- Picks on the website itself: needs a free Supabase project URL and anon key.
- Inviting Andrew to the repo: needs his GitHub username.
- Changing the scheduled tasks' times or adding new ones: ask first.

## Stop conditions

Stop and report when any of these is true:
- Every item is done, blocked or time-boxed out.
- A change would require breaking a ground rule.
- The live pipeline fails in a way the loop can't fix without changing model logic mid-week.

The final summary goes in `LOOP_LOG.md` and the chat. It covers what shipped, dev and post-hoc numbers before and after, and what's blocked.
