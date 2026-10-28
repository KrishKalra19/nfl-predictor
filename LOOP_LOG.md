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

## 5. Weather — no gain, not adopted in models
- `src/weather.py` pulls kickoff-window weather (4 hours) from Open-Meteo: archive for past games, forecast for the next 15 days, indoor games neutral. Cached in `data/raw/weather_games.csv` (3,430 games). It matches nflverse's recorded temperature at r=0.975 and wind at r=0.66.
- Features: shared wind / cold / precipitation, plus per-team "dome team outdoors in the cold" and wind × team passing EPA. `src/exp_features.py` is a reusable A/B harness.
- Dev ensemble: baseline 0.6233 vs weather 0.6234. No member improved meaningfully. Weather shifts totals, not winners.
- Columns stay in features.csv but out of the model feature lists. The data can feed a site touch later.

## 6. Coaching — no gain, not adopted
- Per-team features from the full 1999+ schedule using only earlier games: new-coach flag (first season with the team), log tenure with the team, and career win rate shrunk toward .500 (16 pseudo-games). The win-rate differential correlates 0.27 with margin on its own.
- Dev ensemble: baseline 0.6233 vs coach 0.6234. Mixed members: logreg 0.6244 and MLP 0.6242 were slightly better, GBM 0.6291 and Siamese 0.6263 worse. Coaching quality is already reflected in team form.
- Candidate features are now fenced out of the production feature sets (`NOT_ADOPTED` in `backtest.py`) until they earn a place, so a new column can't silently enter the live model.

## 7. Kicking quality — no gain, not adopted
- `k_fgoe`: the team's latest kicker's field goals made over expected per attempt. Expected comes from a league distance curve, exponentially weighted, shrunk with 15 pseudo-attempts, using earlier games only. The differential correlates only 0.05 with margin.
- Dev ensemble: 0.6232 vs 0.6233 baseline, below the 0.002 adoption bar. Fenced out.

## 8. Kalman-filter team strength — no gain in the model, kept for the site
- `src/kalman.py`: joint Kalman filters over all 32 teams.
  - Margin model: points, with home-field advantage.
  - EPA model: offense and defense latent states, opponent-adjusted by construction.
  - Weekly drift, offseason regression and widening uncertainty.
  - Snapshots are taken per week before any of that week's results are applied, the same convention as the live picks.
- Signal alone (corr of rating difference with margin, 2016–21): Kalman 0.389, tuned ridge 0.392, Elo 0.379. The tuning surface is flat (0.388–0.389), so defaults are kept.
- Dev ensemble: as an addition 0.6240 (worse; redundant with the ridge ratings). As a replacement for the ridge ratings 0.6236 (a tie, but the Siamese net degrades). Not adopted.
- Kept because each rating comes with an uncertainty (±), which suits the power-rankings page (item 13).
- Fixed the A/B harness so swap candidates exclude features in the candidate run.

## 9. QB model upgrade — near miss, not adopted
- `qbx_rating`: QB EPA per play adjusted for each opponent's pregame pass defense, plus QB age at kickoff (centered at 28, with a squared term). Birth dates come from players.csv.
- Signal alone: qbx_rating 0.343 vs the existing qb_rating 0.343 (the opponent adjustment adds nothing). QB age 0.137.
- Dev ensemble: baseline 0.6233. All qbx features 0.6227 (4 of 5 members improve, but −0.0006 is under the 0.002 bar). Age only 0.6235.
- Not adopted per the ground rules. It's the best candidate so far; revisit if a later change makes it cross the bar.
- Backup-QB prior skipped: QB2 snap shares are tiny and noisy, and the existing draft-based prior already covers unknown starters.

## 10. Playoff simulation — done (site feature; model use skipped)
- `src/playoffs.py` simulates the rest of the season 10,000 times (4 s).
  - Game probabilities: frozen model predictions for this week; Kalman margin ratings with their uncertainty for later weeks.
  - Seeding: 4 division winners plus 3 wild cards per conference. Real NFL tiebreakers aren't modeled; ties are broken at random.
- Writes `docs/data/playoff_odds.json`, regenerated on every site build (wrapped so a failure can't break the site).
- Sanity: playoff, division and #1-seed probabilities sum to exactly 14 / 8 / 2.
- Not tested as a model feature: the crude stakes flags already showed zero value, and a fair backtest would need simulations for every historical week. Odds are rendered on the site in item 15.

## 11. "Why this pick" — done
- `src/explain.py`: the ensemble's logistic member is linear in (team A − team B) feature differences, so its log-odds split exactly into per-feature contributions. ~250 features are summed into 9 readable groups (Quarterback, Roster & injuries, Team strength rating, Offense / Defense / Results this season, Recent form, Special teams, Rest & travel) plus Home field. The top 3 are kept with the team each favors and a strength (slight / moderate / strong by log-odds).
- Shared context (week, division game) is excluded: in the symmetric model it doesn't favor either side.
- `predict.py` stores a `why` JSON column per game. The site shows "Grounds" on each case card. Games frozen before this change show none.
- Example, DAL @ PHI: Roster & injuries → DAL (strong), Offense → DAL (strong), Defense → PHI (moderate).
- Checked that predictions are unchanged and deterministic: DAL @ PHI's committed Sunday prediction matches a fresh run to every digit.

## 12. Model spread line — done
- Each case card shows "Line: Model X −n · Vegas Y −m". The model line converts the ensemble win probability to a home spread with the same 13.45-point normal scale used to turn Vegas spreads into probabilities, rounded to the half point. Exactly 50% shows as "Pick'em".
- I didn't use the ridge-margin member's raw margins: heavy regularization (alpha 10000) shrinks them, which would understate every line.
- Week 7 examples: SF −8 (Vegas −2.5), HOU −7.5 (Vegas −10.5), DAL −3.5 (Vegas −4.5).

## 13 + 15. Client list: power rankings with playoff odds — done
- New page `docs/rankings.html` ("Client list" in the nav).
  - Columns: all 32 teams ranked by Kalman margin rating (points vs an average team) with a ± uncertainty bar, movement since last week, record, offense/defense ranks (Kalman EPA states), starting QB, and simulated odds for playoffs, division and #1 seed (item 10).
  - Item 15's playoff odds table lives here rather than on the home page, to keep the docket uncluttered.
- Shared styling moved to `site/style.css`, which is copied to `docs/style.css`. Both pages link it.
- Added `.claude/launch.json` (gitignored) to serve `docs/` locally on port 8765, because file previews can't load relative stylesheets. Verified both pages there: themed, no overflow, no console errors.
- Current top 3: SEA (+7.8 ±4.1), JAX (+6.3), BUF (+4.7). MIN is 6-0 but 6th by rating (29th offense, 1st defense).

## 14. Weekly recap — done
- "Week N in review" section on the home page, generated only from the live log (frozen pre-kickoff numbers) and results. No hand-written narrative.
- Tiles: model record vs Vegas on the same games, Krish's record that week, biggest upset (lowest Vegas probability for the winner), the model's best call and its worst miss (excluding the upset game so tiles don't repeat).
- Week 6: model 7–6, Vegas 8–5, Krish 5–9. Upset ARI over LA 18-16 (Vegas had ARI at 15%). Best call KC over LAC at 80%.

## 16. Polish — done
- K&A seal favicon (`favicon.svg`), plus Open Graph / Twitter tags and a 1200×630 link-preview card (`og.png`, drawn by `src/make_og.py` from the real held-out numbers) for both pages, so a shared link shows a proper card.
- Accessibility: `<main>` landmark on both pages. Contrast checked: muted text on the panel is about 6.6:1. Existing focus outlines were kept.
- Footer on both pages links to the source code and credits nflverse.
- Checked at 375px on the local server: no page-level horizontal scroll; the rankings table scrolls inside its box.
- Skipped the separate print layout: the frame can't print, and the brief already reads cleanly as a page.

## 17. Tests — done (and they caught a real leak)
- `tests/` (pytest, 15 tests, ~8 s) runs on a synthetic two-season, 32-team league built in `conftest.py`, so no data files are needed (CI-friendly).
  - **Leakage:** for a cutoff week, Elo, ridge power ratings, team form, standings and Kalman ratings must be identical whether or not later results exist. A control test confirms the detector catches a deliberately leaky feature.
  - **Symmetry:** on a neutral field, P(A beats B) + P(B beats A) = 1 for logreg, ridge, GBM, MLP and Siamese. Home field is learned with the right sign.
  - **Freeze:** `predict.apply_freeze` (refactored out of `main`) keeps saved rows for started games only.
- **Leak found:** team-form priors and the Kalman EPA baseline used whole-dataset league means. Fixed with `pregame_means()` (expanding mean over earlier dates) and a running mean in the Kalman filter. Impact on the same 1,218 held-out games: ensemble 0.6242 → 0.6244 log-loss, 66.3% → 66.2%. Dev unchanged (0.6234). REPORT.md has a correction note; originals are kept as `outputs/*_v1_prefix.*`.
- Added `src/export_results.py`, which regenerates the site's result CSVs with the held-out set fixed at 1,218 games (later 2026 games belong to the live log). `make_og.py` reads it.
- Full pipeline re-run end to end (`run_weekly.py --no-download`): 13 started games stayed frozen; DAL @ PHI is DAL 60.1%.

## 18. CI — done
- `.github/workflows/ci.yml`: on every push or PR (skipped for docs/outputs-only commits so the scheduled refreshes don't trigger it), install CPU PyTorch and deps, `ruff check src tests`, then `pytest`. First run green in 66 s. README has a CI badge.
- `pyproject.toml` holds the ruff config (pyflakes + syntax errors, line length 130) and the pytest config. Lint cleanup removed 7 unused imports and one dead variable.

## 19. Repo polish — done
- README rewritten as a landing page: CI and site badges, a results table, a mermaid architecture diagram, "keeping it honest", how to run, and a code map. Numbers come from the regenerated exports.
- `requirements.txt` pinned to tested versions. MIT LICENSE added.
- Removed stale files: `exp_log1.txt`, old `dev_preds.csv`, `backtest_dev_all.json`, and the old claude.ai dashboard sources the site never reads.

## 20. Model card — done
- `MODEL_CARD.md`: purpose, intended and non-intended use (not for wagering), data, model, evaluation (with Brier), calibration, live log, leakage controls, and limitations and known biases.

## 21. Krish + Model blend — done (descriptive)
- Partners section: "Joint counsel" line scoring a 50/50 log-odds average of Krish's and the model's probabilities on Krish's graded games. Week 6: 5–9, log-loss 0.784, against Krish 0.889 and the model 0.702. Labeled as too early to read until about 200 cases.

## 22. Pick analysis by reason tag — built, gated until there's data
- A per-tag table (cases, won–lost, log-loss) under the partners board. A tag appears only after 30+ graded picks, so nothing shows yet (largest tag today: 3 picks). Logic verified on the real picks with the threshold lowered.

## Loop summary (2026-10-26)
- **Shipped to the live site:**
  - live "since launch" record with a downloadable log;
  - "Grounds" explanations on every case;
  - model point spread next to the Vegas line;
  - weekly recap;
  - Client list page with Kalman power rankings (± uncertainty) and Monte Carlo playoff odds;
  - favicon and link-preview card;
  - accessibility and footer polish;
  - the joint-counsel line and gated tag analysis.
- **Engineering:**
  - 15-test pytest suite (leakage, symmetry, freeze) on a synthetic league;
  - GitHub Actions CI (green) with a README badge;
  - README landing page with a mermaid diagram;
  - MODEL_CARD.md, MIT license, pinned requirements;
  - reproducible result exports.
- **Model:**
  - Tested 9 ideas honestly on dev (calibration, stacking, weather, coaching, kicking, Kalman as an addition and as a swap, QB opponent adjustment + age); none cleared the 0.002 bar. Best near miss: QB features at −0.0006.
  - The tests found a real (tiny) leak of whole-dataset league means into early-season priors. Fixed. Corrected held-out numbers: 66.2% / 0.6244 (were 66.3% / 0.6242).
- **Still blocked on Krish:** picks on the website (Supabase project), Andrew's GitHub username, any scheduled-task changes.

## Round 2: the previously untested list (2026-10-26 to 28)
Dev seasons 2018–21. Baseline ensemble 0.6234 log-loss, 0.6187 on regular-season weeks 15–18 (222 games).
Stakes features are judged on weeks 15–18 (agreed with Krish): adopt if those improve and the full season isn't hurt.

| Candidate | Full season | Wk 15–18 | Verdict |
|---|---|---|---|
| **Monte Carlo stakes replacing the standings flags** (`sim_`) | 0.6231 | **0.6159** | **Adopted** |
| Monte Carlo stakes added to the flags | 0.6235 | 0.6152 | Swap is simpler and no worse overall |
| Game-day roster, oracle (players who took a snap) | 0.6200 | 0.6086 | Upper bound only: uses in-game information |
| Game-day roster, realistic (Friday view minus Questionable players who sat) | 0.6233 | 0.6186 | No gain |
| Position-unit matchups (pass/rush/sack/explosive/trench products) | 0.6237 | 0.6189 | No gain |
| Team-specific home field, Denver altitude, 2020 no fans | 0.6237 | 0.6192 | No gain |
| Season down-weighting 0.95/season (re-test on current lineup) | 0.6236 | 0.6191 | No gain (matches the first-round grid) |
| Transformer over each team's last 12 games, added / replacing MLP | 0.6237 / 0.6238 | – | No gain (alone 0.6296) |
| Deeper MLP (128-64), added / replacing MLP | 0.6237 / 0.6238 | – | No gain (alone 0.6273) |

- `sim_stakes` (build_features): 2,000 simulations of the rest of the regular season before every week, from pregame Elo, seeded per (season, week) so backtest and live draws match. Per team: playoff odds and this game's leverage on playoff, division and #1-seed odds (P(x | win) − P(x | loss)). Playoff games are set to full leverage. Leakage test added.
- The oracle vs realistic inactives gap says the game-day gain came from information nobody has before kickoff (in-game injuries, healthy scratches). ESPN's free feed does publish injury inactives (`src/gameday.py`), but they add nothing over Friday's Out/Doubtful list, so they're not wired into the model.
- Candidate prefixes `rsg_`, `rsq_` and the old `stakes_` are fenced in `NOT_ADOPTED`; matchup and home-field columns are side-context and stay out unless listed in `SIDE_CTX`.
- **Longer history (2006+ data, `src/exp_history.py`, built in `data/processed_2006`)**, with the adopted stakes features: dev ensemble 0.6227 (train from 2015 on the longer-seeded table), 0.6224 (2012), 0.6219 (2010), **0.6213 (2007)**. All five members improve and the gain is monotonic in history length. Against the live model's 0.6231 the gain is −0.0018, just under the 0.002 bar: near miss, pending Krish's call. Snap counts start in 2013 (nflverse's 2012 file is empty), so roster features are imputed before then.
- **Post-hoc check on the held-out 2022–26 games (1,218; looked at once, after both changes were decided):** original locked lineup 0.6244 / 66.2%. With the Monte Carlo stakes, training from 2015: 0.6249 / 66.5%. With the stakes and training from 2007: 0.6244 / 66.2%. On these games the stakes swap costs 0.0005 overall and the longer history gains it back; both moves are within noise at this sample size. The live log from 2026 week 6 on remains the real test.
